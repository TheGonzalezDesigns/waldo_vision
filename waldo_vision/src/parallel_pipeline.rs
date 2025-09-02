use crate::core_modules::{
    grid_manager::GridManager,
    tracker::TrackedBlob,
};
use crate::pipeline::{PipelineConfig, FrameAnalysis, SceneState, Report};
use std::sync::{Arc, Mutex};
use std::collections::{VecDeque, HashMap};
use tokio::sync::{mpsc, oneshot, RwLock};
use std::time::Instant;

const FRAME_POOL_SIZE: usize = 8;
const WORKER_POOL_SIZE: usize = 4;

#[derive(Clone)]
pub struct FrameBuffer {
    pub data: Vec<u8>,
    pub frame_id: u64,
    pub timestamp: Instant,
}

#[derive(Clone)]
pub struct TemporalState {
    pub tracked_blobs: Vec<TrackedBlob>,
    pub scene_state: SceneState,
    pub significant_event_count: u64,
    pub frames_in_current_state: u32,
}

pub struct FrameTask {
    pub frame_buffer: FrameBuffer,
    pub temporal_state: TemporalState,
    pub result_sender: oneshot::Sender<(FrameAnalysis, TemporalState)>,
}

pub struct WorkerPool {
    task_sender: mpsc::UnboundedSender<FrameTask>,
    workers: Vec<tokio::task::JoinHandle<()>>,
}

impl WorkerPool {
    pub fn new(config: PipelineConfig) -> Self {
        let (task_sender, mut task_receiver) = mpsc::unbounded_channel::<FrameTask>();
        let mut workers = Vec::new();

        // Create a single dispatcher that distributes tasks to workers
        let (worker_senders, worker_receivers): (Vec<_>, Vec<_>) = (0..WORKER_POOL_SIZE)
            .map(|_| mpsc::unbounded_channel::<FrameTask>())
            .unzip();

        // Spawn dispatcher
        let dispatcher_senders = worker_senders;
        tokio::spawn(async move {
            let mut worker_idx = 0;
            while let Some(task) = task_receiver.recv().await {
                let _ = dispatcher_senders[worker_idx].send(task);
                worker_idx = (worker_idx + 1) % WORKER_POOL_SIZE;
            }
        });

        // Spawn workers
        for mut worker_receiver in worker_receivers {
            let worker_config = config.clone();
            
            let worker = tokio::spawn(async move {
                let mut grid_manager = GridManager::new(
                    worker_config.image_width,
                    worker_config.image_height,
                    worker_config.chunk_width,
                    worker_config.chunk_height,
                );
                
                while let Some(task) = worker_receiver.recv().await {
                    let analysis = Self::process_frame_worker(
                        &mut grid_manager,
                        &task.frame_buffer,
                        &task.temporal_state,
                        &worker_config,
                    ).await;
                    
                    let _ = task.result_sender.send(analysis);
                }
            });
            
            workers.push(worker);
        }

        Self {
            task_sender,
            workers,
        }
    }

    async fn process_frame_worker(
        grid_manager: &mut GridManager,
        frame_buffer: &FrameBuffer,
        temporal_state: &TemporalState,
        config: &PipelineConfig,
    ) -> (FrameAnalysis, TemporalState) {
        let status_map = grid_manager.process_frame(&frame_buffer.data).await;
        
        // Create analysis with temporal state
        let analysis = FrameAnalysis {
            report: Report::NoSignificantMention,
            status_map: status_map,
            tracked_blobs: temporal_state.tracked_blobs.clone(),
            scene_state: temporal_state.scene_state.clone(),
            significant_event_count: temporal_state.significant_event_count,
        };

        let new_temporal_state = TemporalState {
            tracked_blobs: temporal_state.tracked_blobs.clone(),
            scene_state: temporal_state.scene_state.clone(),
            significant_event_count: temporal_state.significant_event_count,
            frames_in_current_state: temporal_state.frames_in_current_state + 1,
        };

        (analysis, new_temporal_state)
    }

    pub async fn process_frame(
        &self,
        frame_buffer: FrameBuffer,
        temporal_state: TemporalState,
    ) -> Result<(FrameAnalysis, TemporalState), &'static str> {
        let (result_sender, result_receiver) = oneshot::channel();
        
        let task = FrameTask {
            frame_buffer,
            temporal_state,
            result_sender,
        };

        self.task_sender.send(task)
            .map_err(|_| "Failed to send task to worker pool")?;

        result_receiver.await
            .map_err(|_| "Failed to receive result from worker")
    }
}

pub struct ParallelPipeline {
    config: PipelineConfig,
    worker_pool: WorkerPool,
    frame_buffer_pool: Arc<Mutex<VecDeque<Vec<u8>>>>,
    temporal_state_channel: Arc<RwLock<TemporalState>>,
    frame_counter: Arc<Mutex<u64>>,
    pending_frames: Arc<Mutex<HashMap<u64, (FrameAnalysis, TemporalState)>>>,
    next_expected_frame: Arc<Mutex<u64>>,
}

impl ParallelPipeline {
    pub fn new(config: PipelineConfig) -> Self {
        let worker_pool = WorkerPool::new(config.clone());
        
        let mut frame_pool = VecDeque::with_capacity(FRAME_POOL_SIZE);
        for _ in 0..FRAME_POOL_SIZE {
            let buffer = vec![0u8; (config.image_width * config.image_height * 3) as usize];
            frame_pool.push_back(buffer);
        }

        let initial_temporal_state = TemporalState {
            tracked_blobs: Vec::new(),
            scene_state: SceneState::Calibrating,
            significant_event_count: 0,
            frames_in_current_state: 0,
        };

        Self {
            config,
            worker_pool,
            frame_buffer_pool: Arc::new(Mutex::new(frame_pool)),
            temporal_state_channel: Arc::new(RwLock::new(initial_temporal_state)),
            frame_counter: Arc::new(Mutex::new(0)),
            pending_frames: Arc::new(Mutex::new(HashMap::new())),
            next_expected_frame: Arc::new(Mutex::new(0)),
        }
    }

    pub async fn process_frame(&self, frame_data: &[u8]) -> Result<FrameAnalysis, &'static str> {
        let frame_id = {
            let mut counter = self.frame_counter.lock().unwrap();
            let id = *counter;
            *counter += 1;
            id
        };

        let frame_buffer = self.get_frame_buffer(frame_data, frame_id)?;
        let temporal_state = self.temporal_state_channel.read().await.clone();

        let (analysis, new_temporal_state) = self.worker_pool
            .process_frame(frame_buffer, temporal_state)
            .await?;

        {
            let mut pending = self.pending_frames.lock().unwrap();
            pending.insert(frame_id, (analysis.clone(), new_temporal_state));
        }

        self.try_update_temporal_state().await;
        
        Ok(analysis)
    }

    async fn try_update_temporal_state(&self) {
        let next_frame_id = {
            let next_expected = self.next_expected_frame.lock().unwrap();
            *next_expected
        };

        let pending_result = {
            let mut pending = self.pending_frames.lock().unwrap();
            pending.remove(&next_frame_id)
        };

        if let Some((_, new_temporal_state)) = pending_result {
            let mut temporal_state = self.temporal_state_channel.write().await;
            *temporal_state = new_temporal_state;

            {
                let mut next_expected = self.next_expected_frame.lock().unwrap();
                *next_expected += 1;
            }
        }
    }

    fn get_frame_buffer(&self, frame_data: &[u8], frame_id: u64) -> Result<FrameBuffer, &'static str> {
        let mut buffer = {
            let mut pool = self.frame_buffer_pool.lock().unwrap();
            pool.pop_front().unwrap_or_else(|| {
                vec![0u8; frame_data.len()]
            })
        };

        // Resize buffer if needed
        if buffer.len() != frame_data.len() {
            buffer.resize(frame_data.len(), 0);
        }

        buffer.copy_from_slice(frame_data);

        Ok(FrameBuffer {
            data: buffer,
            frame_id,
            timestamp: Instant::now(),
        })
    }

    pub fn return_frame_buffer(&self, buffer: Vec<u8>) {
        let mut pool = self.frame_buffer_pool.lock().unwrap();
        if pool.len() < FRAME_POOL_SIZE {
            pool.push_back(buffer);
        }
    }
}