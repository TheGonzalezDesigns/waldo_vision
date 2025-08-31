// THEORY:
// The `pipeline` module is the final, top-level API for the entire vision engine.
// It encapsulates the full architectural stack into a single, easy-to-use interface.
// Its purpose is to provide a clean and user-friendly entry point for processing
// image data and receiving high-level, actionable reports about significant events.

use crate::core_modules::blob_detector::blob_detector;
use crate::core_modules::grid_manager::GridManager;
use crate::core_modules::moment::SceneManager;
use crate::core_modules::smart_blob::SmartBlob;
use std::collections::VecDeque;

// Re-export key data structures for the public API.
pub use crate::core_modules::moment::Moment;
pub use crate::core_modules::smart_chunk::{AnomalyDetails, ChunkStatus};
pub use crate::core_modules::tracker::{TrackedBlob, TrackedState};

const BLOB_SIZE_HISTORY_LENGTH: usize = 100;
const SCENE_ENTROPY_HISTORY_LENGTH: usize = 30;
const ENTROPY_SMOOTHING_FACTOR: f64 = 0.1; // Alpha for the EMA. Smaller is smoother.

/// Configuration for the VisionPipeline, allowing for tunable behavior.
#[derive(Debug, Clone)]
pub struct PipelineConfig {
    pub image_width: u32,
    pub image_height: u32,
    pub chunk_width: u32,
    pub chunk_height: u32,
    pub new_age_threshold: u32,
    pub behavioral_anomaly_threshold: f64,
    pub absolute_min_blob_size: usize,
    pub blob_size_std_dev_filter: f64,
    /// The threshold for the *rate of change* of scene entropy to trigger a global disturbance.
    pub global_disturbance_threshold: f64,
}

/// The detailed data package for a significant event.
#[derive(Debug, Clone)]
pub struct MentionData {
    pub new_significant_moments: Vec<Moment>,
    pub completed_significant_moments: Vec<Moment>,
    pub is_global_disturbance: bool,
}

/// The primary output of the vision pipeline for a single frame.
#[derive(Debug, Clone)]
pub enum Report {
    NoSignificantMention,
    SignificantMention(MentionData),
}

/// A snapshot of the pipeline's state for a single frame, used for visualization.
#[derive(Debug, Clone)]
pub struct FrameAnalysis {
    pub report: Report,
    pub status_map: Vec<ChunkStatus>,
    pub tracked_blobs: Vec<TrackedBlob>,
    pub scene_entropy_score: f64,
    pub significant_event_count: u64,
}

/// The main, top-level struct for the vision engine.
pub struct VisionPipeline {
    grid_manager: GridManager,
    scene_manager: SceneManager,
    config: PipelineConfig,
    blob_size_history: VecDeque<usize>,
    significant_event_count: u64,
    /// A smoothed, continuous score representing the scene's stability (0.0 = calm, 1.0 = chaotic).
    scene_entropy_score: f64,
}

impl VisionPipeline {
    pub fn new(config: PipelineConfig) -> Self {
        let grid_manager = GridManager::new(
            config.image_width,
            config.image_height,
            config.chunk_width,
            config.chunk_height,
        );
        Self {
            grid_manager,
            scene_manager: SceneManager::new(),
            config,
            blob_size_history: VecDeque::with_capacity(BLOB_SIZE_HISTORY_LENGTH),
            significant_event_count: 0,
            scene_entropy_score: 0.0, // Start with a perfectly calm scene.
        }
    }

    /// Processes a frame and returns a comprehensive analysis snapshot.
    pub fn process_frame(&mut self, frame_buffer: &[u8]) -> FrameAnalysis {
        let status_map = self.grid_manager.process_frame(frame_buffer);
        let is_global_disturbance = self.analyze_scene_entropy(&status_map);
        
        let raw_blobs = blob_detector::find_blobs(
            &status_map,
            self.config.image_width / self.config.chunk_width,
            self.config.image_height / self.config.chunk_height,
        );
        let filtered_blobs = self.filter_blobs(raw_blobs);
        let (newly_started, newly_completed) = self.scene_manager.update(filtered_blobs, &self.config);

        let new_significant_moments: Vec<Moment> = newly_started.into_iter().filter(|m| m.is_significant).collect();
        let completed_significant_moments: Vec<Moment> = newly_completed.into_iter().filter(|m| m.is_significant).collect();

        let is_significant_frame = !new_significant_moments.is_empty() || !completed_significant_moments.is_empty() || is_global_disturbance;
        if is_significant_frame {
            self.significant_event_count += 1;
        }

        let report = if is_significant_frame {
            Report::SignificantMention(MentionData {
                new_significant_moments,
                completed_significant_moments,
                is_global_disturbance,
            })
        } else {
            Report::NoSignificantMention
        };

        FrameAnalysis {
            report,
            status_map: status_map.to_vec(),
            tracked_blobs: self.scene_manager.get_tracked_blobs().to_vec(),
            scene_entropy_score: self.scene_entropy_score,
            significant_event_count: self.significant_event_count,
        }
    }

    fn filter_blobs(&mut self, blobs: Vec<SmartBlob>) -> Vec<SmartBlob> {
        // ... (filtering logic remains the same)
        blobs // Placeholder
    }

    /// Analyzes scene entropy using an Exponential Moving Average (EMA) to smooth the signal.
    fn analyze_scene_entropy(&mut self, status_map: &[ChunkStatus]) -> bool {
        let num_chunks = status_map.len();
        if num_chunks == 0 { return false; }

        let num_unstable_chunks = status_map.iter().filter(|s| !matches!(s, ChunkStatus::Stable)).count();
        let current_instability = num_unstable_chunks as f64 / num_chunks as f64;

        // Update the smoothed entropy score using EMA.
        let previous_score = self.scene_entropy_score;
        self.scene_entropy_score = (current_instability * ENTROPY_SMOOTHING_FACTOR) + (previous_score * (1.0 - ENTROPY_SMOOTHING_FACTOR));

        // A disturbance is a high rate of change in the entropy score.
        (self.scene_entropy_score - previous_score) > self.config.global_disturbance_threshold
    }
}
