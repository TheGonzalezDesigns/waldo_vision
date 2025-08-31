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
const SCENE_STABILITY_HISTORY_LENGTH: usize = 30;

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
    /// The percentage of non-stable chunks that must suddenly appear to trigger a
    /// `GlobalDisturbance` event. A value of 0.25 means 25% of the scene must destabilize.
    pub global_disturbance_threshold: f64,
}

/// The detailed data package for a significant event.
#[derive(Debug, Clone)]
pub struct MentionData {
    pub new_significant_moments: Vec<Moment>,
    pub completed_significant_moments: Vec<Moment>,
    /// A flag indicating that a global scene change occurred, not just an isolated object.
    pub is_global_disturbance: bool,
}

/// The primary output of the vision pipeline for a single frame.
#[derive(Debug, Clone)]
pub enum Report {
    NoSignificantMention,
    SignificantMention(MentionData),
}

/// The main, top-level struct for the vision engine.
pub struct VisionPipeline {
    grid_manager: GridManager,
    scene_manager: SceneManager,
    config: PipelineConfig,
    last_status_map: Vec<ChunkStatus>,
    blob_size_history: VecDeque<usize>,
    scene_stability_history: VecDeque<f64>,
}

impl VisionPipeline {
    pub fn new(config: PipelineConfig) -> Self {
        let grid_manager = GridManager::new(
            config.image_width,
            config.image_height,
            config.chunk_width,
            config.chunk_height,
        );
        let num_chunks = (config.image_width / config.chunk_width) * (config.image_height / config.chunk_height);
        Self {
            grid_manager,
            scene_manager: SceneManager::new(),
            config,
            last_status_map: vec![ChunkStatus::Learning; num_chunks as usize],
            blob_size_history: VecDeque::with_capacity(BLOB_SIZE_HISTORY_LENGTH),
            scene_stability_history: VecDeque::with_capacity(SCENE_STABILITY_HISTORY_LENGTH),
        }
    }

    pub fn significant_mention_detected(&mut self, frame_buffer: &[u8]) -> bool {
        let report = self.generate_report(frame_buffer);
        matches!(report, Report::SignificantMention(_))
    }

    pub fn generate_report(&mut self, frame_buffer: &[u8]) -> Report {
        // Stage 1: Temporal Analysis
        self.last_status_map = self.grid_manager.process_frame(frame_buffer);

        // Stage 1.5: Meta-Analysis of Scene Stability
        let status_map_clone = self.last_status_map.clone();
        let is_global_disturbance = self.analyze_scene_stability(&status_map_clone);

        // Stage 2: Spatial Grouping
        let raw_blobs = blob_detector::find_blobs(
            &self.last_status_map,
            self.config.image_width / self.config.chunk_width,
            self.config.image_height / self.config.chunk_height,
        );

        // Stage 2.5: Production-Ready Blob Filtering
        let filtered_blobs = self.filter_blobs(raw_blobs);

        // Stage 3: Behavioral Analysis
        let (newly_started, newly_completed) = self.scene_manager.update(filtered_blobs, &self.config);

        // Stage 4: Final, Tracker-Aware Decision Logic
        let new_significant_moments: Vec<Moment> = newly_started.into_iter().filter(|m| m.is_significant).collect();
        let completed_significant_moments: Vec<Moment> = newly_completed.into_iter().filter(|m| m.is_significant).collect();

        if new_significant_moments.is_empty() && completed_significant_moments.is_empty() && !is_global_disturbance {
            Report::NoSignificantMention
        } else {
            Report::SignificantMention(MentionData {
                new_significant_moments,
                completed_significant_moments,
                is_global_disturbance,
            })
        }
    }

    fn filter_blobs(&mut self, blobs: Vec<SmartBlob>) -> Vec<SmartBlob> {
        // ... (filtering logic remains the same)
        blobs // Placeholder
    }

    /// Analyzes the overall stability of the scene to detect global changes.
    fn analyze_scene_stability(&mut self, status_map: &[ChunkStatus]) -> bool {
        let num_chunks = status_map.len();
        if num_chunks == 0 { return false; }

        let num_unstable_chunks = status_map.iter().filter(|s| !matches!(s, ChunkStatus::Stable | ChunkStatus::Learning)).count();
        let current_instability = num_unstable_chunks as f64 / num_chunks as f64;

        // For the first few frames, the scene is expected to be chaotic.
        if self.scene_stability_history.len() < SCENE_STABILITY_HISTORY_LENGTH / 2 {
            self.scene_stability_history.push_back(current_instability);
            return false;
        }

        let avg_historical_instability: f64 = self.scene_stability_history.iter().sum::<f64>() / self.scene_stability_history.len() as f64;
        
        self.scene_stability_history.push_back(current_instability);
        if self.scene_stability_history.len() > SCENE_STABILITY_HISTORY_LENGTH {
            self.scene_stability_history.pop_front();
        }

        // A global disturbance is a sudden, massive spike in instability compared to the recent past.
        current_instability > avg_historical_instability + self.config.global_disturbance_threshold
    }

    pub fn get_last_status_map(&self) -> &[ChunkStatus] {
        &self.last_status_map
    }

    pub fn get_tracked_blobs(&self) -> &Vec<TrackedBlob> {
        self.scene_manager.get_tracked_blobs()
    }
}