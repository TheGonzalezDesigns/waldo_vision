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
}

/// The detailed data package for a significant event.
#[derive(Debug, Clone)]
pub struct MentionData {
    pub new_significant_moments: Vec<Moment>,
    pub completed_significant_moments: Vec<Moment>,
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
        }
    }

    pub fn significant_mention_detected(&mut self, frame_buffer: &[u8]) -> bool {
        let report = self.generate_report(frame_buffer);
        matches!(report, Report::SignificantMention(_))
    }

    pub fn generate_report(&mut self, frame_buffer: &[u8]) -> Report {
        self.last_status_map = self.grid_manager.process_frame(frame_buffer);
        let raw_blobs = blob_detector::find_blobs(
            &self.last_status_map,
            self.config.image_width / self.config.chunk_width,
            self.config.image_height / self.config.chunk_height,
        );
        let filtered_blobs = self.filter_blobs(raw_blobs);
        let (newly_started, newly_completed) = self.scene_manager.update(filtered_blobs, &self.config);

        let new_significant_moments: Vec<Moment> = newly_started.into_iter().filter(|m| m.is_significant).collect();
        let completed_significant_moments: Vec<Moment> = newly_completed.into_iter().filter(|m| m.is_significant).collect();

        if new_significant_moments.is_empty() && completed_significant_moments.is_empty() {
            Report::NoSignificantMention
        } else {
            Report::SignificantMention(MentionData {
                new_significant_moments,
                completed_significant_moments,
            })
        }
    }

    fn filter_blobs(&mut self, blobs: Vec<SmartBlob>) -> Vec<SmartBlob> {
        let (mean, std_dev) = {
            if self.blob_size_history.is_empty() { (0.0, 0.0) } else {
                let sum: usize = self.blob_size_history.iter().sum();
                let mean = sum as f64 / self.blob_size_history.len() as f64;
                let variance = self.blob_size_history.iter()
                    .map(|value| (*value as f64 - mean).powi(2))
                    .sum::<f64>() / self.blob_size_history.len() as f64;
                (mean, variance.sqrt())
            }
        };

        let mut filtered_blobs = Vec::new();
        for blob in blobs {
            if blob.size_in_chunks < self.config.absolute_min_blob_size { continue; }
            if self.blob_size_history.len() >= BLOB_SIZE_HISTORY_LENGTH / 2 {
                let threshold = mean - self.config.blob_size_std_dev_filter * std_dev;
                if (blob.size_in_chunks as f64) < threshold { continue; }
            }
            filtered_blobs.push(blob);
        }

        for blob in &filtered_blobs {
            if self.blob_size_history.len() >= BLOB_SIZE_HISTORY_LENGTH {
                self.blob_size_history.pop_front();
            }
            self.blob_size_history.push_back(blob.size_in_chunks);
        }
        filtered_blobs
    }

    pub fn get_last_status_map(&self) -> &[ChunkStatus] {
        &self.last_status_map
    }

    pub fn get_tracked_blobs(&self) -> &Vec<TrackedBlob> {
        self.scene_manager.get_tracked_blobs()
    }
}
