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
    pub scene_is_stable: bool,
    pub significant_event_count: u64,
}

/// The main, top-level struct for the vision engine.
pub struct VisionPipeline {
    grid_manager: GridManager,
    scene_manager: SceneManager,
    config: PipelineConfig,
    blob_size_history: VecDeque<usize>,
    scene_stability_history: VecDeque<f64>,
    significant_event_count: u64,
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
            scene_stability_history: VecDeque::with_capacity(SCENE_STABILITY_HISTORY_LENGTH),
            significant_event_count: 0,
        }
    }

    /// Processes a frame and returns a comprehensive analysis snapshot.
    pub fn process_frame(&mut self, frame_buffer: &[u8]) -> FrameAnalysis {
        let status_map = self.grid_manager.process_frame(frame_buffer);
        let is_global_disturbance = self.analyze_scene_stability(&status_map);
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
            scene_is_stable: !is_global_disturbance,
            significant_event_count: self.significant_event_count,
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

    /// Analyzes the overall stability of the scene to detect global changes.
    fn analyze_scene_stability(&mut self, status_map: &[ChunkStatus]) -> bool {
        const HYSTERESIS_WINDOW: usize = 5; // Use the average of the last 5 frames for stability checks.

        let num_chunks = status_map.len();
        if num_chunks == 0 { return false; }

        let num_unstable_chunks = status_map.iter().filter(|s| !matches!(s, ChunkStatus::Stable | ChunkStatus::Learning)).count();
        let current_instability = num_unstable_chunks as f64 / num_chunks as f64;

        self.scene_stability_history.push_back(current_instability);
        if self.scene_stability_history.len() > SCENE_STABILITY_HISTORY_LENGTH {
            self.scene_stability_history.pop_front();
        }

        // Wait for the history to be sufficiently populated before making decisions.
        if self.scene_stability_history.len() < SCENE_STABILITY_HISTORY_LENGTH {
            return false;
        }

        // --- Hysteresis Logic ---
        // Calculate the average of the last few frames to smooth out flickering.
        let short_term_history = self.scene_stability_history.iter().rev().take(HYSTERESIS_WINDOW);
        let short_term_avg_instability = short_term_history.sum::<f64>() / HYSTERESIS_WINDOW as f64;

        // Calculate the average of the longer-term history for a stable baseline.
        let long_term_avg_instability: f64 = self.scene_stability_history.iter().sum::<f64>() / self.scene_stability_history.len() as f64;
        
        // A global disturbance is a sustained spike in instability.
        short_term_avg_instability > long_term_avg_instability + self.config.global_disturbance_threshold
    }
}
