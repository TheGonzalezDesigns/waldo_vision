// THEORY:
// The `pipeline` module is the final, top-level API for the entire vision engine.
// It encapsulates the full architectural stack (`GridManager`, `BlobDetector`,
// `SceneManager`) into a single, easy-to-use interface.
//
// Its purpose is to provide a clean and user-friendly entry point for processing
// image data and receiving high-level, actionable reports about significant events.
//
// Key architectural principles:
// 1.  **Encapsulation**: The `VisionPipeline` struct owns and manages all the complex
//     underlying components. A user of this library only needs to interact with this
//     single struct.
// 2.  **Dual-Purpose API**: It provides two distinct methods for different use cases:
//     - `significant_mention_detected`: A simple, high-performance boolean check
//       ideal for real-time loops (e.g., WebSocket pipelines) where only a "go/no-go"
//       signal is needed.
//     - `generate_report`: A richer method that provides a detailed `Report` enum,
//       containing the full data of any detected `Moment`s for deeper analysis.
// 3.  **Tunable Configuration**: It uses a `PipelineConfig` struct to hold key
//     thresholds, allowing users to easily tune the sensitivity and behavior of the
//     filter without needing to understand the internal algorithms.
// 4.  **Final Decision Logic**: This is the layer where the final "judgment call" is
//     made. It analyzes the output of the `SceneManager` and uses the configuration
//     to decide what constitutes a "significant mention."

use crate::core_modules::blob_detector::blob_detector;
use crate::core_modules::grid_manager::GridManager;
use crate::core_modules::moment::SceneManager;

// Re-export key data structures for the public API.
pub use crate::core_modules::moment::Moment;
pub use crate::core_modules::smart_chunk::{AnomalyDetails, ChunkStatus};

/// Configuration for the VisionPipeline, allowing for tunable behavior.
#[derive(Debug, Clone)]
pub struct PipelineConfig {
    /// The `GridManager` needs to know the full image dimensions to work correctly.
    pub image_width: u32,
    pub image_height: u32,
    pub chunk_width: u32,
    pub chunk_height: u32,
    /// The `Tracker` needs a threshold for filtering out fleeting moments.
    pub min_moment_age_for_significance: u32,
    /// The `BlobDetector` needs a threshold for anomaly scores.
    pub significance_threshold: f64,
}

/// The detailed data package for a significant event.
#[derive(Debug, Clone)]
pub struct MentionData {
    /// Moments that have just been created in this frame and meet the significance criteria.
    pub new_significant_moments: Vec<Moment>,
    /// Moments that have just completed in this frame and meet the significance criteria.
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
}

impl VisionPipeline {
    /// Creates a new, configured instance of the vision pipeline.
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
        }
    }

    /// Processes a frame and returns a simple boolean indicating if a significant event occurred.
    pub fn significant_mention_detected(&mut self, frame_buffer: &[u8]) -> bool {
        let report = self.generate_report(frame_buffer);
        matches!(report, Report::SignificantMention(_))
    }

    /// Processes a frame and returns a detailed report of all significant events.
    pub fn generate_report(&mut self, frame_buffer: &[u8]) -> Report {
        // Stage 1: Temporal Analysis
        self.last_status_map = self.grid_manager.process_frame(frame_buffer);

        // Stage 2: Spatial Grouping
        let blobs = blob_detector::find_blobs(
            &self.last_status_map,
            self.config.image_width / self.config.chunk_width,
            self.config.image_height / self.config.chunk_height,
        );

        // Stage 3: Behavioral Analysis
        let (newly_started, newly_completed) = self.scene_manager.update(blobs);

        // Stage 4: Final Decision Logic
        let new_significant_moments: Vec<Moment> = newly_started
            .iter()
            .filter(|m| self.is_moment_significant(m))
            .cloned()
            .collect();

        let completed_significant_moments: Vec<Moment> = newly_completed
            .iter()
            .filter(|m| self.is_moment_significant(m))
            .cloned()
            .collect();

        if new_significant_moments.is_empty() && completed_significant_moments.is_empty() {
            Report::NoSignificantMention
        } else {
            Report::SignificantMention(MentionData {
                new_significant_moments,
                completed_significant_moments,
            })
        }
    }

    /// Analyzes a moment based on the pipeline's configuration to determine if it's significant.
    fn is_moment_significant(&self, moment: &Moment) -> bool {
        // Rule 1: The moment must have a minimum duration.
        let age = moment.end_frame - moment.start_frame;
        if age < self.config.min_moment_age_for_significance as u64 {
            return false;
        }

        // Rule 2: At least one blob in the moment's history must have an anomaly score
        // that exceeds our significance threshold.
        moment
            .blob_history
            .iter()
            .any(|blob| blob.average_anomaly.luminance_score >= self.config.significance_threshold)
    }

    /// Returns a slice of the ChunkStatus map from the most recently processed frame.
    pub fn get_last_status_map(&self) -> &[ChunkStatus] {
        &self.last_status_map
    }
}