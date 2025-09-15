// THEORY:
// The `SmartChunk` is the heart of the temporal analysis layer. It is a stateful,
// learning entity responsible for a single, fixed region of the image grid. Its job
// is to observe the stream of incoming `Chunk` data for its location over time and
// determine if the current change is "normal" or "anomalous."
//
// Key architectural principles:
// 1.  **Stateful, Multi-Channel Memory**: A `SmartChunk` holds its state between
//     frames. It maintains parallel histories for multiple dimensions of change
//     (luminance, color, hue), allowing for a rich, nuanced understanding of
//     how its region is behaving.
// 2.  **Adaptive Learning**: It uses these histories to statistically learn the "normal"
//     behavior for its specific patch of the world. It calculates means and standard
//     deviations for each channel of change, creating adaptive thresholds.
// 3.  **Temporal Focus**: Its analysis is purely temporal. It knows *when* something
//     unusual is happening but knows nothing about its neighbors. It provides the
//     foundational "sensory input" for the higher-level `SmartBlob` detector.
// 4.  **Rich Data Provider**: Its primary role is to produce a rich "signature" of any
//     anomaly. While it uses luminance as the primary trigger for an event, it enriches
//     that event with statistical scores from all other tracked dimensions, allowing
//     higher-level modules to make more intelligent decisions.

use crate::core_modules::D1::pixel::pixel::Pixel;
use crate::core_modules::chunk::chunk::Chunk;
use crate::core_modules::smart_pixel::smart_pixel::{HueDifference, LuminanceDelta, SmartPixel};
use std::collections::VecDeque;

const HISTORY_WINDOW_SIZE: usize = 20;
const ANOMALY_THRESHOLD_STD_DEV: f64 = 3.0;
const STABLE_LUMINANCE_THRESHOLD: f64 = 2.0;

/// Holds the multi-dimensional signature of a detected anomaly.
/// Each field represents the statistical significance (Z-score) of the change.
#[derive(Debug, Clone, PartialEq)]
pub struct AnomalyDetails {
    /// The significance of the change in brightness.
    pub luminance_score: f64,
    /// The significance of the change in overall color energy (sum of RGB).
    pub color_score: f64,
    /// The significance of the change in the color's hue or balance.
    pub hue_score: f64,
}

/// Represents the current state of a SmartChunk based on its temporal analysis.
#[derive(Debug, Clone, PartialEq)]
pub enum ChunkStatus {
    /// The chunk is new and gathering initial data before analysis can begin.
    Learning,
    /// The chunk's rate of change is below the noise threshold.
    Stable,
    /// The chunk's change is significant but statistically consistent with recent motion.
    PredictableMotion,
    /// The chunk's change is a statistical outlier from its learned behavior.
    AnomalousEvent(AnomalyDetails),
}

/// A stateful analyzer for a single chunk location in an image grid.
pub struct SmartChunk {
    // --- Identity ---
    /// The column index of this chunk in the main grid.
    pub chunk_x: u32,
    /// The row index of this chunk in the main grid.
    pub chunk_y: u32,

    // --- Temporal History ---
    /// A sliding window of the average `Pixel` value for this chunk's location over the last N frames.
    average_pixel_history: VecDeque<Pixel>,
    /// A sliding window of the calculated luminance difference between frames.
    luminance_delta_history: VecDeque<LuminanceDelta>,
    /// A sliding window of the calculated color difference between frames.
    color_delta_history: VecDeque<f64>,
    /// A sliding window of the calculated hue difference between frames.
    hue_difference_history: VecDeque<HueDifference>,

    // --- Learned State (Published for advanced analysis) ---
    /// The learned average (mean) change in luminance for this chunk.
    pub mean_luminance_delta: f64,
    /// The learned standard deviation of the change in luminance.
    pub std_dev_luminance_delta: f64,
    /// The learned average (mean) change in color sum for this chunk.
    pub mean_color_delta: f64,
    /// The learned standard deviation of the change in color sum.
    pub std_dev_color_delta: f64,
    /// The learned average (mean) change in hue for this chunk.
    pub mean_hue_difference: f64,
    /// The learned standard deviation of the change in hue.
    pub std_dev_hue_difference: f64,

    // --- Current Status ---
    /// The current calculated status of this chunk.
    pub status: ChunkStatus,
}

impl SmartChunk {
    pub fn new(chunk_x: u32, chunk_y: u32) -> Self {
        Self {
            chunk_x,
            chunk_y,
            average_pixel_history: VecDeque::with_capacity(HISTORY_WINDOW_SIZE + 1),
            luminance_delta_history: VecDeque::with_capacity(HISTORY_WINDOW_SIZE),
            color_delta_history: VecDeque::with_capacity(HISTORY_WINDOW_SIZE),
            hue_difference_history: VecDeque::with_capacity(HISTORY_WINDOW_SIZE),
            mean_luminance_delta: 0.0,
            std_dev_luminance_delta: 0.0,
            mean_color_delta: 0.0,
            std_dev_color_delta: 0.0,
            mean_hue_difference: 0.0,
            std_dev_hue_difference: 0.0,
            status: ChunkStatus::Learning,
        }
    }

    pub fn update(&mut self, new_chunk: &Chunk) {
        let new_average_pixel = new_chunk.average_pixel();

        if let Some(previous_pixel) = self.average_pixel_history.back() {
            let smart_new = SmartPixel::new(new_average_pixel.clone());
            let smart_prev = SmartPixel::new(previous_pixel.clone());

            let new_lum_delta = smart_new.delta_luminance(&smart_prev);
            let new_col_delta = smart_new.delta_color(&smart_prev);
            let new_hue_diff = smart_new.hue_difference(&smart_prev);

            Self::update_history_generic(&mut self.luminance_delta_history, new_lum_delta);
            Self::update_history_generic(&mut self.color_delta_history, new_col_delta as f64);
            Self::update_history_generic(&mut self.hue_difference_history, new_hue_diff);

            if self.luminance_delta_history.len() >= HISTORY_WINDOW_SIZE {
                self.recalculate_statistics();
                self.analyze_status(new_lum_delta, new_col_delta as f64, new_hue_diff);
            }
        }

        Self::update_history_generic(&mut self.average_pixel_history, new_average_pixel);
    }

    fn update_history_generic<T>(history: &mut VecDeque<T>, new_value: T) {
        history.push_back(new_value);
        if history.len() > HISTORY_WINDOW_SIZE {
            history.pop_front();
        }
    }

    fn recalculate_statistics(&mut self) {
        (self.mean_luminance_delta, self.std_dev_luminance_delta) =
            Self::calculate_stats_for_history(&self.luminance_delta_history);
        (self.mean_color_delta, self.std_dev_color_delta) =
            Self::calculate_stats_for_history(&self.color_delta_history);
        (self.mean_hue_difference, self.std_dev_hue_difference) =
            Self::calculate_stats_for_history(&self.hue_difference_history);
    }

    fn calculate_stats_for_history(history: &VecDeque<f64>) -> (f64, f64) {
        let count = history.len() as f64;
        if count < 1.0 {
            return (0.0, 0.0);
        }
        let sum: f64 = history.iter().sum();
        let mean = sum / count;
        let variance = history.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / count;
        (mean, variance.sqrt())
    }

    /// Analyzes the latest deltas to set the chunk's status.
    fn analyze_status(&mut self, new_lum_delta: f64, new_col_delta: f64, new_hue_diff: f64) {
        if new_lum_delta < STABLE_LUMINANCE_THRESHOLD {
            self.status = ChunkStatus::Stable;
            return;
        }

        let lum_score = Self::calculate_significance_score(
            new_lum_delta,
            self.mean_luminance_delta,
            self.std_dev_luminance_delta,
        );

        if lum_score > ANOMALY_THRESHOLD_STD_DEV {
            // Primary trigger fired. Now enrich with other scores.
            let col_score = Self::calculate_significance_score(
                new_col_delta,
                self.mean_color_delta,
                self.std_dev_color_delta,
            );
            let hue_score = Self::calculate_significance_score(
                new_hue_diff,
                self.mean_hue_difference,
                self.std_dev_hue_difference,
            );

            self.status = ChunkStatus::AnomalousEvent(AnomalyDetails {
                luminance_score: lum_score,
                color_score: col_score,
                hue_score: hue_score,
            });
        } else {
            self.status = ChunkStatus::PredictableMotion;
        }
    }

    /// Generic helper to calculate a significance score (Z-score).
    fn calculate_significance_score(value: f64, mean: f64, std_dev: f64) -> f64 {
        if std_dev < 1e-6 {
            // If history was static, any change is infinitely significant. Return a high, stable score.
            return ANOMALY_THRESHOLD_STD_DEV * 2.0;
        }
        (value - mean) / std_dev
    }
}
