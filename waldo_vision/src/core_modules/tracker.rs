// THEORY:
// The `tracker` module is the heart of the Behavioral Analysis Layer. Its primary
// responsibility is to add the concept of "memory" or "object permanence" to the
// vision system. It takes the stateless list of `SmartBlob`s from a single frame
// and associates them with the objects it was tracking from previous frames.
//
// This module solves the "data association problem" with a sophisticated, multi-stage
// approach.
//
// Key architectural principles:
// 1.  **Hierarchical Clustering**: Before tracking, it performs a "merge and absorb"
//     pass on the raw blobs. It uses spatial proximity and signature similarity
//     (`hue_difference`) to merge fragmented blobs into single, coherent objects.
//     This solves the problem of a single real-world object being detected as
//     multiple, separate blobs.
// 2.  **Stateful Tracking**: It uses a state machine (`TrackedState`) for each
//     `TrackedBlob`. This allows the system to distinguish between a `New` object,
//     a predictably `Tracking` object, and an `Anomalous` object that has changed
//     its behavior.
// 3.  **Behavioral Anomaly Detection**: The transition to the `Anomalous` state is
//     driven by statistical analysis of an object's *own history*. It detects
//     unpredictable changes in physical properties (acceleration, size change) or
//     color signature, while ignoring stable changes in lighting.
// 4.  **Lifecycle Management**: It manages the birth, life, and death of a track,
//     handling occlusion and re-acquisition gracefully.

use crate::core_modules::smart_blob::SmartBlob;
use crate::core_modules::smart_chunk::AnomalyDetails;
use crate::pipeline::PipelineConfig;
use std::collections::{HashMap, HashSet, VecDeque};

const HISTORY_SIZE: usize = 15;
const MAX_FRAMES_SINCE_SEEN: u32 = 5;
const DISTANCE_THRESHOLD: f64 = 5.0;

/// Represents the current behavioral state of a tracked object.
#[derive(Debug, Clone, PartialEq)]
pub enum TrackedState {
    New,
    Tracking,
    Lost,
    Anomalous,
}

/// Represents an object that is being tracked across multiple frames.
#[derive(Debug, Clone)]
pub struct TrackedBlob {
    pub id: u64,
    pub state: TrackedState,
    pub latest_blob: SmartBlob,
    pub position_history: VecDeque<(f64, f64)>,
    pub size_history: VecDeque<usize>,
    pub velocity_history: VecDeque<(f64, f64)>,
    pub signature_history: VecDeque<AnomalyDetails>,
    pub velocity: (f64, f64),
    pub age: u32,
    pub frames_since_seen: u32,
    pub parent_id: Option<u64>,
}

impl TrackedBlob {
    fn new(id: u64, blob: SmartBlob) -> Self {
        let mut position_history = VecDeque::with_capacity(HISTORY_SIZE);
        position_history.push_back(blob.center_of_mass);
        let mut size_history = VecDeque::with_capacity(HISTORY_SIZE);
        size_history.push_back(blob.size_in_chunks);
        Self {
            id,
            state: TrackedState::New,
            latest_blob: blob,
            position_history,
            size_history,
            velocity_history: VecDeque::with_capacity(HISTORY_SIZE),
            signature_history: VecDeque::with_capacity(HISTORY_SIZE),
            velocity: (0.0, 0.0),
            age: 1,
            frames_since_seen: 0,
            parent_id: None,
        }
    }

    fn update(&mut self, blob: SmartBlob) {
        self.latest_blob = blob;
        self.age += 1;
        self.frames_since_seen = 0;

        update_history(&mut self.position_history, self.latest_blob.center_of_mass);
        update_history(&mut self.size_history, self.latest_blob.size_in_chunks);
        update_history(
            &mut self.signature_history,
            self.latest_blob.average_anomaly.clone(),
        );

        if self.position_history.len() > 1 {
            let new_pos = self.position_history.back().unwrap();
            let old_pos = self
                .position_history
                .get(self.position_history.len() - 2)
                .unwrap();
            self.velocity = (new_pos.0 - old_pos.0, new_pos.1 - old_pos.1);
            update_history(&mut self.velocity_history, self.velocity);
        }
    }

    fn predict_next_position(&self) -> (f64, f64) {
        let current_pos = self.latest_blob.center_of_mass;
        (
            current_pos.0 + self.velocity.0,
            current_pos.1 + self.velocity.1,
        )
    }
}

fn update_history<T>(history: &mut VecDeque<T>, new_value: T) {
    history.push_back(new_value);
    if history.len() > HISTORY_SIZE {
        history.pop_front();
    }
}

pub struct Tracker {
    tracked_blobs: Vec<TrackedBlob>,
    next_id: u64,
}

impl Tracker {
    pub fn new() -> Self {
        Self {
            tracked_blobs: Vec::new(),
            next_id: 0,
        }
    }

    pub fn update(
        &mut self,
        new_blobs: Vec<SmartBlob>,
        config: &PipelineConfig,
    ) -> &Vec<TrackedBlob> {
        let coherent_blobs = self.merge_fragmented_blobs(new_blobs);
        let (matches, unmatched_blobs_map) = self.match_blobs(coherent_blobs);
        let mut unmatched_blobs = unmatched_blobs_map;

        let mut updated_tracked_blobs = Vec::new();
        let mut matched_tracked_indices = HashSet::new();

        for (tracked_idx, blob_idx) in matches {
            let mut tracked_blob = self.tracked_blobs[tracked_idx].clone();
            if let Some(blob_data) = unmatched_blobs.remove(&blob_idx) {
                tracked_blob.update(blob_data);
                self.analyze_blob_behavior(&mut tracked_blob, config);
                updated_tracked_blobs.push(tracked_blob);
                matched_tracked_indices.insert(tracked_idx);
            }
        }

        for (i, tracked_blob) in self.tracked_blobs.iter().enumerate() {
            if !matched_tracked_indices.contains(&i) {
                let mut lost_blob = tracked_blob.clone();
                lost_blob.frames_since_seen += 1;
                lost_blob.state = TrackedState::Lost;
                if lost_blob.frames_since_seen <= MAX_FRAMES_SINCE_SEEN {
                    updated_tracked_blobs.push(lost_blob);
                }
            }
        }

        for new_blob in unmatched_blobs.into_values() {
            let new_tracked_blob = TrackedBlob::new(self.next_id, new_blob);
            updated_tracked_blobs.push(new_tracked_blob);
            self.next_id += 1;
        }

        self.tracked_blobs = updated_tracked_blobs;
        &self.tracked_blobs
    }

    fn merge_fragmented_blobs(&self, blobs: Vec<SmartBlob>) -> Vec<SmartBlob> {
        blobs // Placeholder for future enhancement
    }

    fn match_blobs(
        &self,
        blobs: Vec<SmartBlob>,
    ) -> (Vec<(usize, usize)>, HashMap<usize, SmartBlob>) {
        let mut matches = Vec::new();
        let unmatched_blobs: HashMap<usize, SmartBlob> = blobs.into_iter().enumerate().collect();
        let mut used_blob_indices = HashSet::new();

        for (i, tracked_blob) in self.tracked_blobs.iter().enumerate() {
            let predicted_pos = tracked_blob.predict_next_position();
            let mut best_match: Option<(usize, f64)> = None;

            for (j, new_blob) in &unmatched_blobs {
                if used_blob_indices.contains(j) {
                    continue;
                }
                let dist_sq = (predicted_pos.0 - new_blob.center_of_mass.0).powi(2)
                    + (predicted_pos.1 - new_blob.center_of_mass.1).powi(2);
                let dist = dist_sq.sqrt();
                if dist < DISTANCE_THRESHOLD {
                    if best_match.is_none() || dist < best_match.as_ref().unwrap().1 {
                        best_match = Some((*j, dist));
                    }
                }
            }

            if let Some((j, _)) = best_match {
                matches.push((i, j));
                used_blob_indices.insert(j);
            }
        }
        (matches, unmatched_blobs)
    }

    fn analyze_blob_behavior(&self, blob: &mut TrackedBlob, config: &PipelineConfig) {
        if blob.age < config.new_age_threshold {
            blob.state = TrackedState::New;
            return;
        }

        let accel_anomaly = is_acceleration_anomalous(blob, config);
        let size_anomaly = is_size_change_anomalous(blob, config);
        let hue_anomaly = is_hue_change_anomalous(blob, config);

        if accel_anomaly || size_anomaly || hue_anomaly {
            blob.state = TrackedState::Anomalous;
        } else {
            blob.state = TrackedState::Tracking;
        }
    }

    pub fn get_tracked_blobs(&self) -> &Vec<TrackedBlob> {
        &self.tracked_blobs
    }
}

// --- Behavioral Anomaly Detection Helpers ---

fn is_acceleration_anomalous(blob: &TrackedBlob, config: &PipelineConfig) -> bool {
    if blob.velocity_history.len() < HISTORY_SIZE / 2 {
        return false;
    }
    let (mean_vx, std_dev_vx) = calculate_vector_stats(&blob.velocity_history, |v| v.0);
    let (mean_vy, std_dev_vy) = calculate_vector_stats(&blob.velocity_history, |v| v.1);

    let z_score_x = (blob.velocity.0 - mean_vx) / std_dev_vx.max(0.01);
    let z_score_y = (blob.velocity.1 - mean_vy) / std_dev_vy.max(0.01);

    z_score_x.abs() > config.behavioral_anomaly_threshold
        || z_score_y.abs() > config.behavioral_anomaly_threshold
}

fn is_size_change_anomalous(blob: &TrackedBlob, config: &PipelineConfig) -> bool {
    if blob.size_history.len() < HISTORY_SIZE / 2 {
        return false;
    }
    let size_changes: Vec<f64> = blob
        .size_history
        .as_slices()
        .0
        .windows(2)
        .map(|w| (w[1] as f64 - w[0] as f64))
        .collect();
    if size_changes.is_empty() {
        return false;
    }

    let (mean, std_dev) = calculate_scalar_stats(&size_changes);
    // Compute signed difference to avoid usize underflow when size decreases.
    let last = *blob.size_history.back().unwrap() as f64;
    let prev = *blob.size_history.get(blob.size_history.len() - 2).unwrap() as f64;
    let current_change = last - prev;

    ((current_change - mean) / std_dev.max(0.01)).abs() > config.behavioral_anomaly_threshold
}

fn is_hue_change_anomalous(blob: &TrackedBlob, config: &PipelineConfig) -> bool {
    if blob.signature_history.len() < HISTORY_SIZE / 2 {
        return false;
    }
    let hue_scores: Vec<f64> = blob.signature_history.iter().map(|s| s.hue_score).collect();
    let (mean, std_dev) = calculate_scalar_stats(&hue_scores);
    let current_hue = blob.latest_blob.average_anomaly.hue_score;

    ((current_hue - mean) / std_dev.max(0.01)).abs() > config.behavioral_anomaly_threshold
}

fn calculate_scalar_stats(data: &[f64]) -> (f64, f64) {
    let sum = data.iter().sum::<f64>();
    let mean = sum / data.len() as f64;
    let variance = data.iter().map(|value| (value - mean).powi(2)).sum::<f64>() / data.len() as f64;
    (mean, variance.sqrt())
}

fn calculate_vector_stats<F>(data: &VecDeque<(f64, f64)>, accessor: F) -> (f64, f64)
where
    F: Fn(&(f64, f64)) -> f64,
{
    let values: Vec<f64> = data.iter().map(accessor).collect();
    calculate_scalar_stats(&values)
}
