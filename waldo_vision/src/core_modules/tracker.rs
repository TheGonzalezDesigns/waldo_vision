// THEORY:
// The `tracker` module is the heart of the Behavioral Analysis Layer. Its primary
// responsibility is to add the concept of "memory" or "object permanence" to the
// vision system. It takes the stateless list of `SmartBlob`s from a single frame
// and associates them with the objects it was tracking from previous frames.
//
// This module solves the "data association problem."
//
// Key architectural principles:
// 1.  **Object Persistence**: It introduces a new stateful object, the `TrackedBlob`,
//     which represents a single object's existence *over time*. This is distinct
//     from a `SmartBlob`, which is a snapshot in a single frame.
// 2.  **Tracking Logic**: The `Tracker` is the engine that manages a list of
//     `TrackedBlob`s. Its core `update` method implements a tracking algorithm
//     (e.g., nearest neighbor based on predicted position) to match new detections
//     to existing tracks.
// 3.  **Lifecycle Management**: The `Tracker` is responsible for the entire lifecycle
//     of an object:
//     - **Birth**: When a new `SmartBlob` appears that cannot be matched, a new
//       `TrackedBlob` is born.
//     - **Tracking**: When a match is found, the `TrackedBlob`'s state (position,
//       velocity, age) is updated.
//     - **Death/Occlusion**: When a `TrackedBlob` is not seen for a certain number
//       of frames, it is considered "lost" and can be removed.
// 4.  **Foundation for Behavior**: The output of this module—a stable, tracked list
//     of objects—is the essential foundation for all higher-level behavioral analysis,
//     such as motion pattern filtering (ignoring pacing) or re-identification.

use crate::core_modules::smart_blob::SmartBlob;
use crate::core_modules::smart_chunk::AnomalyDetails;
use std::collections::{HashSet, VecDeque};

const HISTORY_SIZE: usize = 15;
const MAX_FRAMES_SINCE_SEEN: u32 = 5;
const DISTANCE_THRESHOLD: f64 = 5.0;

/// Represents the current behavioral state of a tracked object.
#[derive(Debug, Clone, PartialEq)]
pub enum TrackedState {
    /// The object has just appeared and is being evaluated.
    New,
    /// The object is being tracked and its behavior is consistent and predictable.
    Tracking,
    /// The object has been temporarily lost (e.g., occlusion).
    Lost,
    /// The object has exhibited a statistically significant change in behavior.
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
        }
    }

    fn update(&mut self, blob: SmartBlob) {
        self.latest_blob = blob;
        self.age += 1;
        self.frames_since_seen = 0;

        // Update histories
        update_history(&mut self.position_history, self.latest_blob.center_of_mass);
        update_history(&mut self.size_history, self.latest_blob.size_in_chunks);
        update_history(&mut self.signature_history, self.latest_blob.average_anomaly.clone());

        // Update velocity
        if self.position_history.len() > 1 {
            let new_pos = self.position_history.back().unwrap();
            let old_pos = self.position_history.get(self.position_history.len() - 2).unwrap();
            self.velocity = (new_pos.0 - old_pos.0, new_pos.1 - old_pos.1);
            update_history(&mut self.velocity_history, self.velocity);
        }

        // The tracker will manage state transitions after the update.
    }
    
    fn predict_next_position(&self) -> (f64, f64) {
        let current_pos = self.latest_blob.center_of_mass;
        (current_pos.0 + self.velocity.0, current_pos.1 + self.velocity.1)
    }
}

/// Generic helper to update a history VecDeque.
fn update_history<T>(history: &mut VecDeque<T>, new_value: T) {
    history.push_back(new_value);
    if history.len() > HISTORY_SIZE {
        history.pop_front();
    }
}

/// Manages the list of `TrackedBlob`s from one frame to the next.
/// This is the engine of the behavioral analysis layer.
pub struct Tracker {
    /// The list of objects currently being tracked.
    tracked_blobs: Vec<TrackedBlob>,
    /// A counter to ensure each new object gets a unique ID.
    next_id: u64,
}

impl Tracker {
    pub fn new() -> Self {
        Self {
            tracked_blobs: Vec::new(),
            next_id: 0,
        }
    }

    /// Updates the tracker with a new set of detected blobs for the current frame.
    pub fn update(&mut self, new_blobs: Vec<SmartBlob>) -> &Vec<TrackedBlob> {
        // --- 0. Spatial Pre-filtering ---
        let legitimate_new_blobs = self.filter_internal_noise(new_blobs);

        // --- 1. Matching ---
        let (matches, mut unmatched_blobs) = self.match_blobs(legitimate_new_blobs);

        // --- 2. State Updating ---
        let mut updated_tracked_blobs = Vec::new();
        let mut matched_tracked_indices: HashSet<usize> = HashSet::new();

        // Update blobs that were successfully matched.
        for (tracked_idx, blob_idx) in matches {
            let mut tracked_blob = self.tracked_blobs[tracked_idx].clone();
            tracked_blob.update(unmatched_blobs.remove(&blob_idx).unwrap());
            updated_tracked_blobs.push(tracked_blob);
            matched_tracked_indices.insert(tracked_idx);
        }

        // Handle blobs that were not matched (occlusion or death).
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

        // Handle new blobs that were not matched (birth).
        for new_blob in unmatched_blobs.into_values() {
            let new_tracked_blob = TrackedBlob::new(self.next_id, new_blob);
            updated_tracked_blobs.push(new_tracked_blob);
            self.next_id += 1;
        }

        self.tracked_blobs = updated_tracked_blobs;
        &self.tracked_blobs
    }

    /// Provides a view into the current list of tracked blobs.
    pub fn get_tracked_blobs(&self) -> &Vec<TrackedBlob> {
        &self.tracked_blobs
    }

    fn filter_internal_noise(&self, new_blobs: Vec<SmartBlob>) -> Vec<SmartBlob> {
        let mut legitimate_new_blobs: Vec<SmartBlob> = Vec::new();
        for new_blob in new_blobs {
            let mut is_internal_noise = false;
            for tracked_blob in &self.tracked_blobs {
                let (top_left, bottom_right) = tracked_blob.latest_blob.bounding_box;
                let (cx, cy) = new_blob.center_of_mass;
                if cx >= top_left.x as f64 && cx <= bottom_right.x as f64 &&
                   cy >= top_left.y as f64 && cy <= bottom_right.y as f64 {
                    is_internal_noise = true;
                    break;
                }
            }
            if !is_internal_noise {
                legitimate_new_blobs.push(new_blob);
            }
        }
        legitimate_new_blobs
    }

    fn match_blobs(&self, legitimate_new_blobs: Vec<SmartBlob>) -> (Vec<(usize, usize)>, std::collections::HashMap<usize, SmartBlob>) {
        let mut matches: Vec<(usize, usize)> = Vec::new();
        let mut matched_new_blob_indices: HashSet<usize> = HashSet::new();
        let mut unmatched_blobs: std::collections::HashMap<usize, SmartBlob> = legitimate_new_blobs.into_iter().enumerate().collect();

        for (i, tracked_blob) in self.tracked_blobs.iter().enumerate() {
            let predicted_pos = tracked_blob.predict_next_position();
            let mut best_match_dist = DISTANCE_THRESHOLD;
            let mut best_match_index: Option<usize> = None;

            for (j, new_blob) in unmatched_blobs.iter() {
                let dist_sq = (predicted_pos.0 - new_blob.center_of_mass.0).powi(2)
                    + (predicted_pos.1 - new_blob.center_of_mass.1).powi(2);
                let dist = dist_sq.sqrt();

                if dist < best_match_dist {
                    best_match_dist = dist;
                    best_match_index = Some(*j);
                }
            }

            if let Some(j) = best_match_index {
                matches.push((i, j));
                matched_new_blob_indices.insert(j);
                unmatched_blobs.remove(&j);
            }
        }
        (matches, unmatched_blobs)
    }
}
