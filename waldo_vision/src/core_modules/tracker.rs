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
use std::collections::{HashSet, VecDeque};

const POSITION_HISTORY_SIZE: usize = 10;
const MAX_FRAMES_SINCE_SEEN: u32 = 5; // How many frames an object can be lost before it's deleted.
const DISTANCE_THRESHOLD: f64 = 5.0; // Max distance (in chunks) to be considered a match.

/// Represents an object that is being tracked across multiple frames.
/// This struct is stateful and holds the history and motion data for a single object.
#[derive(Debug, Clone)]
pub struct TrackedBlob {
    /// A unique and persistent ID for this tracked object.
    pub id: u64,
    /// The most recent `SmartBlob` data for this object, representing its state in the last frame it was seen.
    pub latest_blob: SmartBlob,
    /// A recent history of the object's center of mass, used for velocity calculation.
    pub position_history: VecDeque<(f64, f64)>,
    /// The calculated velocity of the object in chunks per frame.
    pub velocity: (f64, f64),
    /// The number of consecutive frames this object has been tracked.
    pub age: u32,
    /// The number of frames since this object was last seen. Used to handle occlusion and disappearance.
    pub frames_since_seen: u32,
}

impl TrackedBlob {
    /// Creates a new TrackedBlob from a SmartBlob.
    fn new(id: u64, blob: SmartBlob) -> Self {
        let mut position_history = VecDeque::with_capacity(POSITION_HISTORY_SIZE);
        position_history.push_back(blob.center_of_mass);
        Self {
            id,
            latest_blob: blob,
            position_history,
            velocity: (0.0, 0.0),
            age: 1,
            frames_since_seen: 0,
        }
    }

    /// Updates the state of a tracked blob with new data.
    fn update(&mut self, blob: SmartBlob) {
        self.latest_blob = blob;
        self.position_history
            .push_back(self.latest_blob.center_of_mass);
        if self.position_history.len() > POSITION_HISTORY_SIZE {
            self.position_history.pop_front();
        }

        // Calculate new velocity based on the last two positions.
        if self.position_history.len() > 1 {
            let new_pos = self.position_history.back().unwrap();
            let old_pos = self
                .position_history
                .get(self.position_history.len() - 2)
                .unwrap();
            self.velocity = (new_pos.0 - old_pos.0, new_pos.1 - old_pos.1);
        }

        self.age += 1;
        self.frames_since_seen = 0;
    }

    /// Predicts the next position of the blob based on its current velocity.
    fn predict_next_position(&self) -> (f64, f64) {
        let current_pos = self.latest_blob.center_of_mass;
        (
            current_pos.0 + self.velocity.0,
            current_pos.1 + self.velocity.1,
        )
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
        let mut matches: Vec<(usize, usize)> = Vec::new(); // (tracked_index, new_blob_index)
        let mut matched_new_blob_indices: HashSet<usize> = HashSet::new();

        // --- 1. Matching ---
        // Find the best match for each existing tracked blob.
        for (i, tracked_blob) in self.tracked_blobs.iter().enumerate() {
            let predicted_pos = tracked_blob.predict_next_position();
            let mut best_match_dist = DISTANCE_THRESHOLD;
            let mut best_match_index: Option<usize> = None;

            for (j, new_blob) in new_blobs.iter().enumerate() {
                // Skip new blobs that have already been matched.
                if matched_new_blob_indices.contains(&j) {
                    continue;
                }

                let dist_sq = (predicted_pos.0 - new_blob.center_of_mass.0).powi(2)
                    + (predicted_pos.1 - new_blob.center_of_mass.1).powi(2);
                let dist = dist_sq.sqrt();

                if dist < best_match_dist {
                    best_match_dist = dist;
                    best_match_index = Some(j);
                }
            }

            if let Some(j) = best_match_index {
                matches.push((i, j));
                matched_new_blob_indices.insert(j);
            }
        }

        // --- 2. State Updating ---
        let mut updated_tracked_blobs = Vec::new();
        let mut matched_tracked_indices: HashSet<usize> = HashSet::new();

        // Update blobs that were successfully matched.
        for (i, j) in matches {
            let mut tracked_blob = self.tracked_blobs[i].clone();
            tracked_blob.update(new_blobs[j].clone()); // `new_blobs` needs to be mutable or cloned
            updated_tracked_blobs.push(tracked_blob);
            matched_tracked_indices.insert(i);
        }

        // Handle blobs that were not matched (occlusion or death).
        for (i, tracked_blob) in self.tracked_blobs.iter().enumerate() {
            if !matched_tracked_indices.contains(&i) {
                let mut lost_blob = tracked_blob.clone();
                lost_blob.frames_since_seen += 1;
                if lost_blob.frames_since_seen <= MAX_FRAMES_SINCE_SEEN {
                    updated_tracked_blobs.push(lost_blob);
                }
            }
        }

        // Handle new blobs that were not matched (birth).
        for (j, new_blob) in new_blobs.into_iter().enumerate() {
            if !matched_new_blob_indices.contains(&j) {
                let new_tracked_blob = TrackedBlob::new(self.next_id, new_blob);
                updated_tracked_blobs.push(new_tracked_blob);
                self.next_id += 1;
            }
        }

        self.tracked_blobs = updated_tracked_blobs;
        &self.tracked_blobs
    }
}
