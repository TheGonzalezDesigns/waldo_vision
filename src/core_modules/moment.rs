// THEORY:
// The `moment` module is the highest level of the vision system's architecture,
// forming the core of the Behavioral Analysis Layer. Its purpose is to transform the
// continuous stream of tracked object data into a discrete, historical narrative of
// events, called "Moments."
//
// Key architectural principles:
// 1.  **Narrative Creation**: This layer moves beyond simple tracking to storytelling.
//     A `Moment` represents the complete journey of a single object, from its first
//     appearance to its final disappearance.
// 2.  **Stateful Orchestration**: The `SceneManager` is the top-level orchestrator.
//     It owns the `Tracker` and maintains lists of "active" and "completed" moments,
//     managing the entire lifecycle of an event.
// 3.  **Contextual Analysis (Future)**: Once a `Moment` is complete, the system has
//     a rich, self-contained data package (the object's full path, its size and
//     color signature over time). This package is the ideal input for the final
//     AI or heuristic-based analysis to answer complex questions like "What was
//     this object doing?" or "Have I seen this object before in this 'Scene'?"
// 4.  **Final API Endpoint**: The `SceneManager` acts as the primary API for the
//     entire vision pipeline. A user of this library would create a `SceneManager`
//     instance, feed it raw frames, and receive a high-level stream of events
//     (e.g., "Moment Started," "Moment Completed").

use crate::core_modules::smart_blob::smart_blob::SmartBlob;
use crate::core_modules::tracker::{TrackedBlob, Tracker};
use std::collections::HashSet;

/// Represents the complete, historical record of a single tracked object's journey.
#[derive(Debug, Clone)]
pub struct Moment {
    pub id: u64,
    pub start_frame: u64,
    pub end_frame: u64,
    pub path: Vec<(f64, f64)>,
    pub blob_history: Vec<SmartBlob>,
    pub is_active: bool,
}

impl Moment {
    /// Creates a new, active Moment from a newly born TrackedBlob.
    fn new(tracked_blob: &TrackedBlob, start_frame: u64) -> Self {
        Self {
            id: tracked_blob.id,
            start_frame,
            end_frame: start_frame,
            path: vec![tracked_blob.latest_blob.center_of_mass],
            blob_history: vec![tracked_blob.latest_blob.clone()],
            is_active: true,
        }
    }

    /// Appends the latest data from a TrackedBlob to this Moment.
    fn update(&mut self, tracked_blob: &TrackedBlob, current_frame: u64) {
        self.end_frame = current_frame;
        self.path.push(tracked_blob.latest_blob.center_of_mass);
        self.blob_history.push(tracked_blob.latest_blob.clone());
    }

    /// Finalizes the Moment, marking it as no longer active.
    fn complete(&mut self) {
        self.is_active = false;
    }
}

/// The top-level orchestrator for the entire vision pipeline.
pub struct SceneManager {
    tracker: Tracker,
    active_moments: Vec<Moment>,
    completed_moments: Vec<Moment>,
    frame_count: u64,
}

impl SceneManager {
    pub fn new() -> Self {
        Self {
            tracker: Tracker::new(),
            active_moments: Vec::new(),
            completed_moments: Vec::new(),
            frame_count: 0,
        }
    }

    /// The main entry point for the behavioral layer.
    /// Processes the latest blobs, updates the tracker, and manages moment lifecycles.
    pub fn update(&mut self, blobs: Vec<SmartBlob>) {
        self.frame_count += 1;
        let tracked_blobs = self.tracker.update(blobs);

        let mut current_tracked_ids = HashSet::new();

        // Update active moments and identify new ones.
        for tracked_blob in tracked_blobs {
            current_tracked_ids.insert(tracked_blob.id);

            if let Some(active_moment) = self
                .active_moments
                .iter_mut()
                .find(|m| m.id == tracked_blob.id)
            {
                // This is an existing, ongoing moment. Update it.
                active_moment.update(tracked_blob, self.frame_count);
            } else {
                // This is a new blob birth. Create a new active moment.
                self.active_moments
                    .push(Moment::new(tracked_blob, self.frame_count));
            }
        }

        // Handle completed moments (deaths).
        let mut still_active = Vec::new();
        let mut just_completed = Vec::new();

        for mut moment in self.active_moments.drain(..) {
            if current_tracked_ids.contains(&moment.id) {
                // The moment is still active.
                still_active.push(moment);
            } else {
                // The moment's track was lost. It is now complete.
                moment.complete();
                just_completed.push(moment);
            }
        }

        self.active_moments = still_active;
        self.completed_moments.extend(just_completed);
    }

    /// Provides a view into the moments that are currently in progress.
    pub fn get_active_moments(&self) -> &Vec<Moment> {
        &self.active_moments
    }

    /// Provides a view into the historical archive of completed moments.
    pub fn get_completed_moments(&self) -> &Vec<Moment> {
        &self.completed_moments
    }
}
