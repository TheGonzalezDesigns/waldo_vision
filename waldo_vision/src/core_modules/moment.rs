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

use crate::core_modules::smart_blob::SmartBlob;
use crate::core_modules::tracker::{TrackedBlob, Tracker};
use std::collections::HashSet;

/// Represents the complete, historical record of a single tracked object's journey.
#[derive(Debug, Clone)]
pub struct Moment {
    /// A unique ID for this moment, inherited from its `TrackedBlob`.
    pub id: u64,
    /// The frame number when the object first appeared.
    pub start_frame: u64,
    /// The last frame number when the object was seen.
    pub end_frame: u64,
    /// The complete path of the object's center of mass over its lifetime.
    pub path: Vec<(f64, f64)>,
    /// A collection of the `SmartBlob` data for each frame the object was visible.
    /// This contains the rich signature data (size, shape, anomaly scores) for later analysis.
    pub blob_history: Vec<SmartBlob>,
    /// A flag indicating if the moment is still in progress (i.e., the object is still being tracked).
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

/// The top-level orchestrator for the behavioral analysis layer.
/// Owns the `Tracker` and manages the lifecycle of `Moment`s.
pub struct SceneManager {
    /// The underlying object tracker that performs frame-to-frame data association.
    tracker: Tracker,
    /// A list of `Moment`s that are currently in progress.
    active_moments: Vec<Moment>,
    /// A historical archive of `Moment`s that have concluded.
    completed_moments: Vec<Moment>,
    /// A counter for the total number of frames processed.
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

    /// Processes the latest blobs, updates the tracker, and manages moment lifecycles.
    /// Returns a tuple of (newly_started_moments, newly_completed_moments).
    pub fn update(&mut self, blobs: Vec<SmartBlob>) -> (Vec<Moment>, Vec<Moment>) {
        self.frame_count += 1;
        let tracked_blobs = self.tracker.update(blobs);

        let mut current_tracked_ids = HashSet::new();
        let mut newly_started_moments = Vec::new();

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
                let new_moment = Moment::new(tracked_blob, self.frame_count);
                newly_started_moments.push(new_moment.clone());
                self.active_moments.push(new_moment);
            }
        }

        // Handle completed moments (deaths).
        let mut still_active = Vec::new();
        let mut newly_completed_moments = Vec::new();

        for mut moment in self.active_moments.drain(..) {
            if current_tracked_ids.contains(&moment.id) {
                // The moment is still active.
                still_active.push(moment);
            } else {
                // The moment's track was lost. It is now complete.
                moment.complete();
                newly_completed_moments.push(moment.clone());
                self.completed_moments.push(moment);
            }
        }

        self.active_moments = still_active;
        (newly_started_moments, newly_completed_moments)
    }

    /// Provides a view into the moments that are currently in progress.
    pub fn get_active_moments(&self) -> &Vec<Moment> {
        &self.active_moments
    }

    /// Provides a view into the historical archive of completed moments.
    pub fn get_completed_moments(&self) -> &Vec<Moment> {
        &self.completed_moments
    }

    /// Provides a view into the `Tracker`'s current list of tracked objects.
    pub fn get_tracked_blobs(&self) -> &Vec<TrackedBlob> {
        self.tracker.get_tracked_blobs()
    }
}
