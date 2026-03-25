// THEORY:
// The `moment` module is the highest level of the vision system's architecture,
// forming the core of the Behavioral Analysis Layer. Its purpose is to transform the
// continuous stream of tracked object data into a discrete, historical narrative of
// events, called "Moments."

use crate::core_modules::smart_blob::SmartBlob;
use crate::core_modules::tracker::{TrackedBlob, TrackedState, Tracker};
use crate::pipeline::PipelineConfig;
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
    pub is_significant: bool,
}

impl Moment {
    fn new(tracked_blob: &TrackedBlob, start_frame: u64) -> Self {
        Self {
            id: tracked_blob.id,
            start_frame,
            end_frame: start_frame,
            path: vec![tracked_blob.latest_blob.center_of_mass],
            blob_history: vec![tracked_blob.latest_blob.clone()],
            is_active: true,
            is_significant: false,
        }
    }

    fn update(&mut self, tracked_blob: &TrackedBlob, current_frame: u64) {
        self.end_frame = current_frame;
        self.path.push(tracked_blob.latest_blob.center_of_mass);
        self.blob_history.push(tracked_blob.latest_blob.clone());
    }

    fn complete(&mut self) {
        self.is_active = false;
    }
}

/// The top-level orchestrator for the behavioral analysis layer.
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

    pub fn update(
        &mut self,
        blobs: Vec<SmartBlob>,
        config: &PipelineConfig,
    ) -> (Vec<Moment>, Vec<Moment>) {
        self.frame_count += 1;
        let tracked_blobs = self.tracker.update(blobs, config);

        let mut current_tracked_ids = HashSet::new();
        let mut newly_started_moments = Vec::new();

        for tracked_blob in tracked_blobs {
            current_tracked_ids.insert(tracked_blob.id);

            let moment = if let Some(m) = self
                .active_moments
                .iter_mut()
                .find(|m| m.id == tracked_blob.id)
            {
                m.update(tracked_blob, self.frame_count);
                m
            } else {
                let new_moment = Moment::new(tracked_blob, self.frame_count);
                newly_started_moments.push(new_moment.clone());
                self.active_moments.push(new_moment);
                self.active_moments.last_mut().unwrap()
            };

            moment.is_significant = tracked_blob.state == TrackedState::New
                || tracked_blob.state == TrackedState::Anomalous;
        }

        let mut still_active = Vec::new();
        let mut newly_completed_moments = Vec::new();

        for mut moment in self.active_moments.drain(..) {
            if current_tracked_ids.contains(&moment.id) {
                still_active.push(moment);
            } else {
                moment.complete();
                newly_completed_moments.push(moment.clone());
                self.completed_moments.push(moment);
            }
        }

        self.active_moments = still_active;
        (newly_started_moments, newly_completed_moments)
    }

    pub fn get_active_moments(&self) -> &Vec<Moment> {
        &self.active_moments
    }

    pub fn get_completed_moments(&self) -> &Vec<Moment> {
        &self.completed_moments
    }

    pub fn get_tracked_blobs(&self) -> &Vec<TrackedBlob> {
        self.tracker.get_tracked_blobs()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core_modules::smart_blob::Point;
    use crate::core_modules::smart_chunk::AnomalyDetails;

    fn create_mock_config() -> PipelineConfig {
        PipelineConfig {
            image_width: 100,
            image_height: 100,
            chunk_width: 10,
            chunk_height: 10,
            new_age_threshold: 1, // Make it transition to Tracking fast
            behavioral_anomaly_threshold: 3.0,
            absolute_min_blob_size: 1,
            blob_size_std_dev_filter: 2.0,
            disturbance_entry_threshold: 5.0,
            disturbance_exit_threshold: 2.0,
            disturbance_confirmation_frames: 3,
        }
    }

    fn create_mock_blob(id: u64, x: f64, y: f64) -> SmartBlob {
        SmartBlob {
            id,
            bounding_box: (Point { x: 0, y: 0 }, Point { x: 1, y: 1 }),
            chunk_coords: vec![],
            size_in_chunks: 10,
            average_anomaly: AnomalyDetails {
                luminance_score: 5.0,
                color_score: 5.0,
                hue_score: 5.0,
            },
            center_of_mass: (x, y),
        }
    }

    #[test]
    fn test_scene_manager_lifecycle() {
        let mut sm = SceneManager::new();
        let config = create_mock_config();

        let b1 = create_mock_blob(1, 10.0, 10.0);
        let (started, _) = sm.update(vec![b1], &config);
        assert_eq!(started.len(), 1);
        assert_eq!(sm.active_moments.len(), 1);
        let first_id = started[0].id;

        let b2 = create_mock_blob(2, 10.5, 10.5);
        let (started, completed) = sm.update(vec![b2], &config);
        assert_eq!(started.len(), 0);
        assert_eq!(completed.len(), 0);
        assert_eq!(sm.active_moments.len(), 1);
        assert_eq!(sm.active_moments[0].id, first_id);
        assert_eq!(sm.active_moments[0].path.len(), 2);

        for _ in 0..5 {
            sm.update(vec![], &config);
        }
        let (started, completed) = sm.update(vec![], &config);
        assert_eq!(started.len(), 0);
        assert_eq!(completed.len(), 1);
        assert_eq!(sm.active_moments.len(), 0);
        assert_eq!(sm.completed_moments.len(), 1);
        assert_eq!(completed[0].id, first_id);
    }
}
