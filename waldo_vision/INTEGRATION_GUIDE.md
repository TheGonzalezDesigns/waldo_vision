# Integration Guide: Using `waldo_vision` as a Corpus Filter

This guide explains how to integrate the `waldo_vision` crate as a high‑performance, intelligent pre‑filter for a vision pipeline. The goal is to decide whether a raw camera frame is significant enough to forward to an expensive analysis API.

---

## 1. Dependency Installation

Add the `waldo_vision` crate to your project's `Cargo.toml`.

```toml
# In your-app/Cargo.toml

[dependencies]
waldo_vision = { git = "https://github.com/TheGonzalezDesigns/waldo_vision.git", branch = "main" }

# ... other dependencies
```

---

## 2. High-Level Usage

`waldo_vision` exposes a `VisionPipeline` as the single integration point. You create it once with a `PipelineConfig`, then call `process_frame(&[u8])` for each RGBA frame. The return value is a `FrameAnalysis` containing a high‑level `report` plus optional visualization and tracking data.

Flow:
- Initialize a single `VisionPipeline` instance on service start.
- Configure via `PipelineConfig` to tune sensitivity.
- For each incoming frame, call `process_frame`.
- Inspect `analysis.report` to decide whether to forward the frame.

---

## 3. Code Implementation

### Step 3.1: Initialization

```rust
use waldo_vision::pipeline::{VisionPipeline, PipelineConfig};

// Example configuration (load from your config in production)
let config = PipelineConfig {
    image_width: 1920,
    image_height: 1080,
    chunk_width: 10,
    chunk_height: 10,
    new_age_threshold: 5,
    behavioral_anomaly_threshold: 3.0,
    absolute_min_blob_size: 2,
    blob_size_std_dev_filter: 2.0,
    disturbance_entry_threshold: 0.25,
    disturbance_exit_threshold: 0.15,
    disturbance_confirmation_frames: 5,
};

let mut vision_pipeline = VisionPipeline::new(config);
```

### Step 3.2: Replacing Old Filter Logic

```rust
use waldo_vision::pipeline::{Report};

fn process_frame(frame_rgba: &[u8], pipeline: &mut VisionPipeline) {
    // Core integration point
    let analysis = pipeline.process_frame(frame_rgba);

    match analysis.report {
        Report::NoSignificantMention => {
            // Do nothing – saves downstream cost
        }
        Report::SignificantMention(data) => {
            // A statistically significant event occurred or the scene is disturbed.
            // Forward frame (and optionally metadata) to your expensive analysis.
            println!("Significant mention; new: {}, completed: {}, disturbed: {}",
                     data.new_significant_moments.len(),
                     data.completed_significant_moments.len(),
                     data.is_global_disturbance);
            // call_expensive_api(frame_rgba);
        }
    }
}
```

---

## 4. Tuning and Advanced Usage

### Tuning `PipelineConfig`

- `chunk_width` / `chunk_height`: Granularity vs. performance. Smaller detects small objects but costs more; 10x10 is a solid baseline.
- `new_age_threshold`: Frames a blob must live before leaving `New`. Higher reduces sensitivity to fleeting motion.
- `behavioral_anomaly_threshold`: Z‑score threshold used when classifying anomalous behavior in the tracker.
- `absolute_min_blob_size`: Hard minimum blob size (in chunks) to ignore tiny noise.
- `blob_size_std_dev_filter`: Filters blobs smaller than `(mean - k*std)` observed size.
- `disturbance_entry_threshold` / `disturbance_exit_threshold`: Fraction of unstable chunks to enter/exit scene disturbance.
- `disturbance_confirmation_frames`: Hysteresis to confirm disturbance before reporting.

### Advanced: Using Full Results

`process_frame` returns `FrameAnalysis` which includes:
- `report: Report` — `NoSignificantMention` or `SignificantMention(MentionData)` with new/completed `Moment`s and `is_global_disturbance`.
- `status_map: Vec<ChunkStatus>` — per‑chunk status (useful for overlays/heatmaps).
- `tracked_blobs: Vec<TrackedBlob>` — current tracking state per object.
- `scene_state: SceneState` — `Calibrating`, `Stable`, `Volatile`, `Disturbed`.

Example to inspect moments:

```rust
use waldo_vision::pipeline::{Report};

let analysis = pipeline.process_frame(frame_rgba);
if let Report::SignificantMention(data) = analysis.report {
    for m in data.new_significant_moments {
        println!("New moment started: {} (path len: {})", m.id, m.path.len());
    }
    for m in data.completed_significant_moments {
        println!("Moment completed: {} (frames {}..{})", m.id, m.start_frame, m.end_frame);
    }
}
```

This enables targeted actions (e.g., cropping around `blob_history` bbox, prioritizing certain behaviors, etc.).

---

By following this guide, you will replace simplistic heuristics with the `waldo_vision` engine, improving intelligence, performance, and tunability in your vision workflow.
