# Known Bugs and Limitations

This document tracks current issues, edge cases, and limitations with brief remediation ideas.

---

- Visual tester: out-of-order analysis
  - Symptom: `visual_tester` processes frames in parallel while sharing one `VisionPipeline` behind a `Mutex`. Lock acquisition order is nondeterministic, so frames can be analyzed out of order. This breaks tracking and scene stability assumptions.
  - Fix: Pre-convert frames to RGBA in parallel, then feed a single `VisionPipeline` sequentially in frame order; or disable parallelism around `process_frame` and keep a single consumer.

- Visual tester: high memory usage
  - Symptom: Reads entire video into memory before processing, causing large RAM spikes for long videos.
  - Fix: Stream frames (read/process/write incrementally) with a bounded queue; avoid storing all frames.

- Frame buffer bounds not validated
  - Symptom: `GridManager::process_frame` computes byte indices and calls `Pixel::from(&frame_buffer[idx..idx+4])` without checking length. Invalid dimensions or short buffers can panic.
  - Fix: Validate `frame_buffer.len() == image_width * image_height * 4` up front and return a `Result` (or `debug_assert!` with graceful handling in release builds).

- Image size vs. chunk size mismatch (edge loss)
  - Symptom: `grid_width = image_width / chunk_width` (integer division) and same for height; pixels in remainder columns/rows are ignored when dimensions are not exact multiples of chunk size.
  - Fix: Enforce divisibility in config, pad frames to next multiple, or implement partial‑chunk handling at the edges.

- Fragmented blobs not merged
  - Symptom: `Tracker::merge_fragmented_blobs` is a placeholder; a single real object may appear as multiple blobs and become multiple tracks.
  - Fix: Implement spatial/overlap merge with signature similarity (e.g., hue signature + proximity heuristics) before data association.

- Panic‑prone pixel conversion
  - Symptom: `Pixel::from(&[u8])` panics when slice length != 4; called widely in tight loops.
  - Fix: Prefer `From<&[u8; 4]>` for compile‑time safety, or add a `TryFrom<&[u8]>` returning `Result` and validate callers where inputs are not guaranteed.

- Hardcoded region grow threshold
  - Symptom: `blob_detector` uses a fixed `REGION_GROW_THRESHOLD = 1.0`, which may be suboptimal across scenes.
  - Fix: Expose this threshold via `PipelineConfig` or adapt it dynamically from the heat distribution.

- Event counter semantics
  - Symptom: `significant_event_count` increments per frame containing significance (or disturbance), not per discrete event.
  - Fix: Consider tracking counts by started/completed `Moment`s in addition to per‑frame counts for clearer metrics.

---

Non‑bugs recently addressed
- API/docs mismatch: Integration guide referenced non‑existent `significant_mention_detected()`/`generate_report()`. Updated to use `process_frame` and `Report`.
- Release notes: Removed reference to non‑existent `significance_recipe` and clarified actual tunables.

