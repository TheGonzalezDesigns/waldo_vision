# Waldo Vision - Release Notes

## Version 0.2.0 - "Refinement"

**Release Date**: 2025-09-01

This release focuses on refining the core engine, improving stability, and clarifying the public API.

---

### Key Features & Capabilities

*   **Scene Stability State Machine**: Introduces `SceneState` (`Calibrating`, `Stable`, `Volatile`, `Disturbed`) with hysteresis to reduce false positives by modeling global scene stability.
*   **Pipeline Tunables**: Expanded `PipelineConfig` with practical thresholds used end‑to‑end:
    * `new_age_threshold`, `behavioral_anomaly_threshold` (tracker behavior).
    * `absolute_min_blob_size`, `blob_size_std_dev_filter` (blob filtering).
    * `disturbance_entry_threshold`, `disturbance_exit_threshold`, `disturbance_confirmation_frames` (scene state).
*   **Public API Clarification**: The main entrypoint is `VisionPipeline::process_frame(&[u8]) -> FrameAnalysis`.
    * `FrameAnalysis.report` is a `Report` enum: `NoSignificantMention` or `SignificantMention(MentionData)`.
    * Also exposes `status_map`, `tracked_blobs`, `scene_state`, and `significant_event_count` for visualization and diagnostics.

---

## Version 0.1.0 - "Architect"

**Release Date**: 2025-08-30

This is the initial public release of the `waldo_vision` engine. The primary focus of this version was the design and implementation of a robust, multi-layered architecture for real-time motion and event detection.

---

### Key Features & Capabilities

*   **End-to-End Vision Pipeline**: A complete, three-stage pipeline that transforms raw pixel data into a stable, tracked list of objects.
    1.  **Temporal Analysis Layer**: Uses a multi-channel statistical model to learn "normal" behavior for regions of an image and detect anomalous changes in luminance, color, and hue.
    2.  **Spatial Grouping Layer**: Implements a "Heatmap Peak-Finding and Region Growing" algorithm to group anomalous regions into coherent, distinct objects (`SmartBlob`s).
    3.  **Behavioral Analysis Layer**: Implements a "Nearest Neighbor" object tracker to add object permanence, tracking blobs from frame to frame and recording their journeys as `Moment`s.

*   **High-Level API**: Encapsulates the entire engine into a simple `VisionPipeline` struct.
    *   Provides a high-performance `significant_mention_detected()` method for real-time "go/no-go" filtering.
    *   Provides a `generate_report()` method that returns a rich `Report` with detailed data on all significant events.

*   **Tunable Configuration**: All key thresholds for the pipeline are exposed in a `PipelineConfig` struct, allowing for easy tuning of the engine's sensitivity.

*   **Comprehensive Documentation**: The entire codebase is documented with high-level architectural theory, explaining the purpose and design of each module. An `INTEGRATION_GUIDE.md` is also provided for consumers of the crate.

---

### Architectural Highlights

*   **Rust From Scratch**: The entire engine is built from the ground up in pure Rust, with the `image` crate used only for basic image I/O in tests.
*   **Separation of Concerns**: A clean, hierarchical architecture where each module has a single, well-defined responsibility.
*   **Performance Oriented**: Key algorithms are designed for efficiency, using techniques like flattened loops for data extraction and statistical analysis on chunked data to ensure real-time performance.

---
