# Waldo Vision Engine (`waldo_vision`)

`waldo_vision` is a multi-layered computer vision engine built from scratch in pure Rust. It is designed to detect, track, and analyze significant events in real-time video streams, acting as a high-performance pre-filter to more expensive AI analysis systems.

This crate was specifically designed to serve as the vision capability filter for the [Corpus AI Companion](https://github.com/TheGonzalezDesigns/corpus) project.

---

## Key Features

- **Temporal Analysis**: Uses a multi-channel statistical model to learn "normal" environmental behavior and detect anomalous changes.
- **Spatial Grouping**: Implements a "Heatmap Peak-Finding and Region Growing" algorithm to identify coherent objects in motion.
- **Behavioral Analysis**: Includes a robust object tracker that adds object permanence, tracking events over time to form complete "Moments."
- **High-Level API**: Provides a simple, powerful `VisionPipeline` API for easy integration.
- **Tunable**: All key sensitivity thresholds are exposed in a `PipelineConfig` struct.

---

## Integration

For detailed instructions on how to integrate this crate into your project, please see the **[INTEGRATION_GUIDE.md](INTEGRATION_GUIDE.md)**.

---

## License

This project is licensed under the MIT License.
