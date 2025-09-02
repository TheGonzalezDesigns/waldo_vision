// THEORY:
// This file is the main entry point for the `waldo_vision` library crate.
// It follows the standard Rust convention of using `lib.rs` to define the public
// API that will be exposed to external consumers (like the `corpus` orchestrator).
//
// The primary goal is to export the `VisionPipeline` and its associated data
// structures (`PipelineConfig`, `Report`, etc.) as the clean, high-level
// interface for the entire vision engine. All the complex internal modules
// (`core_modules`) are encapsulated and hidden from the end-user, providing a
// clean separation of concerns.

pub mod core_modules;
pub mod pipeline;
pub mod parallel_pipeline;
