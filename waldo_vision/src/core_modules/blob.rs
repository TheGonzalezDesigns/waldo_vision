// THEORY:
// The `SmartBlob` module is the primary component of the Spatial Grouping Layer.
// Its purpose is to transform the purely temporal, single-point-in-space analysis
// from the `SmartChunk` layer into a spatially coherent object. A `SmartBlob`
// represents a single, contiguous region of anomalous activity at a specific
// moment in time.
//
// Key architectural principles:
// 1.  **Spatial Cohesion**: A `SmartBlob` is a collection of `SmartChunk`s that are
//     both in an anomalous state and are physically adjacent to each other on the
//     grid. It finds meaningful shapes in the "noise" of the status map.
// 2.  **Data Aggregation**: It acts as a high-level summary of a motion event.
//     Instead of dealing with dozens of individual chunk statuses, the system can
//     now work with a single `SmartBlob` object that has clearly defined properties
//     like a bounding box, center of mass, and total area.
// 3.  **Stateless Data Container**: Much like `Pixel` and `Chunk`, the `SmartBlob`
//     struct itself is a "dumb" data container. It represents a detected object
//     within a single frame. It does not have a memory of its past positions or
//     behaviors.
// 4.  **Input for the Next Layer**: A list of `SmartBlob`s is the final output of
//     the spatial analysis layer. This list becomes the perfect input for the
//     final architectural layer (Behavioral Analysis), which will track these
//     blobs over time to create "Moments" and narratives.

use crate::core_modules::smart_chunk::AnomalyDetails;

/// A simple struct to represent a 2D point or coordinate on the chunk grid.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Point {
    pub x: u32,
    pub y: u32,
}

/// Represents a single, spatially coherent object detected in a frame.
/// This is a "dumb" data container that summarizes the properties of a detected motion event.
#[derive(Debug, Clone)]
pub struct SmartBlob {
    /// A unique identifier assigned to this blob for the current frame only. Not persistent.
    pub id: u64,
    /// The rectangular box that encloses all chunks in the blob, represented by its
    /// top-left and bottom-right corners.
    pub bounding_box: (Point, Point),
    /// The list of grid coordinates for every chunk that makes up this blob.
    pub chunk_coords: Vec<Point>,
    /// The total number of chunks in the blob, representing its area.
    pub size_in_chunks: usize,
    /// The average "significance" scores (luminance, color, hue) from all chunks
    /// within the blob. This forms the core of the blob's analytical signature.
    pub average_anomaly: AnomalyDetails,
    /// The center of the blob, weighted by the `luminance_score` of each chunk.
    /// This provides a more precise location of the "epicenter" of the motion.
    pub center_of_mass: (f64, f64),
}
