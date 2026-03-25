// THEORY:
// The `BlobDetector` is the engine of the Spatial Grouping Layer. It implements a
// sophisticated "Heatmap Peak-Finding and Region Growing" algorithm, which is a
// more intelligent alternative to simple binary connected-component analysis.
//
// Its purpose is to analyze the "heat map" of anomaly scores from the `GridManager`
// and identify spatially coherent objects, or "blobs."
//
// Key architectural principles & algorithm steps:
// 1.  **Heatmap Generation**: It first transforms the `Vec<ChunkStatus>` into a 2D
//     grid of floating-point "heat" values, using the `luminance_score` from each
//     `AnomalousEvent`. This preserves the magnitude of the anomaly, unlike a simple
//     binary approach.
// 2.  **Peak Finding (Seeding)**: It scans the heatmap to find "local maxima" - chunks
//     that are hotter than all of their immediate neighbors. These peaks are the
//     epicenters of motion and become the "seeds" for new blobs. This ensures we
//     start our analysis from the most significant point of an event.
// 3.  **Region Growing**: For each peak, the algorithm expands outwards, recursively
//     or iteratively adding neighboring chunks to the blob. This process continues
//     as long as the neighbors' heat is above a certain threshold, defining the
//     natural, gradient-based edge of the motion.
// 4.  **Data Aggregation**: Once a blob is fully grown, its high-level properties
//     (bounding box, center of mass, average anomaly scores) are calculated and
//     packaged into a `SmartBlob` struct.
// 5.  **Stateless Utility**: The `BlobDetector` is a stateless utility. Its `find_blobs`
//     function takes a status map for a single frame and produces a list of blobs
//     for that same frame. It has no memory of previous frames.

use crate::core_modules::smart_blob::{Point, SmartBlob};
use crate::core_modules::smart_chunk::{AnomalyDetails, ChunkStatus};

pub mod blob_detector {
    use super::*; // Make structs from parent module available.

    /// The main function of the spatial analysis layer.
    /// Takes a status map and identifies all coherent blobs of anomalous activity.
    pub fn find_blobs(
        status_map: &[ChunkStatus],
        grid_width: u32,
        grid_height: u32,
    ) -> Vec<SmartBlob> {
        // --- 1. Heatmap Generation ---
        // Convert the flat Vec<ChunkStatus> into a 2D grid of f64 heat values.
        // The heat is determined by the luminance_score of an AnomalousEvent.
        // Non-anomalous chunks are given a heat of 0.0.
        let mut heatmap = vec![vec![0.0; grid_width as usize]; grid_height as usize];
        for (i, status) in status_map.iter().enumerate() {
            if let ChunkStatus::AnomalousEvent(details) = status {
                let y = i / grid_width as usize;
                let x = i % grid_width as usize;
                heatmap[y][x] = details.luminance_score;
            }
        }

        // --- 2. Peak Finding ---
        // Find all local maxima in the heatmap to use as seeds for our blobs.
        // A chunk is a peak if its heat is greater than all 8 of its neighbors.
        let mut peaks: Vec<Point> = Vec::new();
        for y in 0..grid_height as usize {
            for x in 0..grid_width as usize {
                let heat = heatmap[y][x];
                // A chunk must have some heat to be a peak.
                if heat == 0.0 {
                    continue;
                }

                let mut is_peak = true;
                // Check all 8 neighbors.
                for dy in -1..=1 {
                    for dx in -1..=1 {
                        // Skip the center point itself.
                        if dy == 0 && dx == 0 {
                            continue;
                        }

                        let ny = y as i32 + dy;
                        let nx = x as i32 + dx;

                        // Check if the neighbor is within the grid boundaries.
                        if ny >= 0 && ny < grid_height as i32 && nx >= 0 && nx < grid_width as i32 {
                            if heatmap[ny as usize][nx as usize] > heat {
                                is_peak = false;
                                break;
                            }
                        }
                    }
                    if !is_peak {
                        break;
                    }
                }

                if is_peak {
                    peaks.push(Point {
                        x: x as u32,
                        y: y as u32,
                    });
                }
            }
        }

        // --- 3. Region Growing & Blob Creation ---
        // For each peak, grow a region and create a blob.
        // A `visited` grid is crucial to ensure we don't process the same chunk twice.
        let mut visited = vec![vec![false; grid_width as usize]; grid_height as usize];
        let mut blobs: Vec<SmartBlob> = Vec::new();
        let mut blob_id_counter = 0;

        for peak in peaks {
            if visited[peak.y as usize][peak.x as usize] {
                continue;
            }

            // Grow a new blob starting from this unvisited peak.
            let new_blob = grow_blob_from_peak(
                peak,
                &heatmap,
                &mut visited,
                blob_id_counter,
                status_map,
                grid_width,
            );
            blobs.push(new_blob);
            blob_id_counter += 1;
        }

        blobs
    }

    /// Defines the minimum "heat" a chunk must have to be included in a growing blob.
    /// This acts as the "cold edge" of the blob, preventing it from growing indefinitely
    /// into areas with very low, insignificant anomaly scores.
    const REGION_GROW_THRESHOLD: f64 = 1.0;

    /// Performs a breadth-first search (BFS) to find all connected chunks for a blob.
    fn grow_blob_from_peak(
        peak: Point,
        heatmap: &[Vec<f64>],
        visited: &mut [Vec<bool>],
        blob_id: u64,
        status_map: &[ChunkStatus],
        grid_width: u32,
    ) -> SmartBlob {
        let mut blob_chunks: Vec<Point> = Vec::new();
        let mut queue: Vec<Point> = vec![peak];
        visited[peak.y as usize][peak.x as usize] = true;

        let grid_height = heatmap.len() as i32;
        let grid_width_i32 = heatmap[0].len() as i32;

        while let Some(current) = queue.pop() {
            blob_chunks.push(current);

            // Check all 4 direct neighbors (not diagonals).
            for (dx, dy) in &[(0, 1), (0, -1), (1, 0), (-1, 0)] {
                let nx = current.x as i32 + dx;
                let ny = current.y as i32 + dy;

                if nx >= 0 && nx < grid_width_i32 && ny >= 0 && ny < grid_height {
                    let nx_u = nx as usize;
                    let ny_u = ny as usize;

                    if !visited[ny_u][nx_u] && heatmap[ny_u][nx_u] >= REGION_GROW_THRESHOLD {
                        visited[ny_u][nx_u] = true;
                        queue.push(Point {
                            x: nx_u as u32,
                            y: ny_u as u32,
                        });
                    }
                }
            }
        }

        // --- Data Aggregation ---
        // Now that we have all the chunks, calculate the final blob properties.
        let mut min_x = u32::MAX;
        let mut min_y = u32::MAX;
        let mut max_x = 0;
        let mut max_y = 0;
        let mut total_lum_score = 0.0;
        let mut total_col_score = 0.0;
        let mut total_hue_score = 0.0;
        let mut total_heat = 0.0;
        let mut center_x = 0.0;
        let mut center_y = 0.0;

        for point in &blob_chunks {
            min_x = min_x.min(point.x);
            min_y = min_y.min(point.y);
            max_x = max_x.max(point.x);
            max_y = max_y.max(point.y);

            let index = (point.y * grid_width + point.x) as usize;
            if let ChunkStatus::AnomalousEvent(details) = &status_map[index] {
                let heat = details.luminance_score;
                total_lum_score += details.luminance_score;
                total_col_score += details.color_score;
                total_hue_score += details.hue_score;
                total_heat += heat;
                center_x += point.x as f64 * heat;
                center_y += point.y as f64 * heat;
            }
        }

        let num_chunks = blob_chunks.len();
        SmartBlob {
            id: blob_id,
            bounding_box: (Point { x: min_x, y: min_y }, Point { x: max_x, y: max_y }),
            chunk_coords: blob_chunks,
            size_in_chunks: num_chunks,
            average_anomaly: AnomalyDetails {
                luminance_score: total_lum_score / num_chunks as f64,
                color_score: total_col_score / num_chunks as f64,
                hue_score: total_hue_score / num_chunks as f64,
            },
            center_of_mass: (center_x / total_heat, center_y / total_heat),
        }
    }
}

#[cfg(test)]
mod test {
    use super::blob_detector::*;
    use crate::core_modules::smart_chunk::{ChunkStatus, AnomalyDetails};
    use rand::distr::StandardUniform;
    use rand::prelude::*;
    use rand::rng;
    use ndarray::Array2;


    // Helper to create a specific anomalous event
    fn mock_anomaly(score: f64) -> ChunkStatus {
        ChunkStatus::AnomalousEvent(AnomalyDetails {
            luminance_score: score,
            color_score: 0.0,
            hue_score: 0.0,
        })
    }

    #[test]
    fn test_find_blobs_detects_injected_pattern() {
        let width = 10i32;
        let height = 10i32;

        let mut status_map = vec![ChunkStatus::Stable; (width * height) as usize];

        // inject a small 3x3 blob pattern
        // gonna peak at (5, 5) with heat 10.0
        let peak_x = 5;
        let peak_y = 5;

        for dy in -1i32..=1i32 {
            for dx in -1i32..=1i32 {
                let x = peak_x + dx;
                let y = peak_y + dy;
                let idx = (y * width + x) as usize;
                
                if dx == 0 && dy == 0 {
                    status_map[idx] = mock_anomaly(10.0); // peak
                } else {
                    status_map[idx] = mock_anomaly(5.0);  // body
                }
            }
        }

        let blobs = find_blobs(&status_map, width as u32, height as u32);

        assert_eq!(blobs.len(), 1, "Should find exactly one coherent blob");
        
        let blob = &blobs[0];
        assert_eq!(blob.size_in_chunks, 9, "The 3x3 pattern should result in 9 chunks");
        
        // check bounds
        let (min_pt, max_pt) = blob.bounding_box;
        assert_eq!(min_pt.x, 4);
        assert_eq!(min_pt.y, 4);
        assert_eq!(max_pt.x, 6);
        assert_eq!(max_pt.y, 6);

        // check center of mass, should be 5 due to symettry
        assert!((blob.center_of_mass.0 - 5.0).abs() < f64::EPSILON);
        assert!((blob.center_of_mass.1 - 5.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_find_blobs_separates_distinct_objects() {
        let width = 20;
        let height = 20;
        let mut status_map = vec![ChunkStatus::Stable; (width * height) as usize];

        status_map[(2 * width + 2) as usize] = mock_anomaly(10.0);
        
        status_map[(15 * width + 15) as usize] = mock_anomaly(10.0);

        let blobs = find_blobs(&status_map, width, height);

        assert_eq!(blobs.len(), 2, "Should have identified two distinct motion epicenters");
    }

    fn generate_random_activity_map() -> (u32, u32, Vec<ChunkStatus>) {
        let mut rng = rng();
        let (height, width) = (rng.random_range(64..500), rng.random_range(64..500));
        let matrix = Array2::from_shape_fn((height, width), |_| rand::random::<ChunkStatus>());
        let map = matrix.into_raw_vec();

        (height as u32, width as u32, map)
    }

    #[test]
    fn test_blob_detector_stability_with_random_noise() {
        let (height, width, status_map) = generate_random_activity_map();
        let blobs = find_blobs(&status_map, width, height);
        
        for blob in blobs {
            let (min, max) = blob.bounding_box;
            assert!(max.x < width && max.y < height);
        }
    }
}
