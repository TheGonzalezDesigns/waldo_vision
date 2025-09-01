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
use std::sync::Arc;
use tokio::task;

pub mod blob_detector {
    use super::*;
    
    // Pre-computed neighbor offsets for peak finding - eliminates inner loops
    const NEIGHBOR_OFFSETS: [(i32, i32); 8] = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1)
    ];
    
    // Pre-computed neighbor offsets for region growing (4-connectivity)
    const GROW_OFFSETS: [(i32, i32); 4] = [(0, 1), (0, -1), (1, 0), (-1, 0)];
    
    // Threshold for when to use parallel processing
    const PARALLEL_THRESHOLD: usize = 10;

    /// The main function of the spatial analysis layer.
    /// Takes a status map and identifies all coherent blobs of anomalous activity.
    pub async fn find_blobs(
        status_map: &[ChunkStatus],
        grid_width: u32,
        grid_height: u32,
    ) -> Vec<SmartBlob> {
        // --- 1. Optimized Heatmap Generation O(n) ---
        let total_size = (grid_width * grid_height) as usize;
        let mut heatmap_flat = vec![0.0; total_size];
        
        // Direct indexing without division/modulo in the loop
        for (i, status) in status_map.iter().enumerate() {
            if let ChunkStatus::AnomalousEvent(details) = status {
                heatmap_flat[i] = details.luminance_score;
            }
        }

        // --- 2. Optimized Peak Finding O(n) ---
        let peaks = find_peaks_optimized(&heatmap_flat, grid_width, grid_height);

        // --- 3. Adaptive Region Growing ---
        // Use parallel processing only when we have many peaks
        if peaks.len() > PARALLEL_THRESHOLD {
            grow_blobs_parallel(peaks, heatmap_flat, status_map, grid_width, grid_height).await
        } else {
            grow_blobs_sequential(peaks, heatmap_flat, status_map, grid_width, grid_height)
        }
    }

    /// Optimized peak finding with single pass O(n) and early termination
    fn find_peaks_optimized(
        heatmap: &[f64],
        grid_width: u32,
        grid_height: u32,
    ) -> Vec<Point> {
        let width = grid_width as i32;
        let height = grid_height as i32;
        
        // Pre-allocate with estimated capacity
        let mut peaks = Vec::with_capacity((grid_width * grid_height / 100) as usize);
        
        // Single pass through flattened heatmap - O(n)
        for (idx, &heat) in heatmap.iter().enumerate() {
            // Early termination for zero heat
            if heat == 0.0 {
                continue;
            }
            
            let y = (idx as i32) / width;
            let x = (idx as i32) % width;
            
            // Check if it's a local maximum using pre-computed offsets
            let mut is_peak = true;
            for &(dx, dy) in &NEIGHBOR_OFFSETS {
                let nx = x + dx;
                let ny = y + dy;
                
                // Boundary check
                if nx >= 0 && nx < width && ny >= 0 && ny < height {
                    let neighbor_idx = (ny * width + nx) as usize;
                    if heatmap[neighbor_idx] > heat {
                        is_peak = false;
                        break; // Early termination
                    }
                }
            }
            
            if is_peak {
                peaks.push(Point {
                    x: x as u32,
                    y: y as u32,
                });
            }
        }
        
        peaks
    }

    /// Sequential blob growing for small numbers of blobs
    fn grow_blobs_sequential(
        peaks: Vec<Point>,
        heatmap_flat: Vec<f64>,
        status_map: &[ChunkStatus],
        grid_width: u32,
        grid_height: u32,
    ) -> Vec<SmartBlob> {
        let total_size = (grid_width * grid_height) as usize;
        let mut visited = vec![false; total_size];
        let mut blobs = Vec::new();
        let mut blob_id_counter = 0;

        for peak in peaks {
            let peak_idx = (peak.y * grid_width + peak.x) as usize;
            if visited[peak_idx] {
                continue;
            }

            let blob = grow_blob_from_peak(
                peak,
                &heatmap_flat,
                &mut visited,
                blob_id_counter,
                status_map,
                grid_width,
                grid_height,
            );
            blobs.push(blob);
            blob_id_counter += 1;
        }

        blobs
    }

    /// Parallel blob growing for large numbers of blobs
    async fn grow_blobs_parallel(
        peaks: Vec<Point>,
        heatmap_flat: Vec<f64>,
        status_map: &[ChunkStatus],
        grid_width: u32,
        grid_height: u32,
    ) -> Vec<SmartBlob> {
        let total_size = (grid_width * grid_height) as usize;
        let mut visited = vec![false; total_size];
        let mut blob_futures = Vec::new();
        let mut blob_id_counter = 0;
        
        // Convert to Arc for sharing across tasks
        let heatmap_arc = Arc::new(heatmap_flat);
        let status_map_arc = Arc::new(status_map.to_vec());

        for peak in peaks {
            let peak_idx = (peak.y * grid_width + peak.x) as usize;
            if visited[peak_idx] {
                continue;
            }

            // Clone Arcs for the async task
            let heatmap_clone = Arc::clone(&heatmap_arc);
            let status_map_clone = Arc::clone(&status_map_arc);
            let mut visited_local = visited.clone();
            
            // Spawn concurrent blob growing tasks
            let blob_future = task::spawn(async move {
                grow_blob_from_peak(
                    peak,
                    &heatmap_clone,
                    &mut visited_local,
                    blob_id_counter,
                    &status_map_clone,
                    grid_width,
                    grid_height,
                )
            });
            
            blob_futures.push(blob_future);
            
            // Mark visited in main thread too
            mark_blob_visited(&mut visited, peak, &heatmap_arc, grid_width, grid_height);
            blob_id_counter += 1;
        }

        // Await all blob growing tasks
        let mut blobs = Vec::new();
        for future in blob_futures {
            if let Ok(blob) = future.await {
                blobs.push(blob);
            }
        }

        blobs
    }

    /// Defines the minimum "heat" a chunk must have to be included in a growing blob.
    const REGION_GROW_THRESHOLD: f64 = 1.0;

    /// Optimized blob growing with pre-allocated buffers - O(blob_size)
    fn grow_blob_from_peak(
        peak: Point,
        heatmap: &[f64],
        visited: &mut [bool],
        blob_id: u64,
        status_map: &[ChunkStatus],
        grid_width: u32,
        grid_height: u32,
    ) -> SmartBlob {
        let width = grid_width as i32;
        let height = grid_height as i32;
        
        // Pre-allocate with estimated capacity
        let mut blob_chunks = Vec::with_capacity(50);
        let mut queue = Vec::with_capacity(50);
        
        let peak_idx = (peak.y * grid_width + peak.x) as usize;
        queue.push(peak);
        visited[peak_idx] = true;

        // Optimized BFS with pre-computed offsets - O(blob_size)
        while let Some(current) = queue.pop() {
            blob_chunks.push(current);

            // Use pre-computed offsets for neighbors
            for &(dx, dy) in &GROW_OFFSETS {
                let nx = current.x as i32 + dx;
                let ny = current.y as i32 + dy;

                if nx >= 0 && nx < width && ny >= 0 && ny < height {
                    let neighbor_idx = (ny * width + nx) as usize;
                    
                    if !visited[neighbor_idx] && heatmap[neighbor_idx] >= REGION_GROW_THRESHOLD {
                        visited[neighbor_idx] = true;
                        queue.push(Point {
                            x: nx as u32,
                            y: ny as u32,
                        });
                    }
                }
            }
        }

        // --- Optimized Data Aggregation O(blob_size) ---
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
    
    /// Helper to mark blob regions as visited - O(blob_size)
    fn mark_blob_visited(
        visited: &mut [bool],
        start: Point,
        heatmap: &[f64],
        grid_width: u32,
        grid_height: u32,
    ) {
        let width = grid_width as i32;
        let height = grid_height as i32;
        let mut stack = vec![start];
        
        while let Some(current) = stack.pop() {
            let idx = (current.y * grid_width + current.x) as usize;
            if visited[idx] {
                continue;
            }
            visited[idx] = true;
            
            for &(dx, dy) in &GROW_OFFSETS {
                let nx = current.x as i32 + dx;
                let ny = current.y as i32 + dy;
                
                if nx >= 0 && nx < width && ny >= 0 && ny < height {
                    let neighbor_idx = (ny * width + nx) as usize;
                    if !visited[neighbor_idx] && heatmap[neighbor_idx] >= REGION_GROW_THRESHOLD {
                        stack.push(Point {
                            x: nx as u32,
                            y: ny as u32,
                        });
                    }
                }
            }
        }
    }
}