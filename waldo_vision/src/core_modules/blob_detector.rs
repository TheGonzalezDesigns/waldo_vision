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
        // --- 1. Optimized Flat Heatmap Generation ---
        // Use flat array for better cache locality
        let total_chunks = (grid_width * grid_height) as usize;
        let mut heatmap = vec![0.0; total_chunks];
        for (i, status) in status_map.iter().enumerate() {
            if let ChunkStatus::AnomalousEvent(details) = status {
                heatmap[i] = details.luminance_score;
            }
        }

        // --- 2. Optimized Peak Finding ---
        let mut peaks: Vec<Point> = Vec::new();
        const NEIGHBORS: [(i32, i32); 8] = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1),  (1, 0),  (1, 1)
        ];
        
        for idx in 0..total_chunks {
            let heat = heatmap[idx];
            if heat == 0.0 { continue; }
            
            let y = (idx as u32) / grid_width;
            let x = (idx as u32) % grid_width;
            
            let mut is_peak = true;
            for &(dx, dy) in &NEIGHBORS {
                let nx = x as i32 + dx;
                let ny = y as i32 + dy;
                
                if nx >= 0 && nx < grid_width as i32 && ny >= 0 && ny < grid_height as i32 {
                    let neighbor_idx = (ny as u32 * grid_width + nx as u32) as usize;
                    if heatmap[neighbor_idx] > heat {
                        is_peak = false;
                        break;
                    }
                }
            }
            
            if is_peak {
                peaks.push(Point { x, y });
            }
        }

        // --- 3. Priority-based Region Growing ---
        let mut visited = vec![false; total_chunks];
        let mut blobs: Vec<SmartBlob> = Vec::new();
        let mut blob_id_counter = 0;
        
        // Sort peaks by heat value (process hottest first)
        let mut peak_priority: Vec<_> = peaks.iter()
            .map(|p| (p, heatmap[(p.y * grid_width + p.x) as usize]))
            .collect();
        peak_priority.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        for (peak, _) in peak_priority {
            let peak_idx = (peak.y * grid_width + peak.x) as usize;
            if visited[peak_idx] { continue; }

            let new_blob = grow_blob_from_peak(
                *peak,
                &heatmap,
                &mut visited,
                blob_id_counter,
                status_map,
                grid_width,
                grid_height,
            );
            blobs.push(new_blob);
            blob_id_counter += 1;
        }

        blobs
    }


    /// Defines the minimum "heat" a chunk must have to be included in a growing blob.
    const REGION_GROW_THRESHOLD: f64 = 1.0;

    /// Performs a breadth-first search (BFS) to find all connected chunks for a blob.
    fn grow_blob_from_peak(
        peak: Point,
        heatmap: &[f64],
        visited: &mut [bool],
        blob_id: u64,
        status_map: &[ChunkStatus],
        grid_width: u32,
        grid_height: u32,
    ) -> SmartBlob {
        use std::collections::BinaryHeap;
        use std::cmp::Ordering;
        
        #[derive(Clone, Copy)]
        struct PriorityNode {
            point: Point,
            heat: f64,
        }
        
        impl PartialEq for PriorityNode {
            fn eq(&self, other: &Self) -> bool {
                self.heat == other.heat
            }
        }
        impl Eq for PriorityNode {}
        impl Ord for PriorityNode {
            fn cmp(&self, other: &Self) -> Ordering {
                self.heat.partial_cmp(&other.heat).unwrap_or(Ordering::Equal)
            }
        }
        impl PartialOrd for PriorityNode {
            fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
                Some(self.cmp(other))
            }
        }
        
        let mut blob_chunks: Vec<Point> = Vec::new();
        let mut pq = BinaryHeap::new();
        
        let peak_idx = (peak.y * grid_width + peak.x) as usize;
        pq.push(PriorityNode { point: peak, heat: heatmap[peak_idx] });
        visited[peak_idx] = true;

        // Priority-based growth: explore high heat areas first
        while let Some(node) = pq.pop() {
            blob_chunks.push(node.point);

            // Check all 4 direct neighbors
            for (dx, dy) in &[(0, 1), (0, -1), (1, 0), (-1, 0)] {
                let nx = node.point.x as i32 + dx;
                let ny = node.point.y as i32 + dy;

                if nx >= 0 && nx < grid_width as i32 && ny >= 0 && ny < grid_height as i32 {
                    let neighbor_idx = (ny as u32 * grid_width + nx as u32) as usize;
                    
                    if !visited[neighbor_idx] && heatmap[neighbor_idx] >= REGION_GROW_THRESHOLD {
                        visited[neighbor_idx] = true;
                        pq.push(PriorityNode {
                            point: Point { x: nx as u32, y: ny as u32 },
                            heat: heatmap[neighbor_idx],
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