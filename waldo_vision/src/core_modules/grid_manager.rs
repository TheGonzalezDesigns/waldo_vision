// THEORY:
// The `GridManager` is the central nervous system of the temporal analysis layer.
// It acts as the owner and operator of the entire 2D grid of `SmartChunk`s. Its
// primary role is to orchestrate the flow of data from a raw image frame down to
// the individual `SmartChunk` analyzers and to collect their results into a coherent,
// high-level "status map."
//
// Key architectural principles:
// 1.  **Orchestration**: It is not an analyzer itself, but a manager. It holds the
//     master list of all `SmartChunk`s and is responsible for calling their `update`
//     methods in the correct sequence.
// 2.  **Data Transformation**: It performs the crucial first step of transforming raw
//     image data into a spatially organized grid of `Chunk` data objects. This
//     slicing operation is the bridge between the raw image and our chunk-based
//     analysis paradigm.
// 3.  **State Aggregation**: After updating every `SmartChunk`, its final job is to
//     aggregate their individual `ChunkStatus` reports into a single, unified data
//     structure (a `Vec<ChunkStatus>`). This "status map" is the final output of
//     the entire temporal layer and the direct input for the next architectural
//     layer (the `SmartBlob` spatial analyzer).
// 4.  **Decoupling**: It decouples the main application logic from the chunk analysis
//     logic. The main loop will only need to interact with the `GridManager`, giving
//     it a new frame and receiving a status map, without needing to know the
//     complex inner workings of the `SmartChunk`s.

use crate::core_modules::chunk::chunk::Chunk;
use crate::core_modules::pixel::pixel::Pixel;
use crate::core_modules::smart_chunk::{ChunkStatus, SmartChunk};
use std::sync::Arc;
use tokio::task;

/// Manages the entire grid of `SmartChunk`s and orchestrates the temporal analysis layer.
pub struct GridManager {
    /// The width of the full image in pixels, needed for chunk extraction math.
    image_width: u32,
    /// The width of the grid in chunks (image_width / chunk_width).
    grid_width: u32,
    /// The height of the grid in chunks (image_height / chunk_height).
    grid_height: u32,
    /// The width of a single chunk in pixels.
    chunk_width: u32,
    /// The height of a single chunk in pixels.
    chunk_height: u32,
    /// A flattened vector holding all the stateful `SmartChunk` analyzers, one for each grid position.
    smart_chunks: Vec<SmartChunk>,
    /// Pre-allocated pixel buffer to reduce allocations
    pixel_buffer: Vec<Vec<Pixel>>,
    /// Pre-calculated byte indices for faster pixel extraction
    byte_indices: Vec<Vec<usize>>,
}

impl GridManager {
    /// Creates a new GridManager for a given image dimension and chunk size.
    pub fn new(image_width: u32, image_height: u32, chunk_width: u32, chunk_height: u32) -> Self {
        let grid_width = image_width / chunk_width;
        let grid_height = image_height / chunk_height;
        let num_chunks = (grid_width * grid_height) as usize;
        let mut smart_chunks = Vec::with_capacity(num_chunks);

        // Initialize smart chunks
        for i in 0..num_chunks {
            let y = i as u32 / grid_width;
            let x = i as u32 % grid_width;
            smart_chunks.push(SmartChunk::new(x, y));
        }
        
        // Pre-allocate pixel buffers for all chunks
        let chunk_pixels = (chunk_width * chunk_height) as usize;
        let mut pixel_buffer = Vec::with_capacity(num_chunks);
        for _ in 0..num_chunks {
            pixel_buffer.push(Vec::with_capacity(chunk_pixels));
        }
        
        // Pre-calculate byte indices for faster pixel extraction
        let mut byte_indices = Vec::with_capacity(num_chunks);
        for chunk_index in 0..num_chunks {
            let chunk_y = chunk_index as u32 / grid_width;
            let chunk_x = chunk_index as u32 % grid_width;
            let start_pixel_x = chunk_x * chunk_width;
            let start_pixel_y = chunk_y * chunk_height;
            
            let mut chunk_byte_indices = Vec::with_capacity(chunk_pixels);
            for i in 0..chunk_pixels {
                let y_offset = i as u32 / chunk_width;
                let x_offset = i as u32 % chunk_width;
                let pixel_y = start_pixel_y + y_offset;
                let pixel_x = start_pixel_x + x_offset;
                let byte_index = ((pixel_y * image_width) + pixel_x) * 4;
                chunk_byte_indices.push(byte_index as usize);
            }
            byte_indices.push(chunk_byte_indices);
        }

        Self {
            image_width,
            grid_width,
            grid_height,
            chunk_width,
            chunk_height,
            smart_chunks,
            pixel_buffer,
            byte_indices,
        }
    }

    /// The main entry point for the vision system with adaptive parallelization.
    /// Takes a raw RGBA image buffer, processes it, and returns a map of chunk statuses.
    pub async fn process_frame(&mut self, frame_buffer: &[u8]) -> Vec<ChunkStatus> {
        let frame_arc = Arc::new(frame_buffer.to_vec());
        let num_chunks = self.smart_chunks.len();
        let chunk_pixels = (self.chunk_width * self.chunk_height) as usize;
        
        // Determine optimal batch size based on CPU cores
        let num_cpus = num_cpus::get();
        let batch_size = (num_chunks + num_cpus - 1) / num_cpus;
        
        // Process chunks in concurrent batches
        let mut chunk_futures = Vec::new();
        
        for batch_start in (0..num_chunks).step_by(batch_size) {
            let batch_end = (batch_start + batch_size).min(num_chunks);
            let frame_clone = Arc::clone(&frame_arc);
            let byte_indices_batch = self.byte_indices[batch_start..batch_end].to_vec();
            let chunk_width = self.chunk_width;
            let chunk_height = self.chunk_height;
            
            // Spawn async task for each batch
            let batch_future = task::spawn(async move {
                let mut batch_chunks = Vec::with_capacity(batch_end - batch_start);
                
                for indices in byte_indices_batch {
                    // Extract pixels using pre-calculated indices
                    let mut chunk_pixels_data = Vec::with_capacity(chunk_pixels);
                    
                    // Use unsafe for faster memory access (validated indices)
                    unsafe {
                        for &byte_idx in &indices {
                            // Direct memory access without bounds checking
                            let r = *frame_clone.get_unchecked(byte_idx);
                            let g = *frame_clone.get_unchecked(byte_idx + 1);
                            let b = *frame_clone.get_unchecked(byte_idx + 2);
                            let a = *frame_clone.get_unchecked(byte_idx + 3);
                            
                            chunk_pixels_data.push(Pixel {
                                red: r,
                                green: g,
                                blue: b,
                                alpha: a,
                            });
                        }
                    }
                    
                    let chunk_data = Chunk::new(chunk_width, chunk_height, chunk_pixels_data);
                    batch_chunks.push(chunk_data);
                }
                
                batch_chunks
            });
            
            chunk_futures.push((batch_start, batch_future));
        }
        
        // Await all batch processing
        let mut all_chunks: Vec<Option<Chunk>> = Vec::with_capacity(num_chunks);
        for _ in 0..num_chunks {
            all_chunks.push(None);
        }
        for (batch_start, future) in chunk_futures {
            if let Ok(batch_chunks) = future.await {
                for (i, chunk) in batch_chunks.into_iter().enumerate() {
                    all_chunks[batch_start + i] = Some(chunk);
                }
            }
        }
        
        // Update smart chunks sequentially (they maintain state)
        for (i, chunk_opt) in all_chunks.into_iter().enumerate() {
            if let Some(chunk_data) = chunk_opt {
                self.smart_chunks[i].update(&chunk_data);
            }
        }
        
        // Collect statuses
        self.smart_chunks
            .iter()
            .map(|sc| sc.status.clone())
            .collect()
    }
}