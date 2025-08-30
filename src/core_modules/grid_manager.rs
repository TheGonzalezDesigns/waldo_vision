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
use crate::core_modules::smart_chunk::smart_chunk::{ChunkStatus, SmartChunk};

pub struct GridManager {
    image_width: u32,
    grid_width: u32,
    grid_height: u32,
    chunk_width: u32,
    chunk_height: u32,
    smart_chunks: Vec<SmartChunk>,
}

impl GridManager {
    /// Creates a new GridManager for a given image dimension and chunk size.
    pub fn new(image_width: u32, image_height: u32, chunk_width: u32, chunk_height: u32) -> Self {
        let grid_width = image_width / chunk_width;
        let grid_height = image_height / chunk_height;
        let num_chunks = (grid_width * grid_height) as usize;
        let mut smart_chunks = Vec::with_capacity(num_chunks);

        for y in 0..grid_height {
            for x in 0..grid_width {
                smart_chunks.push(SmartChunk::new(x, y));
            }
        }

        Self {
            image_width,
            grid_width,
            grid_height,
            chunk_width,
            chunk_height,
            smart_chunks,
        }
    }

    /// The main entry point for the vision system.
    /// Takes a raw RGBA image buffer, processes it, and returns a map of chunk statuses.
    pub fn process_frame(&mut self, frame_buffer: &[u8]) -> Vec<ChunkStatus> {
        // This is the flattened loop to iterate over chunks that we designed earlier.
        for chunk_index in 0..self.smart_chunks.len() {
            let chunk_y = chunk_index as u32 / self.grid_width;
            let chunk_x = chunk_index as u32 % self.grid_width;

            // Calculate the starting pixel position (top-left corner) of the current chunk.
            let start_pixel_x = chunk_x * self.chunk_width;
            let start_pixel_y = chunk_y * self.chunk_height;

            let mut chunk_pixels =
                Vec::with_capacity((self.chunk_width * self.chunk_height) as usize);

            // Iterate over each row of pixels within the current chunk.
            for y_offset in 0..self.chunk_height {
                let current_row = start_pixel_y + y_offset;
                // Calculate the starting byte index for the current row in the main image buffer.
                let row_start_byte_index = (current_row * self.image_width * 4) as usize;

                // Calculate the starting and ending byte indices for the pixels of this chunk in the current row.
                let row_pixel_start_byte_index =
                    row_start_byte_index + (start_pixel_x * 4) as usize;
                let row_pixel_end_byte_index =
                    row_pixel_start_byte_index + (self.chunk_width * 4) as usize;

                // Get a slice of the bytes for the current row segment.
                let row_bytes = &frame_buffer[row_pixel_start_byte_index..row_pixel_end_byte_index];

                // Convert the byte slice into a Vec<Pixel>.
                // The `chunks_exact` method is a highly efficient way to group bytes into RGBA pixels.
                chunk_pixels.extend(row_bytes.chunks_exact(4).map(|rgba| Pixel::from(rgba)));
            }

            let chunk_data = Chunk::new(self.chunk_width, self.chunk_height, chunk_pixels);

            // Update the corresponding SmartChunk with the new data for its location.
            self.smart_chunks[chunk_index].update(&chunk_data);
        }

        // After all chunks are updated, collect their new statuses to create the final status map.
        self.smart_chunks
            .iter()
            .map(|sc| sc.status.clone())
            .collect()
    }
}
