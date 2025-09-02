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
use tokio::sync::{mpsc, oneshot};
use tokio::task;

/// Message type for SmartChunk actors
enum ChunkMessage {
    Update(Chunk, oneshot::Sender<ChunkStatus>),
    Shutdown,
}

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
    /// Channels to communicate with SmartChunk actors
    chunk_actors: Vec<mpsc::Sender<ChunkMessage>>,
}

impl GridManager {
    /// Creates a new GridManager for a given image dimension and chunk size.
    pub fn new(image_width: u32, image_height: u32, chunk_width: u32, chunk_height: u32) -> Self {
        let grid_width = image_width / chunk_width;
        let grid_height = image_height / chunk_height;
        let num_chunks = (grid_width * grid_height) as usize;
        let mut chunk_actors = Vec::with_capacity(num_chunks);

        // Spawn an actor task for each SmartChunk
        for i in 0..num_chunks {
            let y = i as u32 / grid_width;
            let x = i as u32 % grid_width;
            let (tx, mut rx) = mpsc::channel::<ChunkMessage>(10);
            
            // Spawn actor task that owns its SmartChunk
            tokio::spawn(async move {
                let mut smart_chunk = SmartChunk::new(x, y);
                
                while let Some(msg) = rx.recv().await {
                    match msg {
                        ChunkMessage::Update(chunk_data, reply) => {
                            smart_chunk.update(&chunk_data);
                            let _ = reply.send(smart_chunk.status.clone());
                        }
                        ChunkMessage::Shutdown => break,
                    }
                }
            });
            
            chunk_actors.push(tx);
        }

        Self {
            image_width,
            grid_width,
            grid_height,
            chunk_width,
            chunk_height,
            chunk_actors,
        }
    }

    /// The main entry point for the vision system with actor-based parallelism.
    /// Takes a raw RGBA image buffer, processes it, and returns a map of chunk statuses.
    pub async fn process_frame(&mut self, frame_buffer: &[u8]) -> Vec<ChunkStatus> {
        let num_chunks = self.chunk_actors.len();
        let frame_arc = Arc::new(frame_buffer.to_vec());
        
        // Spawn parallel tasks to extract pixels and send to actors
        let mut update_tasks = Vec::with_capacity(num_chunks);
        
        for chunk_index in 0..num_chunks {
            let chunk_y = chunk_index as u32 / self.grid_width;
            let chunk_x = chunk_index as u32 % self.grid_width;
            let start_pixel_x = chunk_x * self.chunk_width;
            let start_pixel_y = chunk_y * self.chunk_height;
            
            let frame_clone = Arc::clone(&frame_arc);
            let chunk_width = self.chunk_width;
            let chunk_height = self.chunk_height;
            let image_width = self.image_width;
            let tx = self.chunk_actors[chunk_index].clone();
            
            // Spawn task to extract pixels and send to actor
            let task = tokio::spawn(async move {
                // Extract pixels for this chunk
                let mut chunk_pixels = Vec::with_capacity((chunk_width * chunk_height) as usize);
                
                for i in 0..(chunk_width * chunk_height) {
                    let y_offset = i / chunk_width;
                    let x_offset = i % chunk_width;
                    let pixel_y = start_pixel_y + y_offset;
                    let pixel_x = start_pixel_x + x_offset;
                    let byte_index = ((pixel_y * image_width) + pixel_x) * 4;
                    
                    let pixel_bytes = &frame_clone[byte_index as usize..(byte_index + 4) as usize];
                    chunk_pixels.push(Pixel::from(pixel_bytes));
                }
                
                let chunk_data = Chunk::new(chunk_width, chunk_height, chunk_pixels);
                
                // Send to actor and wait for response
                let (reply_tx, reply_rx) = oneshot::channel();
                tx.send(ChunkMessage::Update(chunk_data, reply_tx)).await.ok();
                reply_rx.await.ok()
            });
            
            update_tasks.push(task);
        }
        
        // Await all updates and collect statuses
        let mut status_map = Vec::with_capacity(num_chunks);
        for task in update_tasks {
            if let Ok(Some(status)) = task.await {
                status_map.push(status);
            } else {
                status_map.push(ChunkStatus::Learning);
            }
        }
        
        status_map
    }
    
    /// Shutdown all actor tasks cleanly
    pub async fn shutdown(&self) {
        for tx in &self.chunk_actors {
            let _ = tx.send(ChunkMessage::Shutdown).await;
        }
    }
}

impl Drop for GridManager {
    fn drop(&mut self) {
        // Best effort shutdown on drop
        for tx in &self.chunk_actors {
            let _ = tx.try_send(ChunkMessage::Shutdown);
        }
    }
}