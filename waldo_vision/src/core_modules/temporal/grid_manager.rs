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
//     layer (the `Blob` spatial analyzer).
// 4.  **Decoupling**: It decouples the main application logic from the chunk analysis
//     logic. The main loop will only need to interact with the `GridManager`, giving
//     it a new frame and receiving a status map, without needing to know the
//     complex inner workings of the `SmartChunk`s.

use crate::core_modules::data::chunk::chunk::Chunk;
use crate::core_modules::data::pixel::pixel::{Pixel, Pixels};
use crate::core_modules::heuristics::heuristics::{
    HeuristicContext, HeuristicRegistry, RegionFeatures, evaluate_region_features,
};
use crate::core_modules::pooling::spatial_pooler::spatial_pooler::{
    GaussianPooler, PreparedFrame, SpatialPooler,
};
use crate::core_modules::temporal::smart_chunk::{ChunkStatus, SmartChunk};
use crate::core_modules::utils::region::Region;

/// Manages the entire grid of `SmartChunk`s and orchestrates the temporal analysis layer.
pub struct GridManager {
    /// The width of the full image in pixels, needed for chunk extraction math.
    image_width: u32,
    /// The height of the full image in pixels.
    image_height: u32,
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
    /// How to summarize regions spatially before temporal analysis.
    pooling_strategy: PoolingStrategy,
    /// Optional set of heuristics to evaluate per chunk region.
    heuristics: Option<HeuristicRegistry>,
}

/// Spatial pooling strategy for summarizing a region before temporal analysis.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PoolingStrategy {
    /// Legacy uniform box average inside the chunk region.
    Box,
    /// Gaussian-weighted local mean sampled at region center. Sigma derived from chunk size.
    Gaussian,
}

impl GridManager {
    /// Creates a new GridManager for a given image dimension and chunk size.
    pub fn new(image_width: u32, image_height: u32, chunk_width: u32, chunk_height: u32) -> Self {
        let grid_width = image_width / chunk_width;
        let grid_height = image_height / chunk_height;
        let num_chunks = (grid_width * grid_height) as usize;
        let mut smart_chunks = Vec::with_capacity(num_chunks);

        // Use a single, flattened loop for consistency with the processing logic.
        for i in 0..num_chunks {
            let y = i as u32 / grid_width;
            let x = i as u32 % grid_width;
            smart_chunks.push(SmartChunk::new(x, y));
        }

        Self {
            image_width,
            image_height,
            grid_width,
            grid_height,
            chunk_width,
            chunk_height,
            smart_chunks,
            pooling_strategy: PoolingStrategy::Gaussian,
            heuristics: None,
        }
    }

    /// Enable Gaussian pooling. Uses `sigma ≈ 0.5 * chunk_width` for linear RGB blurs.
    pub fn enable_gaussian_pooling(&mut self) {
        self.pooling_strategy = PoolingStrategy::Gaussian;
    }

    /// Attach a set of heuristics to be evaluated per chunk region during processing.
    pub fn set_heuristics(&mut self, registry: HeuristicRegistry) {
        self.heuristics = Some(registry);
    }

    /// Remove any attached heuristics.
    pub fn clear_heuristics(&mut self) {
        self.heuristics = None;
    }

    /// The main entry point for the vision system.
    /// Takes a raw RGBA image buffer, processes it, and returns a map of chunk statuses.
    pub fn process_frame(&mut self, frame_buffer: &[u8]) -> Vec<ChunkStatus> {
        let (_status, _features) = self.process_frame_with_features(frame_buffer);
        _status
    }

    /// Process a frame and optionally compute per-chunk heuristic features.
    /// Returns (status_map, features_per_chunk) where the second item is None if no heuristics are attached.
    pub fn process_frame_with_features(
        &mut self,
        frame_buffer: &[u8],
    ) -> (Vec<ChunkStatus>, Option<Vec<RegionFeatures>>) {
        let mut features_out: Option<Vec<RegionFeatures>> = self
            .heuristics
            .as_ref()
            .map(|_| Vec::with_capacity(self.smart_chunks.len()));

        match self.pooling_strategy {
            PoolingStrategy::Box => {
                // Legacy path: extract chunk pixels directly and average in Chunk::average_pixel.
                // If heuristics are attached, also prepare a PreparedFrame for planes.
                let (mut prepared_frame_if_any, _image_width_usize, _image_height_usize) =
                    if self.heuristics.is_some() {
                        let pixels: Vec<Pixel> = Pixels::try_from(frame_buffer.to_vec())
                            .expect("Invalid RGBA buffer length for image dimensions")
                            .into();
                        (
                            Some(PreparedFrame::new(
                                pixels,
                                self.image_width as usize,
                                self.image_height as usize,
                            )),
                            self.image_width as usize,
                            self.image_height as usize,
                        )
                    } else {
                        (None, self.image_width as usize, self.image_height as usize)
                    };

                for chunk_index in 0..self.smart_chunks.len() {
                    let chunk_y = chunk_index as u32 / self.grid_width;
                    let chunk_x = chunk_index as u32 % self.grid_width;

                    let start_pixel_x = chunk_x * self.chunk_width;
                    let start_pixel_y = chunk_y * self.chunk_height;

                    let mut chunk_pixels =
                        Vec::with_capacity((self.chunk_width * self.chunk_height) as usize);

                    for i in 0..(self.chunk_width * self.chunk_height) {
                        let y_offset = i / self.chunk_width;
                        let x_offset = i % self.chunk_width;

                        let pixel_y = start_pixel_y + y_offset;
                        let pixel_x = start_pixel_x + x_offset;

                        let byte_index = ((pixel_y * self.image_width) + pixel_x) * 4;
                        let pixel_bytes =
                            &frame_buffer[byte_index as usize..(byte_index + 4) as usize];
                        chunk_pixels.push(Pixel::from(pixel_bytes));
                    }

                    let chunk_data = Chunk::new(self.chunk_width, self.chunk_height, chunk_pixels);
                    self.smart_chunks[chunk_index].update(&chunk_data);

                    if let (Some(registry), Some(prepared_frame)) =
                        (self.heuristics.as_ref(), prepared_frame_if_any.as_mut())
                    {
                        let region = Region::new(
                            chunk_x * self.chunk_width,
                            chunk_y * self.chunk_height,
                            self.chunk_width,
                            self.chunk_height,
                        );
                        let mut context = HeuristicContext {
                            prepared_frame,
                            region,
                        };
                        let feature_values = evaluate_region_features(registry, &mut context);
                        if let Some(feature_vectors) = features_out.as_mut() {
                            feature_vectors.push(feature_values);
                        }
                    }
                }
            }
            PoolingStrategy::Gaussian => {
                // Convert entire frame to Pixels once.
                let pixels: Vec<Pixel> = Pixels::try_from(frame_buffer.to_vec())
                    .expect("Invalid RGBA buffer length for image dimensions")
                    .into();

                // Prepare Gaussian pooler and frame-wide planes.
                let sigma = (self.chunk_width as f32) * 0.5f32;
                let pooler = GaussianPooler { sigma };
                let mut pf = pooler.prepare(
                    pixels,
                    self.image_width as usize,
                    self.image_height as usize,
                );

                for chunk_index in 0..self.smart_chunks.len() {
                    let chunk_y = chunk_index as u32 / self.grid_width;
                    let chunk_x = chunk_index as u32 % self.grid_width;
                    let region = Region::new(
                        chunk_x * self.chunk_width,
                        chunk_y * self.chunk_height,
                        self.chunk_width,
                        self.chunk_height,
                    );

                    // Sample Gaussian-averaged pixel at region center.
                    let gaussian_pixel = pooler.sample_region_center_pixel(&mut pf, region);
                    // Wrap in a minimal Chunk (1x1) so downstream temporal logic is unchanged.
                    let chunk_data = Chunk::new(1, 1, vec![gaussian_pixel]);
                    self.smart_chunks[chunk_index].update(&chunk_data);

                    if let Some(registry) = self.heuristics.as_ref() {
                        let mut context = HeuristicContext {
                            prepared_frame: &mut pf,
                            region,
                        };
                        let feature_values = evaluate_region_features(registry, &mut context);
                        if let Some(feature_vectors) = features_out.as_mut() {
                            feature_vectors.push(feature_values);
                        }
                    }
                }
            }
        }

        let status = self
            .smart_chunks
            .iter()
            .map(|sc| sc.status.clone())
            .collect();
        (status, features_out)
    }
}
