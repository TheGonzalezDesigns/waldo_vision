// THEORY:
// The `Chunk` module represents a spatial grouping of pixels. It is a critical component
// for bridging the gap between low-level pixel analysis and high-level scene understanding.
// Its primary purpose is to act as a unit of regional analysis, providing both efficiency
// and noise reduction.
//
// Key architectural principles:
// 1.  **Spatial Pooling**: By grouping pixels (e.g., in a 10x10 block), we move from
//     analyzing millions of individual pixels to tens of thousands of chunks. This is a
//     massive performance optimization.
// 2.  **Noise Reduction**: The core operation of a chunk is `average_pixel`. This single
//     calculation effectively cancels out random, single-pixel noise (like camera sensor
//     artifacts), ensuring that we only focus on spatially coherent changes.
// 3.  **Data Container**: Like `Pixel`, `Chunk` is a "dumb" data container. It holds a
//     `Vec<Pixel>` and knows how to perform summary calculations on its own data. It does
//     not know how to compare itself to other chunks.
//
// The output of a `Chunk` (its average `Pixel`) becomes the input for our temporal
// analysis pipeline, allowing us to efficiently determine if a region of the image
// has changed significantly over time.

pub mod chunk {
    use crate::core_modules::D1::pixel::pixel::{Pixel, Channel};

    /// A "dumb" data container representing a rectangular block of pixels.
    pub struct Chunk {
        /// The width of the chunk in pixels.
        pub width: u32,
        /// The height of the chunk in pixels.
        pub height: u32,
        /// A flattened vector containing all the `Pixel` data within this chunk.
        pub pixels: Vec<Pixel>,
    }

    impl Chunk {
        pub fn new(width: u32, height: u32, pixels: Vec<Pixel>) -> Self {
            // In a real-world scenario, you might add a check here
            // to ensure pixels.len() == (width * height) as usize
            Self {
                width,
                height,
                pixels,
            }
        }

        /// Calculates the average pixel value for the entire chunk.
        /// This is the core operation for summarizing the chunk's state.
        pub fn average_pixel(&self) -> Pixel {
            let num_pixels = self.pixels.len();
            if num_pixels == 0 {
                return Pixel::default(); // Return a default pixel if the chunk is empty
            }

            // Use u32 for sums to prevent overflow when adding many u8 values.
            let mut sum_r: u32 = 0;
            let mut sum_g: u32 = 0;
            let mut sum_b: u32 = 0;
            let mut sum_a: u32 = 0;

            for pixel in &self.pixels {
                sum_r += pixel.red as u32;
                sum_g += pixel.green as u32;
                sum_b += pixel.blue as u32;
                sum_a += pixel.alpha as u32;
            }

            let red = (sum_r / num_pixels as u32) as Channel;
            let green = (sum_g / num_pixels as u32) as Channel;
            let blue = (sum_b / num_pixels as u32) as Channel;
            let alpha = (sum_a / num_pixels as u32) as Channel;

            Pixel::new(red, green, blue, alpha)
        }
    }
}
