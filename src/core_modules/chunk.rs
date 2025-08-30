pub mod chunk {
    use crate::core_modules::pixel::pixel::Pixel;

    pub struct Chunk {
        pub width: u32,
        pub height: u32,
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

            Pixel {
                red: (sum_r / num_pixels as u32) as u8,
                green: (sum_g / num_pixels as u32) as u8,
                blue: (sum_b / num_pixels as u32) as u8,
                alpha: (sum_a / num_pixels as u32) as u8,
            }
        }
    }
}
