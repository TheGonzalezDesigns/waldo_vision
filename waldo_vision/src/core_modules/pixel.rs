// THEORY:
// The `Pixel` module serves as the most fundamental building block of our vision system.
// It is designed as a "dumb" data container, meaning its primary responsibility is to
// represent the raw RGBA data of a single pixel accurately and efficiently.
//
// Key architectural principles:
// 1.  **Data Purity**: It holds the raw `u8` channel values without any interpretation.
// 2.  **Intrinsic Knowledge**: It contains methods (`luminance`, `sum`, `color_ratios`)
//     that calculate properties based *only* on the pixel's own internal data. It knows
//     nothing about other pixels.
// 3.  **Efficiency**: By being a simple, transparent struct, it is fast to create, copy,
//     and store in large collections like `Vec<Pixel>`.
//
// This module intentionally separates the concept of "what a pixel is" from the more
// complex question of "how a pixel relates to others," which is handled by `SmartPixel`.

pub mod pixel {
    pub type Byte = u8;
    pub type Bytes = Vec<Byte>;
    pub type Channel = Byte;
    pub type Luminance = f64;
    pub type Color = i16;
    pub type Sum = f32;

    const CHANNELS: usize = 4;

    /// A "dumb" data container representing a single RGBA pixel.
    #[derive(Debug, Clone, PartialEq)]
    pub struct Pixel {
        /// The red channel value (0-255).
        pub red: Channel,
        /// The green channel value (0-255).
        pub green: Channel,
        /// The blue channel value (0-255).
        pub blue: Channel,
        /// The alpha (transparency) channel value (0-255).
        pub alpha: Channel,
    }

    impl Default for Pixel {
        fn default() -> Self {
            Pixel {
                red: Channel::default(),
                green: Channel::default(),
                blue: Channel::default(),
                alpha: Channel::default(),
            }
        }
    }

    impl Pixel {
        pub fn new(red: Channel, green: Channel, blue: Channel, alpha: Channel) -> Self {
            Pixel {
                red,
                green,
                blue,
                alpha,
            }
        }

        pub fn luminance(&self) -> Luminance {
            0.299 * self.red as f64 + 0.587 * self.green as f64 + 0.114 * self.blue as f64
        }

        pub fn sum(&self) -> Sum {
            (self.red as Color + self.green as Color + self.blue as Color) as Sum
        }

        pub fn color_ratios(&self) -> (f32, f32, f32) {
            let sum = self.sum();
            if sum == 0.0 {
                return (0.0, 0.0, 0.0);
            }
            (
                self.red as f32 / sum,
                self.green as f32 / sum,
                self.blue as f32 / sum,
            )
        }
    }

    impl From<&[Byte]> for Pixel {
        fn from(bytes: &[Byte]) -> Self {
            if bytes.len() != CHANNELS {
                panic!("Cannot convert {} bytes into pixel.", bytes.len());
            }
            Pixel::new(bytes[0], bytes[1], bytes[2], bytes[3])
        }
    }

    impl From<Pixel> for Bytes {
        fn from(pixel: Pixel) -> Self {
            vec![pixel.red, pixel.green, pixel.blue, pixel.alpha]
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn test_pixel_new() {
            let pixel = Pixel::new(10, 20, 30, 40);
            assert_eq!(pixel.red, 10);
            assert_eq!(pixel.green, 20);
            assert_eq!(pixel.blue, 30);
            assert_eq!(pixel.alpha, 40);
        }

        #[test]
        fn test_pixel_default() {
            let pixel = Pixel::default();
            assert_eq!(pixel.red, 0);
            assert_eq!(pixel.green, 0);
            assert_eq!(pixel.blue, 0);
            assert_eq!(pixel.alpha, 0);
        }

        #[test]
        fn test_pixel_luminance() {
            let pixel = Pixel::new(255, 255, 255, 255);
            assert!((pixel.luminance() - 255.0).abs() < 1e-6);

            let black = Pixel::new(0, 0, 0, 255);
            assert_eq!(black.luminance(), 0.0);

            let red = Pixel::new(255, 0, 0, 255);
            assert!((red.luminance() - 76.245).abs() < 1e-3);
        }

        #[test]
        fn test_pixel_sum() {
            let pixel = Pixel::new(10, 20, 30, 255);
            assert_eq!(pixel.sum(), 60.0);
        }

        #[test]
        fn test_pixel_color_ratios() {
            let pixel = Pixel::new(100, 100, 200, 255);
            let (r, g, b) = pixel.color_ratios();
            assert!((r - 0.25f32).abs() < 1e-6);
            assert!((g - 0.25f32).abs() < 1e-6);
            assert!((b - 0.50f32).abs() < 1e-6);

            let black = Pixel::new(0, 0, 0, 255);
            assert_eq!(black.color_ratios(), (0.0, 0.0, 0.0));
        }

        #[test]
        fn test_pixel_from_bytes() {
            let bytes = [10, 20, 30, 40];
            let pixel = Pixel::from(&bytes[..]);
            assert_eq!(pixel, Pixel::new(10, 20, 30, 40));
        }

        #[test]
        #[should_panic(expected = "Cannot convert 3 bytes into pixel.")]
        fn test_pixel_from_bytes_panic() {
            let bytes = [10, 20, 30];
            let _ = Pixel::from(&bytes[..]);
        }

        #[test]
        fn test_bytes_from_pixel() {
            let pixel = Pixel::new(10, 20, 30, 40);
            let bytes: Bytes = pixel.into();
            assert_eq!(bytes, vec![10, 20, 30, 40]);
        }
    }
}
