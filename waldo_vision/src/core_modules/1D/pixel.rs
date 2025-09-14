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
}
