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
    use std::sync::OnceLock;
    pub type Byte = u8;
    pub type Bytes = Vec<Byte>;
    pub type Channel = Byte;
    pub type ComputedChannel = f32;
    pub type NormalizedChannel = f32;
    pub type LinearizedChannel = f32;
    pub type Hue = f32;
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
        /// The Computed red channel value (0.0-255.0).
        pub red_computed: ComputedChannel,
        /// The green channel value (0.0-255.0).
        pub green_computed: ComputedChannel,
        /// The blue channel value (0.0-255.0).
        pub blue_computed: ComputedChannel,
        /// The alpha (transparency) channel value (0.0-255.0).
        pub alpha_computed: ComputedChannel,
        /// The linearized red channel value (0.0-255.0).
        pub red_linearized: LinearizedChannel,
        /// The linearized green channel value (0.0-255.0).
        pub green_linearized: LinearizedChannel,
        /// The linearized blue channel value (0.0-255.0).
        pub blue_linearized: LinearizedChannel,
        // Alpha is not gamma-encoded; keep as-is in linear space. Reuse alpha_computed
        /// The red channel value (0.0-1.0).
        pub red_normalized: NormalizedChannel,
        /// The green channel value (0.0-1.0).
        pub green_normalized: NormalizedChannel,
        /// The blue channel value (0.0-1.-).
        pub blue_normalized: NormalizedChannel,
        /// The alpha (transparency) channel value (0.0-1.0).
        pub alpha_normalized: NormalizedChannel,
    }

    impl Default for Pixel {
        fn default() -> Self {
            Pixel::new(0, 0, 0, 0)
        }
    }

    impl Pixel {
        pub fn new(red: Channel, green: Channel, blue: Channel, alpha: Channel) -> Self {
            Pixel {
                red,
                green,
                blue,
                alpha,
                red_computed: red as ComputedChannel,
                green_computed: green as ComputedChannel,
                blue_computed: blue as ComputedChannel,
                alpha_computed: alpha as ComputedChannel,
                red_linearized: Self::srgb_to_linear_normalized_from_byte(red) * 255.0f32,
                green_linearized: Self::srgb_to_linear_normalized_from_byte(green) * 255.0f32,
                blue_linearized: Self::srgb_to_linear_normalized_from_byte(blue) * 255.0f32,
                red_normalized: red as NormalizedChannel / 255.0f32,
                green_normalized: green as NormalizedChannel / 255.0f32,
                blue_normalized: blue as NormalizedChannel / 255.0f32,
                alpha_normalized: alpha as NormalizedChannel / 255.0f32,
            }
        }

        // Fast path: 256-entry LUT for sRGB (0..255) -> linear normalized (0..1)
        static SRGB_TO_LINEAR_LUT: OnceLock<[NormalizedChannel; 256]> = OnceLock::new();

        #[inline]
        fn srgb_to_linear_normalized_from_byte(srgb_value: Byte) -> NormalizedChannel {
            let table = SRGB_TO_LINEAR_LUT.get_or_init(|| {
                let mut table = [0.0f32; 256];
                let mut i = 0usize;
                while i < 256 {
                    let srgb_normalized = i as NormalizedChannel / 255.0f32;
                    table[i] = if srgb_normalized <= 0.04045f32 {
                        srgb_normalized / 12.92f32
                    } else {
                        ((srgb_normalized + 0.055f32) / 1.055f32).powf(2.4f32)
                    };
                    i += 1;
                }
                table
            });
            table[srgb_value as usize]
        }

        /// =================================Heuristics==================================

        pub fn luminance(&self) -> Luminance {
            0.299 * self.red_computed + 0.587 * self.green_computed + 0.114 * self.blue_computed
        }

        pub fn lightness_HSL(&self) -> LightnessHSL {
            let max = self.red_computed.max(self.blue_computed.max(self.green_computed));
            let min = self.red_computed.min(self.blue_computed.min(self.green_computed));

            (max + min) / 2
        }

        pub fn saturation_HSL(&self) -> SaturationHSL {
            let max = self.red_computed.max(self.blue_computed.max(self.green_computed));
            let min = self.red_computed.min(self.blue_computed.min(self.green_computed));

            (max - min) / (1 - (((max + min) / 2) - 1).abs())
        }

        pub fn saturation_HSV(&self) -> SaturationHSV {
            let max = self.red_computed.max(self.blue_computed.max(self.green_computed));
            let min = self.red_computed.min(self.blue_computed.min(self.green_computed));

            (max - min) / max
        }

        /// Fastest hue using normalized sRGB channels (heuristic, not linearized).
        pub fn hue_optimal(&self) -> Hue {
            let maximum_channel = self
                .red_normalized
                .max(self.green_normalized.max(self.blue_normalized));
            let minimum_channel = self
                .red_normalized
                .min(self.green_normalized.min(self.blue_normalized));
            let chroma = maximum_channel - minimum_channel;

            if chroma <= 1e-6 {
                return 0.0;
            }

            let inverse_chroma = 1.0 / chroma;

            let (base_difference, sector_offset) = if maximum_channel == self.red_normalized {
                (self.green_normalized - self.blue_normalized, 0.0)
            } else if maximum_channel == self.green_normalized {
                (self.blue_normalized - self.red_normalized, 2.0)
            } else {
                (self.red_normalized - self.green_normalized, 4.0)
            };

            let mut hue_degrees = (base_difference * inverse_chroma + sector_offset) * 60.0;
            if hue_degrees < 0.0 {
                hue_degrees += 360.0;
            }
            hue_degrees
        }

        /// Accurate hue using linearized (LUT) normalized channels.
        pub fn hue_accurate(&self) -> Hue {
            let red_linear_normalized = Self::srgb_to_linear_normalized_from_byte(self.red);
            let green_linear_normalized = Self::srgb_to_linear_normalized_from_byte(self.green);
            let blue_linear_normalized = Self::srgb_to_linear_normalized_from_byte(self.blue);

            let maximum_channel = red_linear_normalized
                .max(green_linear_normalized.max(blue_linear_normalized));
            let minimum_channel = red_linear_normalized
                .min(green_linear_normalized.min(blue_linear_normalized));
            let chroma = maximum_channel - minimum_channel;

            if chroma <= 1e-6 {
                return 0.0;
            }

            let inverse_chroma = 1.0 / chroma;

            let (base_difference, sector_offset) = if maximum_channel == red_linear_normalized {
                (green_linear_normalized - blue_linear_normalized, 0.0)
            } else if maximum_channel == green_linear_normalized {
                (blue_linear_normalized - red_linear_normalized, 2.0)
            } else {
                (red_linear_normalized - green_linear_normalized, 4.0)
            };

            let mut hue_degrees = (base_difference * inverse_chroma + sector_offset) * 60.0;
            if hue_degrees < 0.0 {
                hue_degrees += 360.0;
            }
            hue_degrees
        }

        /// Feature-selected hue.
        /// - default (feature optimal or no features): hue_optimal
        /// - with feature accurate:                  hue_accurate
        #[cfg(feature = "accurate")]
        pub fn hue(&self) -> Hue {
            self.hue_accurate()
        }

        #[cfg(not(feature = "accurate"))]
        pub fn hue(&self) -> Hue {
            self.hue_optimal()
        }

        /// =================================Heuristics==================================

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
