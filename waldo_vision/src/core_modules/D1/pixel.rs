// THEORY (1D Pixel Heuristics):
// The `Pixel` module is the most fundamental unit of the vision system. It is a
// "dumb" data container for a single pixel plus a set of 1‑dimensional heuristics —
// metrics that can be computed from this pixel alone, with no knowledge of neighbors
// in space or time. Anything that needs another pixel (comparisons, gradients, motion)
// belongs in higher‑dimension modules like `SmartPixel` (pairwise) or 2D/3D.
//
// What lives here (by design):
// - Raw channels (RGBA) and three common transforms of those channels
//   • computed (0..255 as f32): 1:1 numeric copy of the byte values
//   • normalized (0..1 sRGB):   divide by 255.0, still gamma‑encoded
//   • linearized (0..255 linear): sRGB → linear light, scaled by 255.0
//   Alpha is not gamma‑encoded and is not linearized — it is passed through.
//
// Why so many channel forms?
// - normalized is convenient for ratios and bounded math
// - linearized is correct for colorimetry (hue/saturation/lightness in true RGB space)
// - computed mirrors the raw data range for lightweight arithmetic
//
// Heuristic families (all single‑pixel):
// - Brightness:   luminance (Rec. 601), sum, HSV value (max), HSL lightness (midpoint)
// - Color strength: chroma (max−min), saturation_HSV (chroma/value),
//                   saturation_HSL (chroma / (1−|2L−1|)), colorfulness ≈ chroma,
//                   achromaticity (inverse of saturation relative to value)
// - Hue:          angle on the color wheel in degrees [0, 360)
// - Spread:       channel standard deviation across R,G,B
// - Chromaticity: (x,y) from XYZ (D65) for color temperature/correlates (estimates)
//
// Optimal vs Accurate (feature‑selected):
// - optimal (default): fastest, uses normalized sRGB directly — great for realtime
// - accurate: uses sRGB→linear LUT for color‑correct math — best for analytics
// Enable accurate with Cargo features: `--features accurate`. Without it, optimal is used.
// Internally, the sRGB→linear conversion uses a 256‑entry `OnceLock` LUT; the hot path
// is a table lookup and a multiply — no expensive `powf` per pixel.
//
// Key principles:
// 1) Single‑pixel scope (1D): Heuristics never read neighbors or history.
// 2) Clear separation of concerns: higher‑dimension logic lives elsewhere.
// 3) Efficiency and clarity: minimal temporaries, precomputed channels, documented intent.

pub mod pixel {
    use std::sync::OnceLock;
    pub type Byte = u8;
    pub type Bytes = Vec<Byte>;
    pub type Channel = Byte;
    pub type ComputedChannel = f32;
    pub type NormalizedChannel = f32;
    pub type LinearizedChannel = f32;
    pub type Hue = f32;
    pub type SaturationHSV = f32;
    pub type SaturationHSL = f32;
    pub type ValueHSV = f32;
    pub type LightnessHSL = f32;
    pub type Chroma = f32;
    pub type Colorfulness = f32;
    pub type ChromaticityX = f32;
    pub type ChromaticityY = f32;
    pub type ChannelStdDev = f32;
    pub type Luminance = f64;
    pub type Color = i16;
    pub type Sum = f32;

    const CHANNELS: usize = 4;

    // Fast path: 256-entry LUT for sRGB (0..255) -> linear normalized (0..1)
    static SRGB_TO_LINEAR_LUT: OnceLock<[NormalizedChannel; 256]> = OnceLock::new();

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
            // Zero-cost default: no derived computations, just zeros.
            Pixel {
                red: Channel::default(),
                green: Channel::default(),
                blue: Channel::default(),
                alpha: Channel::default(),
                red_computed: ComputedChannel::default(),
                green_computed: ComputedChannel::default(),
                blue_computed: ComputedChannel::default(),
                alpha_computed: ComputedChannel::default(),
                red_linearized: LinearizedChannel::default(),
                green_linearized: LinearizedChannel::default(),
                blue_linearized: LinearizedChannel::default(),
                red_normalized: NormalizedChannel::default(),
                green_normalized: NormalizedChannel::default(),
                blue_normalized: NormalizedChannel::default(),
                alpha_normalized: NormalizedChannel::default(),
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

        // Fast path LUT is defined at module scope to avoid associated `static`.

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

        /// Luminance estimate (Rec. 601 luma).
        ///
        /// - Interprets perceived brightness as a weighted sum of RGB.
        /// - Useful for fast brightness thresholds and motion/heat maps.
        /// - Uses 0..255 computed channels; cast to f64 for stability.
        pub fn luminance(&self) -> Luminance {
            // Keep legacy definition using 0..255 computed channels; cast to f64 for stability
            0.299_f64 * self.red_computed as f64
                + 0.587_f64 * self.green_computed as f64
                + 0.114_f64 * self.blue_computed as f64
        }

        /// Backward-compatible alias for HSL lightness.
        /// Delegates to feature-selected `lightness_hsl()` (optimal default, accurate with feature).
        pub fn lightness_HSL(&self) -> LightnessHSL {
            self.lightness_hsl()
        }

        /// Backward-compatible alias for HSL saturation.
        /// Delegates to feature-selected `saturation_hsl()` (optimal default, accurate with feature).
        pub fn saturation_HSL(&self) -> SaturationHSL {
            self.saturation_hsl()
        }

        /// Backward-compatible alias for HSV saturation.
        /// Delegates to feature-selected `saturation_hsv()` (optimal default, accurate with feature).
        pub fn saturation_HSV(&self) -> SaturationHSV {
            self.saturation_hsv()
        }

        /// Hue angle in degrees [0, 360) — optimal (fast) variant.
        ///
        /// - Uses normalized sRGB channels (`self.*_normalized`), no linearization.
        /// - Fastest path for heuristics, slight bias vs true linear RGB.
        /// - Good for coarse color bucketing or real-time pipelines.
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

        /// Hue angle in degrees [0, 360) — accurate (colorimetrically correct) variant.
        ///
        /// - Uses LUT-linearized normalized channels (sRGB → linear) for correctness.
        /// - Prefer this when color fidelity matters (analytics, feature extraction).
        pub fn hue_accurate(&self) -> Hue {
            let red_linear_normalized = Self::srgb_to_linear_normalized_from_byte(self.red);
            let green_linear_normalized = Self::srgb_to_linear_normalized_from_byte(self.green);
            let blue_linear_normalized = Self::srgb_to_linear_normalized_from_byte(self.blue);

            let maximum_channel =
                red_linear_normalized.max(green_linear_normalized.max(blue_linear_normalized));
            let minimum_channel =
                red_linear_normalized.min(green_linear_normalized.min(blue_linear_normalized));
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

        /// Feature-selected hue angle in degrees [0, 360).
        /// - Default (no features or `optimal`): uses `hue_optimal()`.
        /// - With feature `accurate`: uses `hue_accurate()`.
        #[cfg(feature = "accurate")]
        pub fn hue(&self) -> Hue {
            self.hue_accurate()
        }

        #[cfg(not(feature = "accurate"))]
        pub fn hue(&self) -> Hue {
            self.hue_optimal()
        }

        /// =================================Heuristics==================================

        /// Fast brightness proxy: raw RGB channel sum (0..255 scale each).
        /// - Cheap heuristic useful for quick thresholds and deltas.
        pub fn sum(&self) -> Sum {
            (self.red as Color + self.green as Color + self.blue as Color) as Sum
        }

        /// Per-channel contribution ratios (R, G, B) that sum to 1.0.
        /// - Useful for hue-like comparisons or color distance metrics.
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

        /// HSV Value (V): brightness defined as max(R, G, B).
        /// - Optimal: uses normalized sRGB; fastest.
        pub fn value_hsv_optimal(&self) -> ValueHSV {
            self.red_normalized
                .max(self.green_normalized.max(self.blue_normalized))
        }

        /// HSV Value (V): brightness defined as max(R, G, B).
        /// - Accurate: uses LUT-linearized normalized channels.
        pub fn value_hsv_accurate(&self) -> ValueHSV {
            Self::srgb_to_linear_normalized_from_byte(self.red).max(
                Self::srgb_to_linear_normalized_from_byte(self.green)
                    .max(Self::srgb_to_linear_normalized_from_byte(self.blue)),
            )
        }

        #[cfg(feature = "accurate")]
        pub fn value_hsv(&self) -> ValueHSV {
            self.value_hsv_accurate()
        }

        #[cfg(not(feature = "accurate"))]
        pub fn value_hsv(&self) -> ValueHSV {
            self.value_hsv_optimal()
        }

        /// HSL Lightness (L): midpoint of max and min channels.
        /// - Optimal: uses normalized sRGB.
        pub fn lightness_hsl_optimal(&self) -> LightnessHSL {
            let maximum_channel = self
                .red_normalized
                .max(self.green_normalized.max(self.blue_normalized));
            let minimum_channel = self
                .red_normalized
                .min(self.green_normalized.min(self.blue_normalized));
            (maximum_channel + minimum_channel) * 0.5
        }

        /// HSL Lightness (L): midpoint of max and min channels.
        /// - Accurate: uses LUT-linearized normalized channels.
        pub fn lightness_hsl_accurate(&self) -> LightnessHSL {
            let maximum_channel = Self::srgb_to_linear_normalized_from_byte(self.red).max(
                Self::srgb_to_linear_normalized_from_byte(self.green)
                    .max(Self::srgb_to_linear_normalized_from_byte(self.blue)),
            );
            let minimum_channel = Self::srgb_to_linear_normalized_from_byte(self.red).min(
                Self::srgb_to_linear_normalized_from_byte(self.green)
                    .min(Self::srgb_to_linear_normalized_from_byte(self.blue)),
            );
            (maximum_channel + minimum_channel) * 0.5
        }

        #[cfg(feature = "accurate")]
        pub fn lightness_hsl(&self) -> LightnessHSL {
            self.lightness_hsl_accurate()
        }

        #[cfg(not(feature = "accurate"))]
        pub fn lightness_hsl(&self) -> LightnessHSL {
            self.lightness_hsl_optimal()
        }

        /// Chroma (C): color purity = max(R,G,B) - min(R,G,B).
        /// - Optimal: uses normalized sRGB (fast heuristic).
        pub fn chroma_optimal(&self) -> Chroma {
            self.red_normalized
                .max(self.green_normalized.max(self.blue_normalized))
                - self
                    .red_normalized
                    .min(self.green_normalized.min(self.blue_normalized))
        }

        /// Chroma (C): color purity = max(R,G,B) - min(R,G,B).
        /// - Accurate: uses LUT-linearized normalized channels.
        pub fn chroma_accurate(&self) -> Chroma {
            Self::srgb_to_linear_normalized_from_byte(self.red).max(
                Self::srgb_to_linear_normalized_from_byte(self.green)
                    .max(Self::srgb_to_linear_normalized_from_byte(self.blue)),
            ) - Self::srgb_to_linear_normalized_from_byte(self.red).min(
                Self::srgb_to_linear_normalized_from_byte(self.green)
                    .min(Self::srgb_to_linear_normalized_from_byte(self.blue)),
            )
        }

        #[cfg(feature = "accurate")]
        pub fn chroma(&self) -> Chroma {
            self.chroma_accurate()
        }

        #[cfg(not(feature = "accurate"))]
        pub fn chroma(&self) -> Chroma {
            self.chroma_optimal()
        }

        /// Saturation (HSV): S = chroma / value.
        /// - Measures distance from gray relative to Value (max channel).
        /// - Optimal: uses normalized sRGB.
        pub fn saturation_hsv_optimal(&self) -> SaturationHSV {
            let maximum_channel = self
                .red_normalized
                .max(self.green_normalized.max(self.blue_normalized));
            if maximum_channel <= 1e-6 {
                return 0.0;
            }
            self.chroma_optimal() / maximum_channel
        }

        /// Saturation (HSV): S = chroma / value.
        /// - Accurate: uses LUT-linearized normalized channels.
        pub fn saturation_hsv_accurate(&self) -> SaturationHSV {
            let maximum_channel = Self::srgb_to_linear_normalized_from_byte(self.red).max(
                Self::srgb_to_linear_normalized_from_byte(self.green)
                    .max(Self::srgb_to_linear_normalized_from_byte(self.blue)),
            );
            if maximum_channel <= 1e-6 {
                return 0.0;
            }
            (maximum_channel
                - Self::srgb_to_linear_normalized_from_byte(self.red).min(
                    Self::srgb_to_linear_normalized_from_byte(self.green)
                        .min(Self::srgb_to_linear_normalized_from_byte(self.blue)),
                ))
                / maximum_channel
        }

        #[cfg(feature = "accurate")]
        pub fn saturation_hsv(&self) -> SaturationHSV {
            self.saturation_hsv_accurate()
        }

        #[cfg(not(feature = "accurate"))]
        pub fn saturation_hsv(&self) -> SaturationHSV {
            self.saturation_hsv_optimal()
        }

        /// Saturation (HSL): S = chroma / (1 - |2L - 1|), where L is HSL lightness.
        /// - More even across lightness; better perceptual uniformity than HSV saturation.
        /// - Optimal: uses normalized sRGB.
        pub fn saturation_hsl_optimal(&self) -> SaturationHSL {
            let lightness = self.lightness_hsl_optimal();
            let denominator = 1.0 - (2.0 * lightness - 1.0).abs();
            if denominator <= 1e-6 {
                return 0.0;
            }
            self.chroma_optimal() / denominator
        }

        /// Saturation (HSL): S = chroma / (1 - |2L - 1|), where L is HSL lightness.
        /// - Accurate: uses LUT-linearized normalized channels.
        pub fn saturation_hsl_accurate(&self) -> SaturationHSL {
            let lightness = self.lightness_hsl_accurate();
            let denominator = 1.0 - (2.0 * lightness - 1.0).abs();
            if denominator <= 1e-6 {
                return 0.0;
            }
            self.chroma_accurate() / denominator
        }

        #[cfg(feature = "accurate")]
        pub fn saturation_hsl(&self) -> SaturationHSL {
            self.saturation_hsl_accurate()
        }

        #[cfg(not(feature = "accurate"))]
        pub fn saturation_hsl(&self) -> SaturationHSL {
            self.saturation_hsl_optimal()
        }

        /// Colorfulness: simple proxy ≈ chroma.
        /// - Higher means more vivid (further from gray).
        /// - Optimal uses sRGB; accurate uses linear.
        pub fn colorfulness_optimal(&self) -> Colorfulness {
            self.chroma_optimal()
        }

        pub fn colorfulness_accurate(&self) -> Colorfulness {
            self.chroma_accurate()
        }

        #[cfg(feature = "accurate")]
        pub fn colorfulness(&self) -> Colorfulness {
            self.colorfulness_accurate()
        }

        #[cfg(not(feature = "accurate"))]
        pub fn colorfulness(&self) -> Colorfulness {
            self.colorfulness_optimal()
        }

        /// Achromaticity: inverse of saturation w.r.t. Value.
        /// - 1.0 means fully gray; 0.0 means fully saturated at that Value.
        /// - Optimal uses sRGB; accurate uses linear.
        pub fn achromaticity_optimal(&self) -> f32 {
            let maximum_channel = self
                .red_normalized
                .max(self.green_normalized.max(self.blue_normalized));
            if maximum_channel <= 1e-6 {
                return 1.0;
            }
            1.0 - self.chroma_optimal() / maximum_channel
        }

        pub fn achromaticity_accurate(&self) -> f32 {
            let maximum_channel = Self::srgb_to_linear_normalized_from_byte(self.red).max(
                Self::srgb_to_linear_normalized_from_byte(self.green)
                    .max(Self::srgb_to_linear_normalized_from_byte(self.blue)),
            );
            if maximum_channel <= 1e-6 {
                return 1.0;
            }
            1.0 - (maximum_channel
                - Self::srgb_to_linear_normalized_from_byte(self.red).min(
                    Self::srgb_to_linear_normalized_from_byte(self.green)
                        .min(Self::srgb_to_linear_normalized_from_byte(self.blue)),
                ))
                / maximum_channel
        }

        #[cfg(feature = "accurate")]
        pub fn achromaticity(&self) -> f32 {
            self.achromaticity_accurate()
        }

        #[cfg(not(feature = "accurate"))]
        pub fn achromaticity(&self) -> f32 {
            self.achromaticity_optimal()
        }

        /// Standard deviation across R,G,B channels.
        /// - Measures channel spread; 0.0 for perfect gray, higher for colorful pixels.
        /// - Optimal uses sRGB; accurate uses linear.
        pub fn channel_stddev_optimal(&self) -> ChannelStdDev {
            let mean = (self.red_normalized + self.green_normalized + self.blue_normalized) / 3.0;
            let vr = (self.red_normalized - mean).powi(2);
            let vg = (self.green_normalized - mean).powi(2);
            let vb = (self.blue_normalized - mean).powi(2);
            ((vr + vg + vb) / 3.0).sqrt()
        }

        pub fn channel_stddev_accurate(&self) -> ChannelStdDev {
            let mean = (Self::srgb_to_linear_normalized_from_byte(self.red)
                + Self::srgb_to_linear_normalized_from_byte(self.green)
                + Self::srgb_to_linear_normalized_from_byte(self.blue))
                / 3.0;
            let vr = (Self::srgb_to_linear_normalized_from_byte(self.red) - mean).powi(2);
            let vg = (Self::srgb_to_linear_normalized_from_byte(self.green) - mean).powi(2);
            let vb = (Self::srgb_to_linear_normalized_from_byte(self.blue) - mean).powi(2);
            ((vr + vg + vb) / 3.0).sqrt()
        }

        #[cfg(feature = "accurate")]
        pub fn channel_stddev(&self) -> ChannelStdDev {
            self.channel_stddev_accurate()
        }

        #[cfg(not(feature = "accurate"))]
        pub fn channel_stddev(&self) -> ChannelStdDev {
            self.channel_stddev_optimal()
        }

        /// Chromaticity (CIE x,y) from RGB.
        /// - Converts RGB → XYZ (D65) and returns x = X/(X+Y+Z), y = Y/(X+Y+Z).
        /// - Optimal uses normalized sRGB (quick estimate); accurate uses linear RGB.
        pub fn chromaticity_xy_optimal(&self) -> (ChromaticityX, ChromaticityY) {
            let x = 0.4124564f32 * self.red_normalized
                + 0.3575761f32 * self.green_normalized
                + 0.1804375f32 * self.blue_normalized;
            let y = 0.2126729f32 * self.red_normalized
                + 0.7151522f32 * self.green_normalized
                + 0.0721750f32 * self.blue_normalized;
            let z = 0.0193339f32 * self.red_normalized
                + 0.1191920f32 * self.green_normalized
                + 0.9503041f32 * self.blue_normalized;
            let sum = x + y + z;
            if sum <= 1e-12 {
                return (0.0, 0.0);
            }
            (x / sum, y / sum)
        }

        pub fn chromaticity_xy_accurate(&self) -> (ChromaticityX, ChromaticityY) {
            let x = 0.4124564f32 * Self::srgb_to_linear_normalized_from_byte(self.red)
                + 0.3575761f32 * Self::srgb_to_linear_normalized_from_byte(self.green)
                + 0.1804375f32 * Self::srgb_to_linear_normalized_from_byte(self.blue);
            let y = 0.2126729f32 * Self::srgb_to_linear_normalized_from_byte(self.red)
                + 0.7151522f32 * Self::srgb_to_linear_normalized_from_byte(self.green)
                + 0.0721750f32 * Self::srgb_to_linear_normalized_from_byte(self.blue);
            let z = 0.0193339f32 * Self::srgb_to_linear_normalized_from_byte(self.red)
                + 0.1191920f32 * Self::srgb_to_linear_normalized_from_byte(self.green)
                + 0.9503041f32 * Self::srgb_to_linear_normalized_from_byte(self.blue);
            let sum = x + y + z;
            if sum <= 1e-12 {
                return (0.0, 0.0);
            }
            (x / sum, y / sum)
        }

        #[cfg(feature = "accurate")]
        pub fn chromaticity_xy(&self) -> (ChromaticityX, ChromaticityY) {
            self.chromaticity_xy_accurate()
        }

        #[cfg(not(feature = "accurate"))]
        pub fn chromaticity_xy(&self) -> (ChromaticityX, ChromaticityY) {
            self.chromaticity_xy_optimal()
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

// -----------------------------------------------------------------------------
// Glossary: Single-Pixel Color Terms (1D)
//
// - Luminance: Perceived brightness from RGB. Here we use a Rec. 601 luma
//   approximation (weighted sum of R,G,B). Useful for thresholding and motion maps.
//
// - Hue: Angle on the color wheel (0°–360°) describing the “color family”
//   (red, green, blue, etc.). Computed from relative differences between channels.
//
// - Value (HSV): Brightness defined as the maximum of the RGB channels. High Value
//   means the pixel is bright regardless of colorfulness.
//
// - Lightness (HSL): Midpoint of the maximum and minimum channels. More balanced
//   across shadows and highlights than HSV Value.
//
// - Chroma: Color purity = max(R,G,B) − min(R,G,B). Zero means perfectly gray; higher
//   values are more vivid.
//
// - Saturation (HSV): Chroma divided by Value. Drops to zero near black, even if hue
//   is well-defined.
//
// - Saturation (HSL): Chroma divided by (1 − |2L − 1|), where L is HSL Lightness.
//   More consistent across dark and bright regions than HSV saturation.
//
// - Colorfulness: A simple proxy for “how vivid” a color is. Here we approximate it by
//   chroma. Higher means further from gray at a given brightness.
//
// - Achromaticity: Inverse of saturation relative to Value. 1.0 means fully gray; 0.0
//   means maximally saturated at that brightness.
//
// - Chromaticity (CIE x,y): Color defined by its proportion of X,Y,Z (with a D65 white
//   point). Independent of overall brightness (Y acts like luminance). Often used as a
//   stepping stone to estimates like correlated color temperature (with caveats).
//
// - Colorimetry: The science and practice of measuring and numerically representing
//   color. Involves color spaces (RGB, XYZ, Lab*), white points (e.g., D65), transfer
//   functions (gamma), and transforms between spaces. In code, this means using
//   linear RGB when doing “geometric” color math, converting to XYZ/Lab* when needed,
//   and comparing colors with perceptual metrics like ΔE (pairwise, higher‑dimension).
//
// - Computed Channel: The 0..255 channel stored as f32 (fast arithmetic in native scale).
//
// - Normalized Channel (sRGB): Channel scaled to 0..1 but still gamma-encoded. Great
//   for quick ratios and bounded math, not strictly “linear” to light.
//
// - Linearized Channel: Gamma-decoded channel proportional to light intensity. Accurate
//   for colorimetry. We use a 256-entry sRGB→linear lookup table for speed.
//
// - Channel Standard Deviation: Spread of R,G,B around their mean. Zero for pure grays;
//   larger for colorful pixels.
//
// - Optimal vs Accurate:
//   • Optimal (default): fastest, uses normalized sRGB; good for realtime heuristics.
//   • Accurate (feature `accurate`): uses linear RGB via LUT for color-correct math.
//
// Note: All items here are 1D (single pixel, no neighbors/time). Multi-pixel or temporal
// heuristics (e.g., contrast vs neighbors, motion, ΔE between pixels) live in higher
// dimension modules like SmartPixel (pairwise) or 2D/3D.
