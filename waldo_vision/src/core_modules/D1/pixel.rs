// THEORY (1D Pixel Heuristics):
// The `Pixel` module is the most fundamental unit of the vision system. It is a
// "dumb" data container for a single pixel plus a set of 1‑dimensional heuristics —
// metrics that can be computed from this pixel alone, with no knowledge of neighbors
// in space or time. Anything that needs another pixel (comparisons, gradients, motion)
// belongs in higher‑dimension modules like `SmartPixel` (pairwise) or 2D/3D.
//
// What lives here (by design):
// - Raw channels (RGBA) and three common transforms of those channels
//   • ComputedChannel (0..255):            1:1 numeric copy of the byte values
//   • NormalizedChannel (0..1, sRGB):      divide by 255.0, still gamma‑encoded
//   • LinearizedChannel (0..1, linear RGB): sRGB → linear light, normalized
//   Alpha is not gamma‑encoded and is not linearized — it is passed through.
//
// Why so many channel forms?
// - normalized is convenient for ratios and bounded math
// - linearized is correct for colorimetry (hue/saturation/lightness in true RGB space)
// - computed mirrors the raw data range for lightweight arithmetic
//
// Heuristic families (all single‑pixel):
// - Brightness:   luminance (Rec. 601), sum, HSV value (max), HSL lightness (midpoint)
// - Color strength: chroma (max−min), saturation_hsv (chroma/value),
//                   saturation_hsv (chroma / (1−|2L−1|)), colorfulness ≈ chroma,
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
    pub type FloatType = f64;
    pub type ComputedChannel = FloatType;
    pub type NormalizedChannel = FloatType;
    pub type LinearizedChannel = FloatType;
    pub type Hue = FloatType;
    pub type SaturationHSV = FloatType;
    pub type SaturationHSL = FloatType;
    pub type ValueHSV = FloatType;
    pub type LightnessHSL = FloatType;
    pub type Chroma = FloatType;
    pub type Colorfulness = FloatType;
    pub type ChromaticityX = FloatType;
    pub type ChromaticityY = FloatType;
    pub type ChannelStdDev = FloatType;
    pub type Luminance = FloatType;
    pub type Color = i16;
    pub type Sum = FloatType;
    pub type Achromaticity = FloatType;
    pub type ColorRatios = (FloatType, FloatType, FloatType);

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
        /// The linearized red channel value (0.0-1.0, sRGB gamma-decoded).
        pub red_linear: LinearizedChannel,
        /// The linearized green channel value (0.0-1.0, sRGB gamma-decoded).
        pub green_linear: LinearizedChannel,
        /// The linearized blue channel value (0.0-1.0, sRGB gamma-decoded).
        pub blue_linear: LinearizedChannel,
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
                red_linear: LinearizedChannel::default(),
                green_linear: LinearizedChannel::default(),
                blue_linear: LinearizedChannel::default(),
                red_normalized: NormalizedChannel::default(),
                green_normalized: NormalizedChannel::default(),
                blue_normalized: NormalizedChannel::default(),
                alpha_normalized: NormalizedChannel::default(),
            }
        }
    }

    impl Pixel {
        pub fn new(red: Channel, green: Channel, blue: Channel, alpha: Channel) -> Self {
            // Precompute linear channels once per pixel (single LUT hit each), 0..1 linear.
            let red_linear_value = Self::srgb_to_linear_normalized_from_byte(red);
            let green_linear_value = Self::srgb_to_linear_normalized_from_byte(green);
            let blue_linear_value = Self::srgb_to_linear_normalized_from_byte(blue);

            Pixel {
                red,
                green,
                blue,
                alpha,
                red_computed: red as ComputedChannel,
                green_computed: green as ComputedChannel,
                blue_computed: blue as ComputedChannel,
                alpha_computed: alpha as ComputedChannel,
                red_linear: red_linear_value,
                green_linear: green_linear_value,
                blue_linear: blue_linear_value,
                red_normalized: red as NormalizedChannel / 255.0,
                green_normalized: green as NormalizedChannel / 255.0,
                blue_normalized: blue as NormalizedChannel / 255.0,
                alpha_normalized: alpha as NormalizedChannel / 255.0,
            }
        }

        // Fast path LUT is defined at module scope to avoid associated `static`.

        #[inline]
        fn srgb_to_linear_normalized_from_byte(srgb_value: Byte) -> NormalizedChannel {
            let table = SRGB_TO_LINEAR_LUT.get_or_init(|| {
                let mut table: [NormalizedChannel; 256] = [0.0; 256];
                let mut i = 0usize;
                while i < 256 {
                    let srgb_normalized: NormalizedChannel = i as NormalizedChannel / 255.0;
                    table[i] = if srgb_normalized <= 0.04045 {
                        srgb_normalized / 12.92
                    } else {
                        ((srgb_normalized + 0.055) / 1.055).powf(2.4)
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
        /// - Uses 0..255 channels via `ComputedChannel`.
        pub fn luminance(&self) -> Luminance {
            0.299 * self.red_computed + 0.587 * self.green_computed + 0.114 * self.blue_computed
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
            let maximum_channel = self.red_linear.max(self.green_linear.max(self.blue_linear));
            let minimum_channel = self.red_linear.min(self.green_linear.min(self.blue_linear));
            let chroma = maximum_channel - minimum_channel;

            if chroma <= 1e-6 {
                return 0.0;
            }

            let inverse_chroma = 1.0 / chroma;

            let (base_difference, sector_offset) = if maximum_channel == self.red_linear {
                (self.green_linear - self.blue_linear, 0.0)
            } else if maximum_channel == self.green_linear {
                (self.blue_linear - self.red_linear, 2.0)
            } else {
                (self.red_linear - self.green_linear, 4.0)
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

        /// Fast brightness proxy: sum of computed RGB channels (0..255 scale each).
        /// - Avoids repeated casting by using precomputed `ComputedChannel` fields.
        pub fn sum(&self) -> Sum {
            self.red_computed + self.green_computed + self.blue_computed
        }

        /// Per-channel contribution ratios (R, G, B) that sum to 1.0.
        /// - Uses `ComputedChannel` values to avoid extra casts.
        pub fn color_ratios(&self) -> ColorRatios {
            let sum = self.sum();
            if sum == 0.0 {
                return (0.0, 0.0, 0.0);
            }
            (
                self.red_computed / sum,
                self.green_computed / sum,
                self.blue_computed / sum,
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
            self.red_linear.max(self.green_linear.max(self.blue_linear))
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
            let maximum_channel = self.red_linear.max(self.green_linear.max(self.blue_linear));
            let minimum_channel = self.red_linear.min(self.green_linear.min(self.blue_linear));
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
            self.red_linear.max(self.green_linear.max(self.blue_linear))
                - self.red_linear.min(self.green_linear.min(self.blue_linear))
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
            let maximum_channel = self.red_linear.max(self.green_linear.max(self.blue_linear));
            if maximum_channel <= 1e-6 {
                return 0.0;
            }
            (maximum_channel - self.red_linear.min(self.green_linear.min(self.blue_linear)))
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
        pub fn achromaticity_optimal(&self) -> Achromaticity {
            let maximum_channel = self
                .red_normalized
                .max(self.green_normalized.max(self.blue_normalized));
            if maximum_channel <= 1e-6 {
                return 1.0;
            }
            1.0 - self.chroma_optimal() / maximum_channel
        }

        pub fn achromaticity_accurate(&self) -> Achromaticity {
            let maximum_channel = self.red_linear.max(self.green_linear.max(self.blue_linear));
            if maximum_channel <= 1e-6 {
                return 1.0;
            }
            1.0 - (maximum_channel - self.red_linear.min(self.green_linear.min(self.blue_linear)))
                / maximum_channel
        }

        #[cfg(feature = "accurate")]
        pub fn achromaticity(&self) -> Achromaticity {
            self.achromaticity_accurate()
        }

        #[cfg(not(feature = "accurate"))]
        pub fn achromaticity(&self) -> Achromaticity {
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
            let mean = (self.red_linear + self.green_linear + self.blue_linear) / 3.0;
            let vr = (self.red_linear - mean).powi(2);
            let vg = (self.green_linear - mean).powi(2);
            let vb = (self.blue_linear - mean).powi(2);
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
            let x = 0.4124564 * self.red_normalized
                + 0.3575761 * self.green_normalized
                + 0.1804375 * self.blue_normalized;
            let y = 0.2126729 * self.red_normalized
                + 0.7151522 * self.green_normalized
                + 0.0721750 * self.blue_normalized;
            let z = 0.0193339 * self.red_normalized
                + 0.1191920 * self.green_normalized
                + 0.9503041 * self.blue_normalized;
            let sum = x + y + z;
            if sum <= 1e-12 {
                return (0.0, 0.0);
            }
            (x / sum, y / sum)
        }

        pub fn chromaticity_xy_accurate(&self) -> (ChromaticityX, ChromaticityY) {
            let x = 0.4124564 * self.red_linear
                + 0.3575761 * self.green_linear
                + 0.1804375 * self.blue_linear;
            let y = 0.2126729 * self.red_linear
                + 0.7151522 * self.green_linear
                + 0.0721750 * self.blue_linear;
            let z = 0.0193339 * self.red_linear
                + 0.1191920 * self.green_linear
                + 0.9503041 * self.blue_linear;
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
// - ComputedChannel: The 0..255 channel stored in floating precision.
//
// - NormalizedChannel (sRGB): Channel scaled to 0..1 but still gamma-encoded. Great
//   for quick ratios and bounded math, not strictly “linear” to light.
//
// - LinearizedChannel: Gamma-decoded channel proportional to light intensity,
//   in 0..1. Most “accurate” heuristics operate on this form directly.
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
