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
    use crate::core_modules::utils::color_ops::{
        achromaticity_from_linear, achromaticity_from_norm, channel_stddev_from_linear,
        channel_stddev_from_norm, chroma_from_linear, chroma_from_norm,
        chromaticity_xy_from_linear, chromaticity_xy_from_norm, colorfulness_from_linear,
        colorfulness_from_norm, hue_from_linear_rgb_deg, hue_from_normalized_rgb_deg,
        lightness_hsl_from_linear, lightness_hsl_from_norm, luminance_601_computed_0_255,
        saturation_hsl_from_linear, saturation_hsl_from_norm, saturation_hsv_from_linear,
        saturation_hsv_from_norm, value_from_linear, value_from_norm,
    };
    use std::sync::OnceLock;
    pub type Byte = u8;
    pub type Bytes = Vec<Byte>;
    pub type Channel = Byte;
    pub type FloatType = f64;
    pub type ComputedChannel = FloatType;
    pub type NormalizedChannel = FloatType;
    pub type LinearizedChannel = FloatType;
    pub type Hue = FloatType;
    pub type HueBias = FloatType;
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
    const RADIANS_240: FloatType = 4.1887902047863905;

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
            luminance_601_computed_0_255(self.red_computed, self.green_computed, self.blue_computed)
        }

        /// Hue angle in degrees [0, 360) — optimal (fast) variant.
        ///
        /// Represents:
        /// - The color “family” (red, green, blue, etc.) as an angle on a color wheel.
        ///
        /// Use/interpretation:
        /// - Fastest for real-time bucketing and coarse color filters.
        /// - Slight bias vs true linear RGB because it uses sRGB-normalized channels.
        /// - Wraps to [0, 360); values near 0 and 360 represent similar hues (reds).
        ///
        /// Implementation detail:
        /// - Uses normalized sRGB (`self.*_normalized`), no linearization.
        pub fn hue_optimal(&self) -> Hue {
            hue_from_normalized_rgb_deg(
                self.red_normalized as f64,
                self.green_normalized as f64,
                self.blue_normalized as f64,
            )
        }

        /// Hue angle in degrees [0, 360) — accurate (linear RGB) variant.
        ///
        /// Represents:
        /// - Same as `hue_optimal`, but computed in linear RGB for color correctness.
        ///
        /// Use/interpretation:
        /// - Prefer for analytics, signatures, and perceptual comparisons.
        /// - More stable near neutral colors and across brightness changes.
        ///
        /// Implementation detail:
        /// - Uses precomputed linear channels (`self.red_linear`, etc.).
        pub fn hue_accurate(&self) -> Hue {
            hue_from_linear_rgb_deg(
                self.red_linear as f32,
                self.green_linear as f32,
                self.blue_linear as f32,
            )
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

        pub fn hue_bias(&self) -> HueBias {
            0.5 * (1.0 + (self.hue().to_radians() - RADIANS_240).cos())
        }

        /// =================================Heuristics==================================

        /// Fast brightness proxy: sum of computed RGB channels (0..255 scale each).
        ///
        /// Represents:
        /// - Crude brightness estimate (not perceptual).
        ///
        /// Use/interpretation:
        /// - Lightweight thresholding, quick deltas for motion heuristics.
        /// - Not gamma-aware; use `luminance()` for perceptual brightness.
        ///
        /// Implementation detail:
        /// - Avoids casts by using `ComputedChannel` fields.
        pub fn sum(&self) -> Sum {
            self.red_computed + self.green_computed + self.blue_computed
        }

        /// Per-channel contribution ratios (R, G, B) that sum to 1.0.
        ///
        /// Represents:
        /// - Relative contributions of R, G, B to the pixel’s color.
        ///
        /// Use/interpretation:
        /// - Simple, brightness-invariant color descriptor.
        /// - Useful for hue-like distance or clustering in RGB space.
        ///
        /// Implementation detail:
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
        ///
        /// Represents:
        /// - Brightness in HSV; not perceptually uniform.
        ///
        /// Use/interpretation:
        /// - Quick brightness gating; combine with saturation for vividness.
        ///
        /// Implementation detail:
        /// - Optimal: uses normalized sRGB; fastest.
        pub fn value_hsv_optimal(&self) -> ValueHSV {
            value_from_norm(
                self.red_normalized as f64,
                self.green_normalized as f64,
                self.blue_normalized as f64,
            )
        }

        /// HSV Value (V): brightness defined as max(R, G, B).
        /// - Accurate: uses linear RGB; more faithful under gamma.
        pub fn value_hsv_accurate(&self) -> ValueHSV {
            value_from_linear(
                self.red_linear as f64,
                self.green_linear as f64,
                self.blue_linear as f64,
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
        ///
        /// Represents:
        /// - “Lightness” in HSL; better balance across shadows/highlights than HSV Value.
        ///
        /// Use/interpretation:
        /// - Useful for UI/theming transforms and light/dark segregation.
        ///
        /// Implementation detail:
        /// - Optimal uses normalized sRGB.
        pub fn lightness_hsl_optimal(&self) -> LightnessHSL {
            lightness_hsl_from_norm(
                self.red_normalized as f64,
                self.green_normalized as f64,
                self.blue_normalized as f64,
            )
        }

        /// HSL Lightness (L): midpoint of max and min channels (linear RGB).
        /// - Accurate uses linear RGB; preferred for analysis.
        pub fn lightness_hsl_accurate(&self) -> LightnessHSL {
            lightness_hsl_from_linear(
                self.red_linear as f64,
                self.green_linear as f64,
                self.blue_linear as f64,
            )
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
        ///
        /// Represents:
        /// - Distance from gray along the color axes (ignoring brightness).
        ///
        /// Use/interpretation:
        /// - Higher chroma → more vivid color; pair with value/lightness for context.
        ///
        /// Implementation detail:
        /// - Optimal uses normalized sRGB (fast heuristic).
        pub fn chroma_optimal(&self) -> Chroma {
            chroma_from_norm(
                self.red_normalized as f64,
                self.green_normalized as f64,
                self.blue_normalized as f64,
            )
        }

        /// Chroma (C): color purity = max(R,G,B) - min(R,G,B) in linear RGB.
        /// - Accurate uses linear RGB; better behaved under gamma and near gray.
        pub fn chroma_accurate(&self) -> Chroma {
            chroma_from_linear(
                self.red_linear as f64,
                self.green_linear as f64,
                self.blue_linear as f64,
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
        ///
        /// Represents:
        /// - Colorfulness relative to brightness; drops near black.
        ///
        /// Use/interpretation:
        /// - Good for “how vivid is this pixel right now?”
        ///
        /// Implementation detail:
        /// - Optimal uses normalized sRGB.
        pub fn saturation_hsv_optimal(&self) -> SaturationHSV {
            saturation_hsv_from_norm(
                self.red_normalized as f64,
                self.green_normalized as f64,
                self.blue_normalized as f64,
            )
        }

        /// Saturation (HSV): S = chroma / value (linear RGB).
        /// - Accurate uses linear RGB; more consistent across tones.
        pub fn saturation_hsv_accurate(&self) -> SaturationHSV {
            saturation_hsv_from_linear(
                self.red_linear as f64,
                self.green_linear as f64,
                self.blue_linear as f64,
            )
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
        ///
        /// Represents:
        /// - Colorfulness normalized by lightness; more even than HSV saturation.
        ///
        /// Use/interpretation:
        /// - Better for UI/graphics workflows where perceptual consistency matters.
        ///
        /// Implementation detail:
        /// - Optimal uses normalized sRGB.
        pub fn saturation_hsl_optimal(&self) -> SaturationHSL {
            saturation_hsl_from_norm(
                self.red_normalized as f64,
                self.green_normalized as f64,
                self.blue_normalized as f64,
            )
        }

        /// Saturation (HSL): S = chroma / (1 - |2L - 1|) (linear RGB).
        /// - Accurate uses linear RGB; more stable across tones.
        pub fn saturation_hsl_accurate(&self) -> SaturationHSL {
            saturation_hsl_from_linear(
                self.red_linear as f64,
                self.green_linear as f64,
                self.blue_linear as f64,
            )
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
        ///
        /// Represents:
        /// - Overall “vividness” of the pixel’s color.
        ///
        /// Use/interpretation:
        /// - Quick feature for segmentation or saliency.
        ///
        /// Implementation detail:
        /// - Optimal uses sRGB; accurate uses linear.
        pub fn colorfulness_optimal(&self) -> Colorfulness {
            colorfulness_from_norm(
                self.red_normalized as f64,
                self.green_normalized as f64,
                self.blue_normalized as f64,
            )
        }

        pub fn colorfulness_accurate(&self) -> Colorfulness {
            colorfulness_from_linear(
                self.red_linear as f64,
                self.green_linear as f64,
                self.blue_linear as f64,
            )
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
        ///
        /// Represents:
        /// - Degree of “grayness” at the current brightness.
        ///
        /// Use/interpretation:
        /// - 1.0 → fully gray; 0.0 → maximally saturated at that Value.
        ///
        /// Implementation detail:
        /// - Optimal uses sRGB; accurate uses linear.
        pub fn achromaticity_optimal(&self) -> Achromaticity {
            achromaticity_from_norm(
                self.red_normalized as f64,
                self.green_normalized as f64,
                self.blue_normalized as f64,
            )
        }

        pub fn achromaticity_accurate(&self) -> Achromaticity {
            achromaticity_from_linear(
                self.red_linear as f64,
                self.green_linear as f64,
                self.blue_linear as f64,
            )
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
        ///
        /// Represents:
        /// - Channel spread; 0.0 for gray, larger for colorful pixels.
        ///
        /// Use/interpretation:
        /// - Simple measure of how “colored” a pixel is, agnostic to hue.
        ///
        /// Implementation detail:
        /// - Optimal uses sRGB; accurate uses linear.
        pub fn channel_stddev_optimal(&self) -> ChannelStdDev {
            channel_stddev_from_norm(
                self.red_normalized as f64,
                self.green_normalized as f64,
                self.blue_normalized as f64,
            )
        }

        pub fn channel_stddev_accurate(&self) -> ChannelStdDev {
            channel_stddev_from_linear(
                self.red_linear as f64,
                self.green_linear as f64,
                self.blue_linear as f64,
            )
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
        ///
        /// Represents:
        /// - The color’s location on the CIE chromaticity diagram (x,y), independent of
        ///   overall luminance. Captures “what color it is” rather than “how bright”.
        ///
        /// Use/interpretation:
        /// - Useful for estimating correlated color temperature (CCT) and comparing colors
        ///   across brightness changes. Often paired with Y (luminance) for full XYZ.
        ///
        /// Implementation detail:
        /// - Converts RGB → XYZ (D65), then x = X/(X+Y+Z), y = Y/(X+Y+Z).
        /// - Optimal uses normalized sRGB (quick estimate); accurate uses linear RGB.
        pub fn chromaticity_xy_optimal(&self) -> (ChromaticityX, ChromaticityY) {
            chromaticity_xy_from_norm(
                self.red_normalized as f64,
                self.green_normalized as f64,
                self.blue_normalized as f64,
            )
        }

        pub fn chromaticity_xy_accurate(&self) -> (ChromaticityX, ChromaticityY) {
            chromaticity_xy_from_linear(
                self.red_linear as f64,
                self.green_linear as f64,
                self.blue_linear as f64,
            )
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

    /// Newtype wrapper to enable safe conversions without violating orphan rules.
    #[derive(Debug, Clone, PartialEq)]
    pub struct Pixels(pub Vec<Pixel>);

    impl From<Pixels> for Vec<Pixel> {
        fn from(pv: Pixels) -> Self {
            pv.0
        }
    }

    impl Pixels {
        pub fn into_inner(self) -> Vec<Pixel> {
            self.0
        }
    }

    /// Convert RGBA bytes to `Pixels` with validation.
    impl TryFrom<&[Byte]> for Pixels {
        type Error = &'static str;
        fn try_from(bytes: &[Byte]) -> Result<Self, Self::Error> {
            if bytes.is_empty() {
                return Err("byte buffer is empty");
            }
            if bytes.len() % CHANNELS != 0 {
                return Err("byte buffer length must be a multiple of 4 (RGBA)");
            }
            Ok(Pixels(
                bytes.chunks_exact(CHANNELS).map(Pixel::from).collect(),
            ))
        }
    }

    /// Owned variant: accepts `Vec<u8>` and converts to `Pixels`.
    impl TryFrom<Bytes> for Pixels {
        type Error = &'static str;
        fn try_from(bytes: Bytes) -> Result<Self, Self::Error> {
            Pixels::try_from(bytes.as_slice())
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
// - LUT (Lookup Table): A precomputed array used to replace repeated runtime
//   computation with a fast indexed lookup. Here, a 256‑entry sRGB→linear table
//   converts each 8‑bit channel (0..255) to its linear‑RGB value in 0..1. The table
//   is initialized once (thread‑safe via OnceLock), then reused, turning expensive
//   powf(2.4) evaluations into constant‑time memory reads. Typical cost: ~1 KB RAM.
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
