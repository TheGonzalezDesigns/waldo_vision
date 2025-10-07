pub mod heuristics {
    use crate::core_modules::pooling::spatial_pooler::spatial_pooler::{
        BasePlane, PlaneKey, PreparedFrame,
    };
    use crate::core_modules::utils::region::Region;

    /// Context passed to heuristic evaluation.
    pub struct HeuristicContext<'a> {
        pub prepared_frame: &'a mut PreparedFrame,
        pub region: Region,
    }

    impl<'a> HeuristicContext<'a> {
        /// Sample blurred linear RGB at the region center.
        pub fn sample_blurred_linear_rgb_center(&mut self, sigma: f32) -> (f32, f32, f32) {
            let red = self
                .prepared_frame
                .sample_plane_center(PlaneKey::blurred(BasePlane::RedLinear, sigma), self.region);
            let green = self.prepared_frame.sample_plane_center(
                PlaneKey::blurred(BasePlane::GreenLinear, sigma),
                self.region,
            );
            let blue = self
                .prepared_frame
                .sample_plane_center(PlaneKey::blurred(BasePlane::BlueLinear, sigma), self.region);
            (red, green, blue)
        }
    }

    // Static plane requirements for RGB triplets under each mode.
    #[cfg(feature = "accurate")]
    const REQUIRED_RED_GREEN_BLUE_PLANES_LINEAR: &[PlaneKey] = &[
        PlaneKey {
            base: BasePlane::RedLinear,
            sigma: None,
        },
        PlaneKey {
            base: BasePlane::GreenLinear,
            sigma: None,
        },
        PlaneKey {
            base: BasePlane::BlueLinear,
            sigma: None,
        },
    ];
    #[cfg(not(feature = "accurate"))]
    const REQUIRED_RED_GREEN_BLUE_PLANES_NORMALIZED: &[PlaneKey] = &[
        PlaneKey {
            base: BasePlane::RedNormalized,
            sigma: None,
        },
        PlaneKey {
            base: BasePlane::GreenNormalized,
            sigma: None,
        },
        PlaneKey {
            base: BasePlane::BlueNormalized,
            sigma: None,
        },
    ];

    /// A pluggable scalar feature computed for a region.
    pub trait Heuristic {
        /// Stable identifier for logging/registry schemas.
        fn name(&self) -> &'static str;
        /// Plane requirements (declares which prepared planes are needed).
        fn requires(&self) -> &'static [PlaneKey];
        /// Evaluate feature value for the given region.
        fn evaluate(&self, context: &mut HeuristicContext) -> f64;
    }

    /// Collection of heuristics to evaluate per region.
    pub struct HeuristicRegistry {
        pub heurs: Vec<Box<dyn Heuristic + Send + Sync>>, // flexible for runtime composition
    }

    impl HeuristicRegistry {
        pub fn new() -> Self {
            Self { heurs: Vec::new() }
        }
        pub fn with(mut self, h: Box<dyn Heuristic + Send + Sync>) -> Self {
            self.heurs.push(h);
            self
        }
    }

    /// Output of heuristic evaluation for a region.
    pub struct RegionFeatures {
        pub feature_names: Vec<&'static str>,
        pub feature_values: Vec<f64>,
    }

    pub fn evaluate_region_features(
        registry: &HeuristicRegistry,
        context: &mut HeuristicContext,
    ) -> RegionFeatures {
        // Pre-ensure declared planes to avoid redundant compute.
        for heuristic in &registry.heurs {
            for &key in heuristic.requires() {
                context.prepared_frame.ensure_plane(key);
            }
        }

        // Evaluate in registration order.
        let mut feature_names = Vec::with_capacity(registry.heurs.len());
        let mut feature_values = Vec::with_capacity(registry.heurs.len());
        for heuristic in &registry.heurs {
            feature_names.push(heuristic.name());
            feature_values.push(heuristic.evaluate(context));
        }
        RegionFeatures {
            feature_names,
            feature_values,
        }
    }

    // ---------------------------- Built-in heuristics ----------------------------

    /// Local Brightness Contrast C_V at region center.
    /// C_V = clamp01((V̄ - V)/max(V̄, eps)).
    pub struct LocalBrightnessContrast {
        pub sigma: f32,
        pub epsilon: f32,
    }

    impl Heuristic for LocalBrightnessContrast {
        fn name(&self) -> &'static str {
            "local_brightness_contrast"
        }
        fn requires(&self) -> &'static [PlaneKey] {
            // NOTE: We cannot return a dynamically-sigma key here as 'static. The planes are ensured in eval.
            // For static declaration, we declare the base Value plane which is always needed.
            const REQS: &[PlaneKey] = &[PlaneKey {
                base: BasePlane::Value,
                sigma: None,
            }];
            REQS
        }
        fn evaluate(&self, context: &mut HeuristicContext) -> f64 {
            let value_at_center = context
                .prepared_frame
                .sample_plane_center(PlaneKey::base(BasePlane::Value), context.region)
                as f64;
            let neighborhood_mean = context.prepared_frame.sample_plane_center(
                PlaneKey::blurred(BasePlane::Value, self.sigma),
                context.region,
            ) as f64;
            let denominator = neighborhood_mean.max(self.epsilon as f64);
            let raw_contrast = (neighborhood_mean - value_at_center) / denominator;
            raw_contrast.max(0.0).min(1.0)
        }
    }

    /// Desaturation far-ness: 1 - S at region center.
    pub struct DesaturationFar;

    impl Heuristic for DesaturationFar {
        fn name(&self) -> &'static str {
            "desaturation_far"
        }
        fn requires(&self) -> &'static [PlaneKey] {
            const REQS: &[PlaneKey] = &[PlaneKey {
                base: BasePlane::Saturation,
                sigma: None,
            }];
            REQS
        }
        fn evaluate(&self, context: &mut HeuristicContext) -> f64 {
            let saturation = context
                .prepared_frame
                .sample_plane_center(PlaneKey::base(BasePlane::Saturation), context.region)
                as f64;
            (1.0 - saturation).max(0.0).min(1.0)
        }
    }

    /// Hue bias at region center (already in [0,1]). Useful as a depth cue input.
    pub struct HueBias;

    impl Heuristic for HueBias {
        fn name(&self) -> &'static str {
            "hue_bias"
        }
        fn requires(&self) -> &'static [PlaneKey] {
            const REQS: &[PlaneKey] = &[PlaneKey {
                base: BasePlane::HueBias,
                sigma: None,
            }];
            REQS
        }
        fn evaluate(&self, context: &mut HeuristicContext) -> f64 {
            context
                .prepared_frame
                .sample_plane_center(PlaneKey::base(BasePlane::HueBias), context.region)
                as f64
        }
    }

    // ---------------- Gaussian-mapped equivalents of classic Pixel heuristics ----------------

    /// Gaussian Luminance scaled to 0..255, driven by the global color mode.
    /// - Optimal: Rec.601 on normalized sRGB (≈ fast)
    /// - Accurate: Rec.709 on linear RGB (color-correct)
    pub struct GaussianLuminance {
        pub sigma: f32,
    }

    impl Heuristic for GaussianLuminance {
        fn name(&self) -> &'static str {
            "gaussian_luminance"
        }
        fn requires(&self) -> &'static [PlaneKey] {
            #[cfg(feature = "accurate")]
            {
                REQUIRED_RED_GREEN_BLUE_PLANES_LINEAR
            }
            #[cfg(not(feature = "accurate"))]
            {
                REQUIRED_RED_GREEN_BLUE_PLANES_NORMALIZED
            }
        }
        fn evaluate(&self, context: &mut HeuristicContext) -> f64 {
            #[cfg(feature = "accurate")]
            {
                use crate::core_modules::utils::color_ops::luminance_709_linear_0_255;
                let (red, green, blue) = context.sample_blurred_linear_rgb_center(self.sigma);
                return luminance_709_linear_0_255(red, green, blue);
            }
            #[cfg(not(feature = "accurate"))]
            {
                use crate::core_modules::utils::color_ops::luminance_601_norm_0_255;
                let red = context.prepared_frame.sample_plane_center(
                    PlaneKey::blurred(BasePlane::RedNormalized, self.sigma),
                    context.region,
                ) as f64;
                let green = context.prepared_frame.sample_plane_center(
                    PlaneKey::blurred(BasePlane::GreenNormalized, self.sigma),
                    context.region,
                ) as f64;
                let blue = context.prepared_frame.sample_plane_center(
                    PlaneKey::blurred(BasePlane::BlueNormalized, self.sigma),
                    context.region,
                ) as f64;
                return luminance_601_norm_0_255(red, green, blue);
            }
        }
    }

    /// Gaussian Color Sum, scaled to 0..765 (3 * 255), driven by color mode.
    /// - Optimal: sum of normalized sRGB * 255
    /// - Accurate: sum of linear RGB * 255
    pub struct GaussianColorSum {
        pub sigma: f32,
    }

    impl Heuristic for GaussianColorSum {
        fn name(&self) -> &'static str {
            "gaussian_color_sum"
        }
        fn requires(&self) -> &'static [PlaneKey] {
            #[cfg(feature = "accurate")]
            {
                REQUIRED_RED_GREEN_BLUE_PLANES_LINEAR
            }
            #[cfg(not(feature = "accurate"))]
            {
                REQUIRED_RED_GREEN_BLUE_PLANES_NORMALIZED
            }
        }
        fn evaluate(&self, context: &mut HeuristicContext) -> f64 {
            #[cfg(feature = "accurate")]
            {
                use crate::core_modules::utils::color_ops::color_sum_linear_0_765;
                let (red, green, blue) = context.sample_blurred_linear_rgb_center(self.sigma);
                return color_sum_linear_0_765(red, green, blue);
            }
            #[cfg(not(feature = "accurate"))]
            {
                use crate::core_modules::utils::color_ops::color_sum_norm_0_765;
                let red = context.prepared_frame.sample_plane_center(
                    PlaneKey::blurred(BasePlane::RedNormalized, self.sigma),
                    context.region,
                ) as f64;
                let green = context.prepared_frame.sample_plane_center(
                    PlaneKey::blurred(BasePlane::GreenNormalized, self.sigma),
                    context.region,
                ) as f64;
                let blue = context.prepared_frame.sample_plane_center(
                    PlaneKey::blurred(BasePlane::BlueNormalized, self.sigma),
                    context.region,
                ) as f64;
                return color_sum_norm_0_765(red, green, blue);
            }
        }
    }

    /// Gaussian Hue computed from blurred RGB at the region center, driven by color mode.
    pub struct GaussianHue {
        pub sigma: f32,
    }

    impl Heuristic for GaussianHue {
        fn name(&self) -> &'static str {
            "gaussian_hue"
        }
        fn requires(&self) -> &'static [PlaneKey] {
            #[cfg(feature = "accurate")]
            {
                REQUIRED_RED_GREEN_BLUE_PLANES_LINEAR
            }
            #[cfg(not(feature = "accurate"))]
            {
                REQUIRED_RED_GREEN_BLUE_PLANES_NORMALIZED
            }
        }
        fn evaluate(&self, context: &mut HeuristicContext) -> f64 {
            #[cfg(feature = "accurate")]
            {
                use crate::core_modules::utils::color_ops::{lab_hue_deg, rgb_linear_to_lab};
                let (red, green, blue) = context.sample_blurred_linear_rgb_center(self.sigma);
                let (_l, a, b) = rgb_linear_to_lab(red as f64, green as f64, blue as f64);
                return lab_hue_deg(a, b);
            }
            #[cfg(not(feature = "accurate"))]
            {
                use crate::core_modules::utils::color_ops::hue_from_normalized_rgb_deg;
                let red = context.prepared_frame.sample_plane_center(
                    PlaneKey::blurred(BasePlane::RedNormalized, self.sigma),
                    context.region,
                ) as f64;
                let green = context.prepared_frame.sample_plane_center(
                    PlaneKey::blurred(BasePlane::GreenNormalized, self.sigma),
                    context.region,
                ) as f64;
                let blue = context.prepared_frame.sample_plane_center(
                    PlaneKey::blurred(BasePlane::BlueNormalized, self.sigma),
                    context.region,
                ) as f64;
                return hue_from_normalized_rgb_deg(red, green, blue);
            }
        }
    }

    // ------------------------- More Gaussian reinterpretations -------------------------

    /// Linear HSL Lightness at center from blurred planes (0..1).
    pub struct GaussianLightnessHsl {
        pub sigma: f32,
    }
    impl Heuristic for GaussianLightnessHsl {
        fn name(&self) -> &'static str {
            "gaussian_lightness_hsl"
        }
        fn requires(&self) -> &'static [PlaneKey] {
            #[cfg(feature = "accurate")]
            {
                REQUIRED_RED_GREEN_BLUE_PLANES_LINEAR
            }
            #[cfg(not(feature = "accurate"))]
            {
                REQUIRED_RED_GREEN_BLUE_PLANES_NORMALIZED
            }
        }
        fn evaluate(&self, context: &mut HeuristicContext) -> f64 {
            #[cfg(feature = "accurate")]
            {
                use crate::core_modules::utils::color_ops::lightness_hsl_from_linear;
                let (red, green, blue) = context.sample_blurred_linear_rgb_center(self.sigma);
                return lightness_hsl_from_linear(red as f64, green as f64, blue as f64);
            }
            #[cfg(not(feature = "accurate"))]
            {
                use crate::core_modules::utils::color_ops::lightness_hsl_from_norm;
                let red = context.prepared_frame.sample_plane_center(
                    PlaneKey::blurred(BasePlane::RedNormalized, self.sigma),
                    context.region,
                ) as f64;
                let green = context.prepared_frame.sample_plane_center(
                    PlaneKey::blurred(BasePlane::GreenNormalized, self.sigma),
                    context.region,
                ) as f64;
                let blue = context.prepared_frame.sample_plane_center(
                    PlaneKey::blurred(BasePlane::BlueNormalized, self.sigma),
                    context.region,
                ) as f64;
                return lightness_hsl_from_norm(red, green, blue);
            }
        }
    }

    /// Linear Chroma (max-min) at center from blurred planes (0..1).
    pub struct GaussianChroma {
        pub sigma: f32,
    }
    impl Heuristic for GaussianChroma {
        fn name(&self) -> &'static str {
            "gaussian_chroma"
        }
        fn requires(&self) -> &'static [PlaneKey] {
            #[cfg(feature = "accurate")]
            {
                REQUIRED_RED_GREEN_BLUE_PLANES_LINEAR
            }
            #[cfg(not(feature = "accurate"))]
            {
                REQUIRED_RED_GREEN_BLUE_PLANES_NORMALIZED
            }
        }
        fn evaluate(&self, context: &mut HeuristicContext) -> f64 {
            #[cfg(feature = "accurate")]
            {
                use crate::core_modules::utils::color_ops::{
                    lab_chroma_normalized, rgb_linear_to_lab,
                };
                let (red, green, blue) = context.sample_blurred_linear_rgb_center(self.sigma);
                let (_l, a, b) = rgb_linear_to_lab(red as f64, green as f64, blue as f64);
                return lab_chroma_normalized(a, b);
            }
            #[cfg(not(feature = "accurate"))]
            {
                use crate::core_modules::utils::color_ops::chroma_from_norm;
                let red = context.prepared_frame.sample_plane_center(
                    PlaneKey::blurred(BasePlane::RedNormalized, self.sigma),
                    context.region,
                ) as f64;
                let green = context.prepared_frame.sample_plane_center(
                    PlaneKey::blurred(BasePlane::GreenNormalized, self.sigma),
                    context.region,
                ) as f64;
                let blue = context.prepared_frame.sample_plane_center(
                    PlaneKey::blurred(BasePlane::BlueNormalized, self.sigma),
                    context.region,
                ) as f64;
                return chroma_from_norm(red, green, blue);
            }
        }
    }

    /// Linear HSV Saturation at center from blurred planes (0..1).
    pub struct GaussianSaturationHsv {
        pub sigma: f32,
    }
    impl Heuristic for GaussianSaturationHsv {
        fn name(&self) -> &'static str {
            "gaussian_saturation_hsv"
        }
        fn requires(&self) -> &'static [PlaneKey] {
            #[cfg(feature = "accurate")]
            {
                REQUIRED_RED_GREEN_BLUE_PLANES_LINEAR
            }
            #[cfg(not(feature = "accurate"))]
            {
                REQUIRED_RED_GREEN_BLUE_PLANES_NORMALIZED
            }
        }
        fn evaluate(&self, context: &mut HeuristicContext) -> f64 {
            #[cfg(feature = "accurate")]
            {
                use crate::core_modules::utils::color_ops::saturation_hsv_from_linear;
                let (red, green, blue) = context.sample_blurred_linear_rgb_center(self.sigma);
                return saturation_hsv_from_linear(red as f64, green as f64, blue as f64);
            }
            #[cfg(not(feature = "accurate"))]
            {
                use crate::core_modules::utils::color_ops::saturation_hsv_from_norm;
                let red = context.prepared_frame.sample_plane_center(
                    PlaneKey::blurred(BasePlane::RedNormalized, self.sigma),
                    context.region,
                ) as f64;
                let green = context.prepared_frame.sample_plane_center(
                    PlaneKey::blurred(BasePlane::GreenNormalized, self.sigma),
                    context.region,
                ) as f64;
                let blue = context.prepared_frame.sample_plane_center(
                    PlaneKey::blurred(BasePlane::BlueNormalized, self.sigma),
                    context.region,
                ) as f64;
                return saturation_hsv_from_norm(red, green, blue);
            }
        }
    }

    /// Linear HSL Saturation at center from blurred planes (0..1).
    pub struct GaussianSaturationHsl {
        pub sigma: f32,
    }
    impl Heuristic for GaussianSaturationHsl {
        fn name(&self) -> &'static str {
            "gaussian_saturation_hsl"
        }
        fn requires(&self) -> &'static [PlaneKey] {
            #[cfg(feature = "accurate")]
            {
                REQUIRED_RED_GREEN_BLUE_PLANES_LINEAR
            }
            #[cfg(not(feature = "accurate"))]
            {
                REQUIRED_RED_GREEN_BLUE_PLANES_NORMALIZED
            }
        }
        fn evaluate(&self, context: &mut HeuristicContext) -> f64 {
            #[cfg(feature = "accurate")]
            {
                use crate::core_modules::utils::color_ops::saturation_hsl_from_linear;
                let (red, green, blue) = context.sample_blurred_linear_rgb_center(self.sigma);
                return saturation_hsl_from_linear(red as f64, green as f64, blue as f64);
            }
            #[cfg(not(feature = "accurate"))]
            {
                use crate::core_modules::utils::color_ops::saturation_hsl_from_norm;
                let red = context.prepared_frame.sample_plane_center(
                    PlaneKey::blurred(BasePlane::RedNormalized, self.sigma),
                    context.region,
                ) as f64;
                let green = context.prepared_frame.sample_plane_center(
                    PlaneKey::blurred(BasePlane::GreenNormalized, self.sigma),
                    context.region,
                ) as f64;
                let blue = context.prepared_frame.sample_plane_center(
                    PlaneKey::blurred(BasePlane::BlueNormalized, self.sigma),
                    context.region,
                ) as f64;
                return saturation_hsl_from_norm(red, green, blue);
            }
        }
    }

    /// Linear Colorfulness (≈ Chroma) at center from blurred planes (0..1).
    pub struct GaussianColorfulness {
        pub sigma: f32,
    }
    impl Heuristic for GaussianColorfulness {
        fn name(&self) -> &'static str {
            "gaussian_colorfulness"
        }
        fn requires(&self) -> &'static [PlaneKey] {
            #[cfg(feature = "accurate")]
            {
                REQUIRED_RED_GREEN_BLUE_PLANES_LINEAR
            }
            #[cfg(not(feature = "accurate"))]
            {
                REQUIRED_RED_GREEN_BLUE_PLANES_NORMALIZED
            }
        }
        fn evaluate(&self, context: &mut HeuristicContext) -> f64 {
            #[cfg(feature = "accurate")]
            {
                use crate::core_modules::utils::color_ops::{
                    lab_chroma_normalized, rgb_linear_to_lab,
                };
                let (red, green, blue) = context.sample_blurred_linear_rgb_center(self.sigma);
                let (_l, a, b) = rgb_linear_to_lab(red as f64, green as f64, blue as f64);
                return lab_chroma_normalized(a, b);
            }
            #[cfg(not(feature = "accurate"))]
            {
                use crate::core_modules::utils::color_ops::colorfulness_from_norm;
                let red = context.prepared_frame.sample_plane_center(
                    PlaneKey::blurred(BasePlane::RedNormalized, self.sigma),
                    context.region,
                ) as f64;
                let green = context.prepared_frame.sample_plane_center(
                    PlaneKey::blurred(BasePlane::GreenNormalized, self.sigma),
                    context.region,
                ) as f64;
                let blue = context.prepared_frame.sample_plane_center(
                    PlaneKey::blurred(BasePlane::BlueNormalized, self.sigma),
                    context.region,
                ) as f64;
                return colorfulness_from_norm(red, green, blue);
            }
        }
    }

    /// Linear Achromaticity (1 - chroma/value) at center from blurred planes (0..1).
    pub struct GaussianAchromaticity {
        pub sigma: f32,
    }
    impl Heuristic for GaussianAchromaticity {
        fn name(&self) -> &'static str {
            "gaussian_achromaticity"
        }
        fn requires(&self) -> &'static [PlaneKey] {
            #[cfg(feature = "accurate")]
            {
                REQUIRED_RED_GREEN_BLUE_PLANES_LINEAR
            }
            #[cfg(not(feature = "accurate"))]
            {
                REQUIRED_RED_GREEN_BLUE_PLANES_NORMALIZED
            }
        }
        fn evaluate(&self, context: &mut HeuristicContext) -> f64 {
            #[cfg(feature = "accurate")]
            {
                use crate::core_modules::utils::color_ops::achromaticity_from_linear;
                let (red, green, blue) = context.sample_blurred_linear_rgb_center(self.sigma);
                return achromaticity_from_linear(red as f64, green as f64, blue as f64);
            }
            #[cfg(not(feature = "accurate"))]
            {
                use crate::core_modules::utils::color_ops::achromaticity_from_norm;
                let red = context.prepared_frame.sample_plane_center(
                    PlaneKey::blurred(BasePlane::RedNormalized, self.sigma),
                    context.region,
                ) as f64;
                let green = context.prepared_frame.sample_plane_center(
                    PlaneKey::blurred(BasePlane::GreenNormalized, self.sigma),
                    context.region,
                ) as f64;
                let blue = context.prepared_frame.sample_plane_center(
                    PlaneKey::blurred(BasePlane::BlueNormalized, self.sigma),
                    context.region,
                ) as f64;
                return achromaticity_from_norm(red, green, blue);
            }
        }
    }

    /// Linear channel std dev across R,G,B at center from blurred planes.
    pub struct GaussianChannelStddev {
        pub sigma: f32,
    }
    impl Heuristic for GaussianChannelStddev {
        fn name(&self) -> &'static str {
            "gaussian_channel_stddev"
        }
        fn requires(&self) -> &'static [PlaneKey] {
            #[cfg(feature = "accurate")]
            {
                REQUIRED_RED_GREEN_BLUE_PLANES_LINEAR
            }
            #[cfg(not(feature = "accurate"))]
            {
                REQUIRED_RED_GREEN_BLUE_PLANES_NORMALIZED
            }
        }
        fn evaluate(&self, context: &mut HeuristicContext) -> f64 {
            #[cfg(feature = "accurate")]
            {
                use crate::core_modules::utils::color_ops::channel_stddev_from_linear;
                let (red, green, blue) = context.sample_blurred_linear_rgb_center(self.sigma);
                return channel_stddev_from_linear(red as f64, green as f64, blue as f64);
            }
            #[cfg(not(feature = "accurate"))]
            {
                use crate::core_modules::utils::color_ops::channel_stddev_from_norm;
                let red = context.prepared_frame.sample_plane_center(
                    PlaneKey::blurred(BasePlane::RedNormalized, self.sigma),
                    context.region,
                ) as f64;
                let green = context.prepared_frame.sample_plane_center(
                    PlaneKey::blurred(BasePlane::GreenNormalized, self.sigma),
                    context.region,
                ) as f64;
                let blue = context.prepared_frame.sample_plane_center(
                    PlaneKey::blurred(BasePlane::BlueNormalized, self.sigma),
                    context.region,
                ) as f64;
                return channel_stddev_from_norm(red, green, blue);
            }
        }
    }

    /// Chromaticity x at center from blurred linear planes.
    pub struct GaussianChromaticityX {
        pub sigma: f32,
    }
    impl Heuristic for GaussianChromaticityX {
        fn name(&self) -> &'static str {
            "gaussian_chromaticity_x"
        }
        fn requires(&self) -> &'static [PlaneKey] {
            #[cfg(feature = "accurate")]
            {
                REQUIRED_RED_GREEN_BLUE_PLANES_LINEAR
            }
            #[cfg(not(feature = "accurate"))]
            {
                REQUIRED_RED_GREEN_BLUE_PLANES_NORMALIZED
            }
        }
        fn evaluate(&self, context: &mut HeuristicContext) -> f64 {
            #[cfg(feature = "accurate")]
            {
                use crate::core_modules::utils::color_ops::chromaticity_xy_from_linear;
                let (red, green, blue) = context.sample_blurred_linear_rgb_center(self.sigma);
                let (x, _y) = chromaticity_xy_from_linear(red as f64, green as f64, blue as f64);
                return x;
            }
            #[cfg(not(feature = "accurate"))]
            {
                use crate::core_modules::utils::color_ops::chromaticity_xy_from_norm;
                let red = context.prepared_frame.sample_plane_center(
                    PlaneKey::blurred(BasePlane::RedNormalized, self.sigma),
                    context.region,
                ) as f64;
                let green = context.prepared_frame.sample_plane_center(
                    PlaneKey::blurred(BasePlane::GreenNormalized, self.sigma),
                    context.region,
                ) as f64;
                let blue = context.prepared_frame.sample_plane_center(
                    PlaneKey::blurred(BasePlane::BlueNormalized, self.sigma),
                    context.region,
                ) as f64;
                let (x, _y) = chromaticity_xy_from_norm(red, green, blue);
                return x;
            }
        }
    }

    /// Chromaticity y at center from blurred linear planes.
    pub struct GaussianChromaticityY {
        pub sigma: f32,
    }
    impl Heuristic for GaussianChromaticityY {
        fn name(&self) -> &'static str {
            "gaussian_chromaticity_y"
        }
        fn requires(&self) -> &'static [PlaneKey] {
            #[cfg(feature = "accurate")]
            {
                REQUIRED_RED_GREEN_BLUE_PLANES_LINEAR
            }
            #[cfg(not(feature = "accurate"))]
            {
                REQUIRED_RED_GREEN_BLUE_PLANES_NORMALIZED
            }
        }
        fn evaluate(&self, context: &mut HeuristicContext) -> f64 {
            #[cfg(feature = "accurate")]
            {
                use crate::core_modules::utils::color_ops::chromaticity_xy_from_linear;
                let (red, green, blue) = context.sample_blurred_linear_rgb_center(self.sigma);
                let (_x, y) = chromaticity_xy_from_linear(red as f64, green as f64, blue as f64);
                return y;
            }
            #[cfg(not(feature = "accurate"))]
            {
                use crate::core_modules::utils::color_ops::chromaticity_xy_from_norm;
                let red = context.prepared_frame.sample_plane_center(
                    PlaneKey::blurred(BasePlane::RedNormalized, self.sigma),
                    context.region,
                ) as f64;
                let green = context.prepared_frame.sample_plane_center(
                    PlaneKey::blurred(BasePlane::GreenNormalized, self.sigma),
                    context.region,
                ) as f64;
                let blue = context.prepared_frame.sample_plane_center(
                    PlaneKey::blurred(BasePlane::BlueNormalized, self.sigma),
                    context.region,
                ) as f64;
                let (_x, y) = chromaticity_xy_from_norm(red, green, blue);
                return y;
            }
        }
    }

    // ----------------------------- Depth heuristics -----------------------------

    /// Alias: DepthCv is the same as LocalBrightnessContrastCv.
    pub type DepthCv = LocalBrightnessContrast;

    /// Combined depth score: w_l*C_V + w_s*(1-S) + w_h*HueBias at region center.
    pub struct DepthCombined {
        pub sigma: f32,   // for V̄ blur
        pub epsilon: f32, // for C_V normalization
        pub w_l: f32,
        pub w_s: f32,
        pub w_h: f32,
    }

    impl Heuristic for DepthCombined {
        fn name(&self) -> &'static str {
            "depth_combined"
        }
        fn requires(&self) -> &'static [PlaneKey] {
            // Value, Saturation, and HueBias planes (Value will be blurred in eval)
            const REQS: &[PlaneKey] = &[
                PlaneKey {
                    base: BasePlane::Value,
                    sigma: None,
                },
                PlaneKey {
                    base: BasePlane::Saturation,
                    sigma: None,
                },
                PlaneKey {
                    base: BasePlane::HueBias,
                    sigma: None,
                },
            ];
            REQS
        }
        fn evaluate(&self, context: &mut HeuristicContext) -> f64 {
            // C_V
            let value_at_center = context
                .prepared_frame
                .sample_plane_center(PlaneKey::base(BasePlane::Value), context.region);
            let neighborhood_mean = context.prepared_frame.sample_plane_center(
                PlaneKey::blurred(BasePlane::Value, self.sigma),
                context.region,
            );
            let denominator = neighborhood_mean.max(self.epsilon);
            let local_brightness_contrast = ((neighborhood_mean - value_at_center) / denominator)
                .max(0.0)
                .min(1.0);

            // 1 - S
            let saturation = context
                .prepared_frame
                .sample_plane_center(PlaneKey::base(BasePlane::Saturation), context.region);
            let desaturation_far_value = (1.0 - saturation).max(0.0).min(1.0);

            // Hue bias
            let hue_bias_value = context
                .prepared_frame
                .sample_plane_center(PlaneKey::base(BasePlane::HueBias), context.region);

            let depth_score = self.w_l * local_brightness_contrast
                + self.w_s * desaturation_far_value
                + self.w_h * hue_bias_value;
            depth_score.max(0.0).min(1.0) as f64
        }
    }

    // (Removed Pixel-based Gaussian heuristics to eliminate round-trips and duplication.)
}
pub mod pairwise;
