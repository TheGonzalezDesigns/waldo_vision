pub mod spatial_pooler {
    use std::collections::HashMap;

    use crate::core_modules::data::pixel::pixel::Pixel;
    use crate::core_modules::pooling::gaussian_engine::gaussian_engine::yvv_gaussian_blur_plane;
    use crate::core_modules::utils::color_ops::{
        ColorMode, DEFAULT_COLOR_MODE, hue_bias_from_hue_deg, hue_from_linear_rgb_deg,
        hue_from_normalized_rgb_deg, saturation_hsv_from_linear, saturation_hsv_from_norm,
    };
    use crate::core_modules::utils::region::Region;

    /// Quantized sigma key to allow use in Hash/Eq (millis precision).
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
    pub struct SigmaKey(pub u32);

    impl From<f32> for SigmaKey {
        fn from(s: f32) -> Self {
            // Clamp to non-negative; quantize to 1e-3 resolution
            let q = (s.max(0.0) * 1000.0).round() as u32;
            SigmaKey(q)
        }
    }

    /// Base planes that can be prepared from a frame.
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
    pub enum BasePlane {
        /// Linear RGB channels in [0,1].
        RedLinear,
        GreenLinear,
        BlueLinear,
        /// Normalized sRGB channels in [0,1] (gamma-encoded).
        RedNormalized,
        GreenNormalized,
        BlueNormalized,
        /// HSV-derived Value and Saturation in [0,1].
        Value,
        Saturation,
        /// Hue bias in [0,1] favoring cool hues (≈240°).
        HueBias,
    }

    /// A specific plane request, optionally blurred with sigma.
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
    pub struct PlaneKey {
        pub base: BasePlane,
        pub sigma: Option<SigmaKey>,
    }

    impl PlaneKey {
        pub fn base(base: BasePlane) -> Self {
            Self { base, sigma: None }
        }
        pub fn blurred(base: BasePlane, sigma: f32) -> Self {
            Self {
                base,
                sigma: Some(SigmaKey::from(sigma)),
            }
        }
    }

    /// Prepared per-frame resources and lazily computed planes for heuristics/pooling.
    pub struct PreparedFrame {
        pub w: usize,
        pub h: usize,
        pub pixels: Vec<Pixel>,
        planes: HashMap<PlaneKey, Vec<f32>>, // length = w*h
    }

    impl PreparedFrame {
        pub fn new(pixels: Vec<Pixel>, w: usize, h: usize) -> Self {
            Self {
                w,
                h,
                pixels,
                planes: HashMap::new(),
            }
        }

        /// Ensure a plane is available in the cache and return it.
        pub fn ensure_plane(&mut self, key: PlaneKey) -> &Vec<f32> {
            if !self.planes.contains_key(&key) {
                let base_plane = self.compute_base_plane(key.base);
                let plane = if let Some(skey) = key.sigma {
                    let sigma = (skey.0 as f32) / 1000.0;
                    yvv_gaussian_blur_plane(&base_plane, self.w, self.h, sigma)
                } else {
                    base_plane
                };
                self.planes.insert(key, plane);
            }
            self.planes.get(&key).expect("plane exists")
        }

        fn compute_base_plane(&self, base: BasePlane) -> Vec<f32> {
            match base {
                BasePlane::RedLinear => self.pixels.iter().map(|p| p.red_linear as f32).collect(),
                BasePlane::GreenLinear => {
                    self.pixels.iter().map(|p| p.green_linear as f32).collect()
                }
                BasePlane::BlueLinear => self.pixels.iter().map(|p| p.blue_linear as f32).collect(),
                BasePlane::RedNormalized => self
                    .pixels
                    .iter()
                    .map(|p| p.red_normalized as f32)
                    .collect(),
                BasePlane::GreenNormalized => self
                    .pixels
                    .iter()
                    .map(|p| p.green_normalized as f32)
                    .collect(),
                BasePlane::BlueNormalized => self
                    .pixels
                    .iter()
                    .map(|p| p.blue_normalized as f32)
                    .collect(),
                BasePlane::Value => match DEFAULT_COLOR_MODE {
                    ColorMode::Optimal => self
                        .pixels
                        .iter()
                        .map(|p| {
                            p.red_normalized
                                .max(p.green_normalized.max(p.blue_normalized))
                                as f32
                        })
                        .collect(),
                    ColorMode::Accurate => self
                        .pixels
                        .iter()
                        .map(|p| p.red_linear.max(p.green_linear.max(p.blue_linear)) as f32)
                        .collect(),
                },
                BasePlane::Saturation => match DEFAULT_COLOR_MODE {
                    ColorMode::Optimal => self
                        .pixels
                        .iter()
                        .map(|p| {
                            saturation_hsv_from_norm(
                                p.red_normalized as f64,
                                p.green_normalized as f64,
                                p.blue_normalized as f64,
                            ) as f32
                        })
                        .collect(),
                    ColorMode::Accurate => self
                        .pixels
                        .iter()
                        .map(|p| {
                            saturation_hsv_from_linear(
                                p.red_linear as f64,
                                p.green_linear as f64,
                                p.blue_linear as f64,
                            ) as f32
                        })
                        .collect(),
                },
                BasePlane::HueBias => {
                    const TARGET_DEG: f64 = 240.0;
                    match DEFAULT_COLOR_MODE {
                        ColorMode::Optimal => self
                            .pixels
                            .iter()
                            .map(|p| {
                                let h = hue_from_normalized_rgb_deg(
                                    p.red_normalized as f64,
                                    p.green_normalized as f64,
                                    p.blue_normalized as f64,
                                );
                                hue_bias_from_hue_deg(h, TARGET_DEG) as f32
                            })
                            .collect(),
                        ColorMode::Accurate => self
                            .pixels
                            .iter()
                            .map(|p| {
                                let h = hue_from_linear_rgb_deg(
                                    p.red_linear as f32,
                                    p.green_linear as f32,
                                    p.blue_linear as f32,
                                );
                                hue_bias_from_hue_deg(h, TARGET_DEG) as f32
                            })
                            .collect(),
                    }
                }
            }
        }

        /// Sample the center pixel of a region from a specific plane.
        pub fn sample_plane_center(&mut self, key: PlaneKey, region: Region) -> f32 {
            let frame_width = self.w;
            let frame_height = self.h;
            let (center_x, center_y) =
                region.center_coordinates(frame_width as u32, frame_height as u32);
            let plane = self.ensure_plane(key);
            plane[center_y as usize * frame_width + center_x as usize]
        }

        // Removed gaussian_center_pixel helper to avoid sRGB round-trips in core heuristics.
    }

    /// Interface to produce per-region observations using different spatial pooling strategies.
    pub trait SpatialPooler {
        /// Build any per-frame resources once.
        fn prepare(&self, pixels: Vec<Pixel>, w: usize, h: usize) -> PreparedFrame;
        /// Produce a Gaussian- or box-averaged Pixel at the region center.
        fn sample_region_center_pixel(&self, pf: &mut PreparedFrame, region: Region) -> Pixel;
    }

    /// Uniform box average using raw pixels — mirrors current chunk averaging semantics.
    pub struct BoxPooler;

    impl SpatialPooler for BoxPooler {
        fn prepare(&self, pixels: Vec<Pixel>, w: usize, h: usize) -> PreparedFrame {
            PreparedFrame::new(pixels, w, h)
        }

        fn sample_region_center_pixel(
            &self,
            prepared_frame: &mut PreparedFrame,
            region: Region,
        ) -> Pixel {
            let mut sum_red: u32 = 0;
            let mut sum_green: u32 = 0;
            let mut sum_blue: u32 = 0;
            let mut sum_alpha: u32 = 0;
            let mut pixel_count: u32 = 0;

            let start_x = region.x_coordinate as usize;
            let start_y = region.y_coordinate as usize;
            let end_x = (region.x_coordinate + region.width).min(prepared_frame.w as u32) as usize;
            let end_y = (region.y_coordinate + region.height).min(prepared_frame.h as u32) as usize;

            for y in start_y..end_y {
                let row_base = y * prepared_frame.w;
                for x in start_x..end_x {
                    let pixel = &prepared_frame.pixels[row_base + x];
                    sum_red += pixel.red as u32;
                    sum_green += pixel.green as u32;
                    sum_blue += pixel.blue as u32;
                    sum_alpha += pixel.alpha as u32;
                    pixel_count += 1;
                }
            }

            if pixel_count == 0 {
                return Pixel::default();
            }
            let red_average = (sum_red / pixel_count) as u8;
            let green_average = (sum_green / pixel_count) as u8;
            let blue_average = (sum_blue / pixel_count) as u8;
            let alpha_average = (sum_alpha / pixel_count) as u8;
            Pixel::new(red_average, green_average, blue_average, alpha_average)
        }
    }

    /// Gaussian-weighted center sampling using blurred linear RGB planes.
    pub struct GaussianPooler {
        pub sigma: f32,
    }

    impl GaussianPooler {
        fn linear_to_srgb_u8(x: f32) -> u8 {
            // Inverse of sRGB EOTF; clamp to [0,1].
            let x = x.max(0.0).min(1.0);
            let srgb = if x <= 0.003_130_8 {
                12.92 * x
            } else {
                1.055 * x.powf(1.0 / 2.4) - 0.055
            };
            (srgb.max(0.0).min(1.0) * 255.0).round() as u8
        }
    }

    impl SpatialPooler for GaussianPooler {
        fn prepare(&self, pixels: Vec<Pixel>, w: usize, h: usize) -> PreparedFrame {
            PreparedFrame::new(pixels, w, h)
        }

        fn sample_region_center_pixel(
            &self,
            prepared_frame: &mut PreparedFrame,
            region: Region,
        ) -> Pixel {
            let red = prepared_frame
                .sample_plane_center(PlaneKey::blurred(BasePlane::RedLinear, self.sigma), region);
            let green = prepared_frame.sample_plane_center(
                PlaneKey::blurred(BasePlane::GreenLinear, self.sigma),
                region,
            );
            let blue = prepared_frame
                .sample_plane_center(PlaneKey::blurred(BasePlane::BlueLinear, self.sigma), region);
            let (center_x, center_y) =
                region.center_coordinates(prepared_frame.w as u32, prepared_frame.h as u32);
            let alpha = prepared_frame.pixels
                [center_y as usize * prepared_frame.w + center_x as usize]
                .alpha;
            let red_u8 = Self::linear_to_srgb_u8(red);
            let green_u8 = Self::linear_to_srgb_u8(green);
            let blue_u8 = Self::linear_to_srgb_u8(blue);
            Pixel::new(red_u8, green_u8, blue_u8, alpha)
        }
    }
}
