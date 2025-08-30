pub mod smart_pixel {
    use crate::core_modules::pixel::pixel::*;

    pub type ColorDelta = u16;
    pub type LuminanceDelta = f64;
    pub type HueDifference = f64;

    pub struct SmartPixel {
        pub pixel: Pixel,
        sum: Sum,
        luminance: Luminance,
    }

    impl SmartPixel {
        pub fn new(pixel: Pixel) -> Self {
            Self {
                sum: pixel.sum(),
                luminance: pixel.luminance(),
                pixel,
            }
        }

        pub fn delta_color(&self, other: &SmartPixel) -> ColorDelta {
            (self.sum - other.sum).abs() as ColorDelta
        }

        pub fn delta_luminance(&self, other: &SmartPixel) -> LuminanceDelta {
            (self.luminance - other.luminance).abs()
        }

        pub fn hue_difference(&self, other: &SmartPixel) -> HueDifference {
            let (r1, g1, b1) = self.pixel.color_ratios();
            let (r2, g2, b2) = other.pixel.color_ratios();

            let diff = (r1 - r2).abs() + (g1 - g2).abs() + (b1 - b2).abs();

            diff as f64
        }
    }
}
