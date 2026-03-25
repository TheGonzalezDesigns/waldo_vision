// THEORY:
// The `SmartPixel` module provides the analytical capabilities for our vision system.
// It follows the "separation of concerns" principle by acting as a "smart" wrapper
// around a "dumb" `Pixel` data object. Its entire purpose is to compare two pixels
// and quantify the difference between them in various meaningful ways.
//
// Key architectural principles:
// 1.  **Comparative Analysis**: All core methods (`delta_luminance`, `hue_difference`)
//     take another pixel as input. A `SmartPixel` is meaningless on its own; its value
//     is in calculating relationships.
// 2.  **Multiple "Lenses"**: It provides different ways to measure "difference," as each
//     is useful for a different task.
//     - `delta_luminance`: Best for robust motion detection (heat map).
//     - `hue_difference`: Best for creating color-based signatures (object ID).
//     - `delta_color`: A fast, low-cost alternative for rough difference.
// 3.  **Optimization**: It pre-calculates and caches values like `sum` and `luminance`
//     in its constructor. This is a performance optimization for one-to-many comparisons,
//     where a single pixel from a new frame is compared against a history of pixels.

pub mod smart_pixel {
    use crate::core_modules::pixel::pixel::*;

    pub type ColorDelta = u16;
    pub type LuminanceDelta = f64;
    pub type HueDifference = f64;

    /// An analytical tool that wraps a `Pixel` to provide optimized comparison methods.
    pub struct SmartPixel {
        /// The raw `Pixel` data this `SmartPixel` is analyzing.
        pub pixel: Pixel,
        /// The pre-calculated sum of the RGB channels, cached for performance.
        sum: Sum,
        /// The pre-calculated luminance of the pixel, cached for performance.
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

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn test_smart_pixel_new() {
            let pixel = Pixel::new(100, 100, 100, 255);
            let smart = SmartPixel::new(pixel.clone());
            assert_eq!(smart.pixel, pixel);
            assert_eq!(smart.sum, 300.0);
            assert!((smart.luminance - 100.0).abs() < 1e-6);
        }

        #[test]
        fn test_delta_color() {
            let p1 = SmartPixel::new(Pixel::new(10, 10, 10, 255));
            let p2 = SmartPixel::new(Pixel::new(20, 20, 20, 255));
            assert_eq!(p1.delta_color(&p2), 30);
            assert_eq!(p2.delta_color(&p1), 30);
        }

        #[test]
        fn test_delta_luminance() {
            let p1 = SmartPixel::new(Pixel::new(0, 0, 0, 255));
            let p2 = SmartPixel::new(Pixel::new(100, 100, 100, 255));
            assert!((p1.delta_luminance(&p2) - 100.0).abs() < 1e-6);
        }

        #[test]
        fn test_hue_difference() {
            let p1 = SmartPixel::new(Pixel::new(255, 0, 0, 255));
            let p2 = SmartPixel::new(Pixel::new(0, 255, 0, 255));
            // (1, 0, 0) vs (0, 1, 0) => |1-0| + |0-1| + |0-0| = 2.0
            assert!((p1.hue_difference(&p2) - 2.0).abs() < 1e-6);

            let p3 = SmartPixel::new(Pixel::new(100, 100, 100, 255));
            let p4 = SmartPixel::new(Pixel::new(50, 50, 50, 255));
            // Both are (0.33, 0.33, 0.33) approximately
            assert!(p3.hue_difference(&p4) < 1e-6);
        }
    }
}
