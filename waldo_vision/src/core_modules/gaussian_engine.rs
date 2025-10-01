pub mod gaussian_engine {
    use crate::core_modules::D1::pixel::pixel::Pixel;

    pub type Sigma = f32;
    const CUTOFF: Sigma = 3.0f32;

    pub fn gaussian_kernel_1d(sigma: Sigma) -> Vec<f32> {
        assert!(sigma > 0.0, "Sigma must be greater than 0");

        let radius = (CUTOFF * sigma).ceil() as i32;
        let size = (2 * radius + 1) as usize;
        let mut kernel = Vec::with_capacity(size);
        for i in -radius..=radius {
            let val = (-((i * i) as f32) / (2.0 * sigma * sigma)).exp();
            kernel.push(val);
        }

        //normalize kernal so it sums up to 1;
        let sum: f32 = kernel.iter().sum();
        for k in &mut kernel {
            *k /= sum.max(1e-12);
        }

        kernel
    }

    pub fn blur_horizontal(src: &[f32], w: usize, h: usize, k: &[f32]) -> Vec<f32> {
        let mut out = vec![0.0f32; w * h];
        let r = (k.len() / 2) as isize;

        for y in 0..h {
            let row = &src[y * w..(y + 1) * w];
            let row_out = &mut out[y * w..(y + 1) * w];

            for x in 0..w {
                let mut acc = 0.0;
                for (i, &kw) in k.iter().enumerate() {
                    let dx = x as isize + (i as isize - r);
                    if dx >= 0 && dx < w as isize {
                        acc += row[dx as usize] * kw;
                    }
                }
                row_out[x] = acc;
            }
        }
        out
    }

    pub fn blur_vertical(src: &[f32], w: usize, h: usize, k: &[f32]) -> Vec<f32> {
        let mut out = vec![0.0f32; w * h];
        let r = (k.len()/2) as isize;

        for y in 0..h {
            for x in 0..w {
                let mut acc = 0.0;
                for (i, &kw) in k.iter().enumerate() {
                    let dy = y as isize + (i as isize - r);
                    if dy >= 0 && dy < h as isize {
                        acc += src[dy as usize * w + x] * kw;
                    }
                }
                out[y*w + x] = acc;
            }
        }

        out
    }

    pub fn gaussian_blur_plane(src: &[f32], w: usize, h: usize, sigma: f32) -> Vec<f32> {
        let k = gaussian_kernel_1d(sigma);
        let tmp = blur_horizontal(src, w, h, &k);
        blur_vertical(&tmp, w, h, &k)
    }

    /// Convenience: run separable Gaussian blur over a brightness plane derived from Pixels.
    /// Uses existing per-pixel metrics (no re-linearization):
    /// - Default: `value_hsv()` based on normalized sRGB
    /// - With feature `accurate`: `value_hsv()` uses linear RGB
    pub fn gaussian_blur_pixels_luma(pixels: &[Pixel], w: usize, h: usize, sigma: f32) -> Vec<f32> {
        let plane: Vec<f32> = pixels.iter().map(|p| p.value_hsv() as f32).collect();
        gaussian_blur_plane(&plane, w, h, sigma)
    }

    /// Separable Gaussian blur over RGB channels, returning packed RGBA bytes (u8).
    /// - Color channels are blurred in normalized space and then scaled back to 0..255.
    /// - Alpha channel is preserved from input pixels as-is (no blur).
    pub fn gaussian_blur_pixels_rgba_bytes(
        pixels: &[Pixel],
        w: usize,
        h: usize,
        sigma: f32,
    ) -> Vec<u8> {
        // Extract precomputed channels (normalized by default; linear with `accurate`).
        let n = pixels.len();
        let mut r = Vec::with_capacity(n);
        let mut g = Vec::with_capacity(n);
        let mut b = Vec::with_capacity(n);
        #[cfg(feature = "accurate")]
        for p in pixels {
            r.push(p.red_linear as f32);
            g.push(p.green_linear as f32);
            b.push(p.blue_linear as f32);
        }
        #[cfg(not(feature = "accurate"))]
        for p in pixels {
            r.push(p.red_normalized as f32);
            g.push(p.green_normalized as f32);
            b.push(p.blue_normalized as f32);
        }
        let r_b = gaussian_blur_plane(&r, w, h, sigma);
        let g_b = gaussian_blur_plane(&g, w, h, sigma);
        let b_b = gaussian_blur_plane(&b, w, h, sigma);

        let mut out = Vec::with_capacity(pixels.len() * 4);
        for i in 0..pixels.len() {
            let ru = (r_b[i].clamp(0.0, 1.0) * 255.0).round().clamp(0.0, 255.0) as u8;
            let gu = (g_b[i].clamp(0.0, 1.0) * 255.0).round().clamp(0.0, 255.0) as u8;
            let bu = (b_b[i].clamp(0.0, 1.0) * 255.0).round().clamp(0.0, 255.0) as u8;
            let au = pixels[i].alpha; // preserve alpha
            out.push(ru);
            out.push(gu);
            out.push(bu);
            out.push(au);
        }
        out
    }
}
