pub mod gaussian_engine {
    use crate::core_modules::D1::pixel::pixel::Pixel;
    use rayon::prelude::*;

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

        // Normalize kernel so it sums to 1.
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
                    let dx = (x as isize + (i as isize - r)).clamp(0, (w as isize) - 1) as usize;
                    acc += row[dx] * kw;
                }
                row_out[x] = acc;
            }
        }
        out
    }

    pub fn blur_vertical(src: &[f32], w: usize, h: usize, k: &[f32]) -> Vec<f32> {
        let mut out = vec![0.0f32; w * h];
        let r = (k.len() / 2) as isize;

        for y in 0..h {
            for x in 0..w {
                let mut acc = 0.0;
                for (i, &kw) in k.iter().enumerate() {
                    let dy = (y as isize + (i as isize - r)).clamp(0, (h as isize) - 1) as usize;
                    acc += src[dy * w + x] * kw;
                }
                out[y * w + x] = acc;
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
        // Use linear RGB for blur to avoid gamma-space artifacts.
        for p in pixels {
            r.push(p.red_linear as f32);
            g.push(p.green_linear as f32);
            b.push(p.blue_linear as f32);
        }
        // Use Young–Van Vliet recursive Gaussian (fused, row/col parallel when enabled)
        let planes: [&[f32]; 3] = [&r[..], &g[..], &b[..]];
        let blurred = yvv_gaussian_blur_planes(&planes, w, h, sigma);
        let (r_b, g_b, b_b) = (blurred[0].clone(), blurred[1].clone(), blurred[2].clone());

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

    // ========================= Depth Cues (HSV-based) =========================

    /// Extract Value (HSV) plane from Pixels in [0,1].
    /// Uses `Pixel::value_hsv()` which is normalized sRGB by default,
    /// or linear RGB when the `accurate` feature is enabled.
    pub fn value_plane_from_pixels(pixels: &[Pixel]) -> Vec<f32> {
        pixels.iter().map(|p| p.value_hsv() as f32).collect()
    }

    /// Extract Saturation (HSV) plane from Pixels in [0,1].
    pub fn saturation_plane_from_pixels(pixels: &[Pixel]) -> Vec<f32> {
        pixels.iter().map(|p| p.saturation_hsv() as f32).collect()
    }

    /// Compute Hue Bias plane in [0,1] favoring cool hues (≈240°) as "far".
    /// Mapping: B_H = 0.5 * (1 + cos(h − 240°)), where h is in radians.
    pub fn hue_bias_plane_from_pixels(pixels: &[Pixel]) -> Vec<f32> {
        const H_COOL_DEG: f64 = 240.0;
        let target = H_COOL_DEG.to_radians();
        pixels
            .iter()
            .map(|p| {
                let h_rad = p.hue().to_radians();
                let val = 0.5 * (1.0 + (h_rad - target).cos());
                // Clamp and guard against non-finite values for robustness
                let val = if val.is_finite() {
                    val.clamp(0.0, 1.0)
                } else {
                    0.5
                };
                val as f32
            })
            .collect()
    }

    /// Local Brightness Contrast C_V in [0,1].
    /// C_V high when pixel is darker than local mean (recedes),
    /// low when pixel is brighter than local mean (advances).
    pub fn local_brightness_contrast(v: &[f32], v_bar: &[f32], epsilon: f32) -> Vec<f32> {
        assert_eq!(v.len(), v_bar.len());
        let mut out = Vec::with_capacity(v.len());
        for i in 0..v.len() {
            let denom = v_bar[i].max(epsilon);
            let raw = (v_bar[i] - v[i]) / denom;
            out.push(raw.max(0.0).min(1.0));
        }
        out
    }

    /// Combine depth cues into a single depth score in [0,1].
    pub fn combine_depth(
        c_v: &[f32], // Local Brightness Contrast [0,1]
        s_f: &[f32], // Desaturation far-ness [0,1] (1 - S)
        b_h: &[f32], // Hue bias [0,1]
        w_l: f32,
        w_s: f32,
        w_h: f32,
    ) -> Vec<f32> {
        let n = c_v.len();
        assert_eq!(n, s_f.len());
        assert_eq!(n, b_h.len());
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            let d = w_l * c_v[i] + w_s * s_f[i] + w_h * b_h[i];
            out.push(d.max(0.0).min(1.0));
        }
        out
    }

    /// High-level: compute a per-pixel depth score from Pixels using HSV-derived cues.
    /// - `sigma`: Gaussian sigma for local mean on Value plane
    /// - `epsilon`: small positive to stabilize normalization (e.g., 1e-3)
    /// - `weights`: (w_l, w_s, w_h) for (C_V, 1-S, Hue Bias)
    pub fn depth_from_pixels_hsv(
        pixels: &[Pixel],
        w: usize,
        h: usize,
        sigma: f32,
        epsilon: f32,
        weights: (f32, f32, f32),
    ) -> Vec<f32> {
        let v = value_plane_from_pixels(pixels);
        let s = saturation_plane_from_pixels(pixels);
        let b_h = hue_bias_plane_from_pixels(pixels);

        let v_bar = yvv_gaussian_blur_plane(&v, w, h, sigma);
        let c_v = local_brightness_contrast(&v, &v_bar, epsilon);
        let s_f: Vec<f32> = s.into_iter().map(|sv| 1.0 - sv).collect();
        let (w_l, w_s, w_h) = weights;
        combine_depth(&c_v, &s_f, &b_h, w_l, w_s, w_h)
    }

    // Parallel FIR helpers removed; runtime auto-parallelization is used in YvV path.

    // ========================= Fused Multi-Plane Blur =========================

    /// Fused separable Gaussian blur over N planes sharing the same sigma.
    /// - Reduces memory traffic by iterating once per pixel for all planes.
    /// - Each plane length must be `w*h`.
    pub fn gaussian_blur_planes(
        src_planes: &[&[f32]],
        w: usize,
        h: usize,
        sigma: f32,
    ) -> Vec<Vec<f32>> {
        let nplanes = src_planes.len();
        assert!(nplanes > 0, "no planes provided");
        for p in src_planes.iter() {
            assert_eq!(p.len(), w * h, "all planes must be w*h length");
        }

        let k = gaussian_kernel_1d(sigma);
        let r = (k.len() / 2) as isize;

        // Horizontal pass (fused across planes) with mirror addressing
        let mut tmp_planes: Vec<Vec<f32>> = (0..nplanes).map(|_| vec![0.0f32; w * h]).collect();
        for y in 0..h {
            // Accumulators per plane reused across x
            let mut acc: Vec<f32> = vec![0.0; nplanes];
            for x in 0..w {
                // Reset accumulators for this pixel
                for a in &mut acc {
                    *a = 0.0;
                }
                for (i, &kw) in k.iter().enumerate() {
                    let dx = (x as isize + (i as isize - r)).clamp(0, (w as isize) - 1) as usize;
                    let idx = y * w + dx;
                    for p in 0..nplanes {
                        acc[p] += src_planes[p][idx] * kw;
                    }
                }
                let out_idx = y * w + x;
                for p in 0..nplanes {
                    tmp_planes[p][out_idx] = acc[p];
                }
            }
        }

        // Vertical pass (fused across planes) with mirror addressing
        let mut out_planes: Vec<Vec<f32>> = (0..nplanes).map(|_| vec![0.0f32; w * h]).collect();
        for y in 0..h {
            let mut acc: Vec<f32> = vec![0.0; nplanes];
            for x in 0..w {
                for a in &mut acc {
                    *a = 0.0;
                }
                for (i, &kw) in k.iter().enumerate() {
                    let dy = (y as isize + (i as isize - r)).clamp(0, (h as isize) - 1) as usize;
                    let idx = dy * w + x;
                    for p in 0..nplanes {
                        acc[p] += tmp_planes[p][idx] * kw;
                    }
                }
                let out_idx = y * w + x;
                for p in 0..nplanes {
                    out_planes[p][out_idx] = acc[p];
                }
            }
        }

        out_planes
    }

    // ========================= YvV IIR Gaussian (fused) =========================
    #[inline]
    fn should_parallelize_rows(w: usize, h: usize, nplanes: usize) -> bool {
        let total = w.saturating_mul(h);
        rayon::current_num_threads() > 1 && total >= 262_144 && h >= 64 && nplanes >= 1
    }

    #[inline]
    fn should_parallelize_cols(w: usize, h: usize, nplanes: usize) -> bool {
        let total = w.saturating_mul(h);
        rayon::current_num_threads() > 1 && total >= 262_144 && w >= 64 && nplanes >= 1
    }

    #[derive(Clone, Copy, Debug)]
    struct YvvCoeffs {
        a0: f32,
        a1: f32,
        a2: f32,
        a3: f32,
        b1: f32,
        b2: f32,
        b3: f32,
    }

    fn normalize_dc(mut c: YvvCoeffs) -> YvvCoeffs {
        let bsum = c.b1 + c.b2 + c.b3;
        let asum = c.a0 + c.a1 + c.a2 + c.a3;
        let target = 1.0 - bsum;
        let g = if asum.abs() > 1e-12 {
            target / asum
        } else {
            1.0
        };
        c.a0 *= g;
        c.a1 *= g;
        c.a2 *= g;
        c.a3 *= g;
        c
    }

    fn yvv_coeffs(sigma: f32) -> YvvCoeffs {
        // Standard Young–Van Vliet 3rd-order coefficients
        let sigma = sigma.max(1e-6);
        let q = if sigma >= 2.5 {
            0.98711 * sigma - 0.96330
        } else {
            let s2 = sigma * sigma;
            3.97156 - 4.14554 * (1.0 - 0.26891 * s2).sqrt()
        } as f32;

        let b0 = 1.57825 + 2.44413 * q + 1.4281 * q * q + 0.422205 * q * q * q;
        let b1 = 2.44413 * q + 2.85619 * q * q + 1.26661 * q * q * q;
        let b2 = -(1.4281 * q * q + 1.26661 * q * q * q);
        let b3 = 0.422205 * q * q * q;
        let bsum = b1 + b2 + b3;
        let b0_inv = 1.0f32 / b0.max(1e-12);
        let b1n = b1 * b0_inv;
        let b2n = b2 * b0_inv;
        let b3n = b3 * b0_inv;
        let a0 = 1.0 - bsum * b0_inv; // initial DC-approximate a0
        let a1 = a0 * b1n;
        let a2 = a0 * b2n;
        let a3 = a0 * b3n;
        normalize_dc(YvvCoeffs {
            a0,
            a1,
            a2,
            a3,
            b1: b1n,
            b2: b2n,
            b3: b3n,
        })
    }

    pub fn yvv_gaussian_blur_plane(src: &[f32], w: usize, h: usize, sigma: f32) -> Vec<f32> {
        let planes = [&src[..]];
        let out = yvv_gaussian_blur_planes(&planes, w, h, sigma);
        out.into_iter().next().unwrap()
    }

    pub fn yvv_gaussian_blur_planes(
        src_planes: &[&[f32]],
        w: usize,
        h: usize,
        sigma: f32,
    ) -> Vec<Vec<f32>> {
        let nplanes = src_planes.len();
        assert!(nplanes > 0);
        for p in src_planes.iter() {
            assert_eq!(p.len(), w * h);
        }
        let c = yvv_coeffs(sigma);
        // Compute 1D DC gain of combined (forward+backward) pass and scale to unity.
        // For anti-causal including a0: T = 2 * (a0+a1+a2+a3) / (1 - (b1+b2+b3)).
        let bsum = c.b1 + c.b2 + c.b3;
        let asum = c.a0 + c.a1 + c.a2 + c.a3;
        let g1d = if (1.0 - bsum).abs() > 1e-12 {
            (1.0 - bsum) / (2.0 * asum.max(1e-12))
        } else {
            1.0
        };
        let y_ss = (asum / (1.0 - bsum));

        // Horizontal pass (fused) with runtime auto-parallelization.
        let mut tmp: Vec<Vec<f32>> = (0..nplanes).map(|_| vec![0.0f32; w * h]).collect();

        if should_parallelize_rows(w, h, nplanes) {
            // Compute each row in parallel into temporary per-row buffers, then scatter.
            let row_buffers: Vec<Vec<Vec<f32>>> = (0..h)
                .into_par_iter()
                .map(|y| {
                    let row_base = y * w;
                    // allocate per-plane row buffers
                    let mut rows: Vec<Vec<f32>> = (0..nplanes).map(|_| vec![0.0f32; w]).collect();
                    // Forward
                    let mut xm1: Vec<f32> = (0..nplanes).map(|p| src_planes[p][row_base]).collect();
                    let mut xm2 = xm1.clone();
                    let mut xm3 = xm1.clone();
                    // Precharge y-history to DC of constant extension at the border.
                    let y_dc = g1d * (asum / (1.0 - bsum));
                    let mut ym1: Vec<f32> = vec![y_ss; nplanes];
                    let mut ym2: Vec<f32> = vec![y_ss; nplanes];
                    let mut ym3: Vec<f32> = vec![y_ss; nplanes];
                    for x in 0..w {
                        let idx = row_base + x;
                        for p in 0..nplanes {
                            let x0 = src_planes[p][idx];
                            let yv = c.a0 * x0
                                + c.a1 * xm1[p]
                                + c.a2 * xm2[p]
                                + c.a3 * xm3[p]
                                + c.b1 * ym1[p]
                                + c.b2 * ym2[p]
                                + c.b3 * ym3[p];
                            rows[p][x] = g1d * yv;
                            xm3[p] = xm2[p];
                            xm2[p] = xm1[p];
                            xm1[p] = x0;
                            ym3[p] = ym2[p];
                            ym2[p] = ym1[p];
                            ym1[p] = yv;
                        }
                    }
                    // Backward (anti-causal)
                    let last_idx = row_base + (w - 1);
                    let mut xp1: Vec<f32> = (0..nplanes).map(|p| src_planes[p][last_idx]).collect();
                    let mut xp2 = xp1.clone();
                    let mut xp3 = xp1.clone();
                    let mut yp1: Vec<f32> = vec![y_ss; nplanes];
                    let mut yp2: Vec<f32> = vec![y_ss; nplanes];
                    let mut yp3: Vec<f32> = vec![y_ss; nplanes];
                    for x in (0..w).rev() {
                        let idx = row_base + x;
                        for p in 0..nplanes {
                            let x0 = src_planes[p][idx];
                            let yv = c.a0 * x0
                                + c.a1 * xp1[p]
                                + c.a2 * xp2[p]
                                + c.a3 * xp3[p]
                                + c.b1 * yp1[p]
                                + c.b2 * yp2[p]
                                + c.b3 * yp3[p];
                            rows[p][x] += g1d * yv;
                            xp3[p] = xp2[p];
                            xp2[p] = xp1[p];
                            xp1[p] = x0;
                            yp3[p] = yp2[p];
                            yp2[p] = yp1[p];
                            yp1[p] = yv;
                        }
                    }
                    rows
                })
                .collect();
            // Scatter into tmp buffers
            for y in 0..h {
                let row_base = y * w;
                for p in 0..nplanes {
                    tmp[p][row_base..row_base + w].copy_from_slice(&row_buffers[y][p]);
                }
            }
        } else {
            for y in 0..h {
                let row_base = y * w;
                // Forward
                let mut xm1: Vec<f32> = (0..nplanes).map(|p| src_planes[p][row_base]).collect();
                let mut xm2 = xm1.clone();
                let mut xm3 = xm1.clone();
                let mut ym1: Vec<f32> = vec![y_ss; nplanes];
                let mut ym2: Vec<f32> = vec![y_ss; nplanes];
                let mut ym3: Vec<f32> = vec![y_ss; nplanes];
                for x in 0..w {
                    let idx = row_base + x;
                    for p in 0..nplanes {
                        let x0 = src_planes[p][idx];
                        let yv = c.a0 * x0
                            + c.a1 * xm1[p]
                            + c.a2 * xm2[p]
                            + c.a3 * xm3[p]
                            + c.b1 * ym1[p]
                            + c.b2 * ym2[p]
                            + c.b3 * ym3[p];
                        tmp[p][idx] = g1d * yv;
                        xm3[p] = xm2[p];
                        xm2[p] = xm1[p];
                        xm1[p] = x0;
                        ym3[p] = ym2[p];
                        ym2[p] = ym1[p];
                        ym1[p] = yv;
                    }
                }
                // Backward
                let last_idx = row_base + (w - 1);
                let mut xp1: Vec<f32> = (0..nplanes).map(|p| src_planes[p][last_idx]).collect();
                let mut xp2 = xp1.clone();
                let mut xp3 = xp1.clone();
                let mut yp1: Vec<f32> = vec![y_ss; nplanes];
                let mut yp2: Vec<f32> = vec![y_ss; nplanes];
                let mut yp3: Vec<f32> = vec![y_ss; nplanes];
                for x in (0..w).rev() {
                    let idx = row_base + x;
                    for p in 0..nplanes {
                        let x0 = src_planes[p][idx];
                        let yv = c.a0 * x0
                            + c.a1 * xp1[p]
                            + c.a2 * xp2[p]
                            + c.a3 * xp3[p]
                            + c.b1 * yp1[p]
                            + c.b2 * yp2[p]
                            + c.b3 * yp3[p];
                        tmp[p][idx] += g1d * yv;
                        xp3[p] = xp2[p];
                        xp2[p] = xp1[p];
                        xp1[p] = x0;
                        yp3[p] = yp2[p];
                        yp2[p] = yp1[p];
                        yp1[p] = yv;
                    }
                }
            }
        }

        // Vertical pass (fused) with runtime auto-parallelization.
        let mut out: Vec<Vec<f32>> = (0..nplanes).map(|_| vec![0.0f32; w * h]).collect();
        if should_parallelize_cols(w, h, nplanes) {
            // Compute each column in parallel to temporary per-column buffers, then scatter.
            let col_buffers: Vec<Vec<Vec<f32>>> = (0..w)
                .into_par_iter()
                .map(|x| {
                    // allocate per-plane column buffers of length h
                    let mut cols: Vec<Vec<f32>> = (0..nplanes).map(|_| vec![0.0f32; h]).collect();
                    // Forward down column
                    let top_idx = x;
                    let mut xm1: Vec<f32> = (0..nplanes).map(|p| tmp[p][top_idx]).collect();
                    let mut xm2 = xm1.clone();
                    let mut xm3 = xm1.clone();
                    let mut ym1: Vec<f32> = vec![y_ss; nplanes];
                    let mut ym2: Vec<f32> = vec![y_ss; nplanes];
                    let mut ym3: Vec<f32> = vec![y_ss; nplanes];
                    for y in 0..h {
                        let idx = y * w + x;
                        for p in 0..nplanes {
                            let x0 = tmp[p][idx];
                            let yv = c.a0 * x0
                                + c.a1 * xm1[p]
                                + c.a2 * xm2[p]
                                + c.a3 * xm3[p]
                                + c.b1 * ym1[p]
                                + c.b2 * ym2[p]
                                + c.b3 * ym3[p];
                            cols[p][y] = g1d * yv;
                            xm3[p] = xm2[p];
                            xm2[p] = xm1[p];
                            xm1[p] = x0;
                            ym3[p] = ym2[p];
                            ym2[p] = ym1[p];
                            ym1[p] = yv;
                        }
                    }
                    // Backward up column
                    let bottom_idx = (h - 1) * w + x;
                    let mut xp1: Vec<f32> = (0..nplanes).map(|p| tmp[p][bottom_idx]).collect();
                    let mut xp2 = xp1.clone();
                    let mut xp3 = xp1.clone();
                    let mut yp1: Vec<f32> = vec![y_ss; nplanes];
                    let mut yp2: Vec<f32> = vec![y_ss; nplanes];
                    let mut yp3: Vec<f32> = vec![y_ss; nplanes];
                    for y in (0..h).rev() {
                        let idx = y * w + x;
                        for p in 0..nplanes {
                            let x0 = tmp[p][idx];
                            let yv = c.a0 * x0
                                + c.a1 * xp1[p]
                                + c.a2 * xp2[p]
                                + c.a3 * xp3[p]
                                + c.b1 * yp1[p]
                                + c.b2 * yp2[p]
                                + c.b3 * yp3[p];
                            cols[p][y] += g1d * yv;
                            xp3[p] = xp2[p];
                            xp2[p] = xp1[p];
                            xp1[p] = x0;
                            yp3[p] = yp2[p];
                            yp2[p] = yp1[p];
                            yp1[p] = yv;
                        }
                    }
                    cols
                })
                .collect();
            // Scatter into out buffers
            for x in 0..w {
                for p in 0..nplanes {
                    for y in 0..h {
                        out[p][y * w + x] = col_buffers[x][p][y];
                    }
                }
            }
        } else {
            for x in 0..w {
                let top_idx = x;
                let mut xm1: Vec<f32> = (0..nplanes).map(|p| tmp[p][top_idx]).collect();
                let mut xm2 = xm1.clone();
                let mut xm3 = xm1.clone();
                let mut ym1: Vec<f32> = vec![y_ss; nplanes];
                let mut ym2: Vec<f32> = vec![y_ss; nplanes];
                let mut ym3: Vec<f32> = vec![y_ss; nplanes];
                for y in 0..h {
                    let idx = y * w + x;
                    for p in 0..nplanes {
                        let x0 = tmp[p][idx];
                        let yv = c.a0 * x0
                            + c.a1 * xm1[p]
                            + c.a2 * xm2[p]
                            + c.a3 * xm3[p]
                            + c.b1 * ym1[p]
                            + c.b2 * ym2[p]
                            + c.b3 * ym3[p];
                        out[p][idx] = g1d * yv;
                        xm3[p] = xm2[p];
                        xm2[p] = xm1[p];
                        xm1[p] = x0;
                        ym3[p] = ym2[p];
                        ym2[p] = ym1[p];
                        ym1[p] = yv;
                    }
                }
                let bottom_idx = (h - 1) * w + x;
                let mut xp1: Vec<f32> = (0..nplanes).map(|p| tmp[p][bottom_idx]).collect();
                let mut xp2 = xp1.clone();
                let mut xp3 = xp1.clone();
                let mut yp1: Vec<f32> = vec![y_ss; nplanes];
                let mut yp2: Vec<f32> = vec![y_ss; nplanes];
                let mut yp3: Vec<f32> = vec![y_ss; nplanes];
                for y in (0..h).rev() {
                    let idx = y * w + x;
                    for p in 0..nplanes {
                        let x0 = tmp[p][idx];
                        let yv = c.a0 * x0
                            + c.a1 * xp1[p]
                            + c.a2 * xp2[p]
                            + c.a3 * xp3[p]
                            + c.b1 * yp1[p]
                            + c.b2 * yp2[p]
                            + c.b3 * yp3[p];
                        out[p][idx] += g1d * yv;
                        xp3[p] = xp2[p];
                        xp2[p] = xp1[p];
                        xp1[p] = x0;
                        yp3[p] = yp2[p];
                        yp2[p] = yp1[p];
                        yp1[p] = yv;
                    }
                }
            }
        }

        out
    }

    // Removed explicit parallel FIR helper; YvV auto-parallelization covers throughput needs.

    // ========================= Tests =========================

    #[cfg(test)]
    mod tests {
        use super::*;

        fn approx_eq(a: f32, b: f32, eps: f32) -> bool {
            (a - b).abs() <= eps
        }

        #[test]
        fn blur_constant_plane_preserves_value_no_edges() {
            let w = 7usize;
            let h = 5usize;
            let sigma = 2.0f32;
            let plane = vec![1.0f32; w * h];
            let out = gaussian_blur_plane(&plane, w, h, sigma);
            for v in out {
                assert!(approx_eq(v, 1.0, 1e-6), "expected 1.0, got {}", v);
            }
        }

        #[test]
        fn rgba_blur_preserves_alpha() {
            // 2x2 image with varying alpha
            let w = 2usize;
            let h = 2usize;
            let px = vec![
                Pixel::new(10, 20, 30, 0),
                Pixel::new(40, 50, 60, 64),
                Pixel::new(70, 80, 90, 128),
                Pixel::new(100, 110, 120, 255),
            ];
            let out = gaussian_blur_pixels_rgba_bytes(&px, w, h, 1.5);
            let alphas: Vec<u8> = out.chunks_exact(4).map(|c| c[3]).collect();
            assert_eq!(alphas, vec![0, 64, 128, 255]);
        }

        #[test]
        fn fused_matches_separate_for_two_planes() {
            let w = 3usize;
            let h = 3usize;
            let sigma = 1.5f32;
            // simple gradient planes
            let p0: Vec<f32> = (0..w * h).map(|i| (i as f32) / 10.0).collect();
            let p1: Vec<f32> = (0..w * h).map(|i| 1.0 - (i as f32) / 12.0).collect();

            let s0 = gaussian_blur_plane(&p0, w, h, sigma);
            let s1 = gaussian_blur_plane(&p1, w, h, sigma);
            let fused = gaussian_blur_planes(&[&p0, &p1], w, h, sigma);

            for i in 0..(w * h) {
                assert!(
                    approx_eq(s0[i], fused[0][i], 1e-5),
                    "plane 0 mismatch at {}",
                    i
                );
                assert!(
                    approx_eq(s1[i], fused[1][i], 1e-5),
                    "plane 1 mismatch at {}",
                    i
                );
            }
        }

        #[test]
        fn yvv_constant_plane_preserves_value() {
            let (w, h) = (64usize, 64usize);
            let sigma = 6.0f32;
            let plane = vec![1.0f32; w * h];
            let out = yvv_gaussian_blur_plane(&plane, w, h, sigma);
            let mean: f32 = out.iter().sum::<f32>() / (w * h) as f32;
            assert!(approx_eq(mean, 1.0, 1e-3), "mean {}", mean);
        }

        #[test]
        fn yvv_impulse_is_symmetric() {
            let (w, h) = (65usize, 1usize);
            let sigma = 4.0f32;
            let mut plane = vec![0.0f32; w * h];
            let c = w / 2;
            plane[c] = 1.0;
            let out = yvv_gaussian_blur_plane(&plane, w, h, sigma);
            for i in 0..c {
                let l = out[c - i];
                let r = out[c + i];
                assert!(approx_eq(l, r, 1e-4), "asym at {}: {} vs {}", i, l, r);
            }
        }

        // Parallel helper-specific test removed (runtime auto-parallel now).
    }
}
