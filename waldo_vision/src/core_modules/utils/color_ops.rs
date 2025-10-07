//! Lightweight color math helpers over linear RGB triples (0..1).
//! These avoid constructing `Pixel` and any sRGB round-trips.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ColorMode {
    Optimal,
    Accurate,
}

#[cfg(feature = "accurate")]
pub const DEFAULT_COLOR_MODE: ColorMode = ColorMode::Accurate;
#[cfg(not(feature = "accurate"))]
pub const DEFAULT_COLOR_MODE: ColorMode = ColorMode::Optimal;

#[inline]
pub fn luminance_601_computed_0_255(r_comp: f64, g_comp: f64, b_comp: f64) -> f64 {
    // Classic Rec.601 luma on 0..255 computed channels.
    (0.299f64 * r_comp + 0.587f64 * g_comp + 0.114f64 * b_comp)
}

#[inline]
pub fn luminance_709_linear_0_255(r_lin: f32, g_lin: f32, b_lin: f32) -> f64 {
    // ITU-R BT.709 luma Y from linear RGB (0..1). Scale to 0..255.
    let y = 0.2126f32 * r_lin + 0.7152f32 * g_lin + 0.0722f32 * b_lin;
    (y.clamp(0.0, 1.0) * 255.0) as f64
}

#[inline]
pub fn luminance_601_norm_0_255(r_n: f64, g_n: f64, b_n: f64) -> f64 {
    // Apply Rec.601 weights on normalized sRGB and scale to 0..255.
    luminance_601_computed_0_255(r_n * 255.0, g_n * 255.0, b_n * 255.0)
}

#[inline]
pub fn color_sum_linear_0_765(r_lin: f32, g_lin: f32, b_lin: f32) -> f64 {
    // Sum of channels, each scaled to 0..255, matches 0..765 range.
    ((r_lin + g_lin + b_lin).clamp(0.0, 3.0) * 255.0) as f64
}

#[inline]
pub fn color_sum_norm_0_765(r_n: f64, g_n: f64, b_n: f64) -> f64 {
    ((r_n + g_n + b_n).clamp(0.0, 3.0) * 255.0) as f64
}

#[inline]
pub fn hue_from_linear_rgb_deg(r_lin: f32, g_lin: f32, b_lin: f32) -> f64 {
    let r = r_lin as f64;
    let g = g_lin as f64;
    let b = b_lin as f64;
    let maxc = r.max(g.max(b));
    let minc = r.min(g.min(b));
    let chroma = maxc - minc;
    if chroma <= 1e-12 {
        return 0.0;
    }
    let inv_c = 1.0 / chroma;
    let (base, sector) = if maxc == r {
        (g - b, 0.0)
    } else if maxc == g {
        (b - r, 2.0)
    } else {
        (r - g, 4.0)
    };
    let mut h = (base * inv_c + sector) * 60.0;
    if h < 0.0 {
        h += 360.0;
    }
    h
}

#[inline]
pub fn hue_from_normalized_rgb_deg(r_n: f64, g_n: f64, b_n: f64) -> f64 {
    let maxc = r_n.max(g_n.max(b_n));
    let minc = r_n.min(g_n.min(b_n));
    let chroma = maxc - minc;
    if chroma <= 1e-12 {
        return 0.0;
    }
    let inv_c = 1.0 / chroma;
    let (base, sector) = if maxc == r_n {
        (g_n - b_n, 0.0)
    } else if maxc == g_n {
        (b_n - r_n, 2.0)
    } else {
        (r_n - g_n, 4.0)
    };
    let mut h = (base * inv_c + sector) * 60.0;
    if h < 0.0 {
        h += 360.0;
    }
    h
}

#[inline]
pub fn hue_bias_from_hue_deg(h_deg: f64, target_deg: f64) -> f64 {
    let h_rad = h_deg.to_radians();
    let target = target_deg.to_radians();
    let v = 0.5 * (1.0 + (h_rad - target).cos());
    if v.is_finite() {
        v.clamp(0.0, 1.0)
    } else {
        0.5
    }
}

// ----------------------- Value / Lightness / Chroma -----------------------

#[inline]
pub fn value_from_norm(r: f64, g: f64, b: f64) -> f64 {
    r.max(g.max(b))
}

#[inline]
pub fn value_from_linear(r: f64, g: f64, b: f64) -> f64 {
    r.max(g.max(b))
}

#[inline]
pub fn lightness_hsl_from_norm(r: f64, g: f64, b: f64) -> f64 {
    let maxc = r.max(g.max(b));
    let minc = r.min(g.min(b));
    0.5 * (maxc + minc)
}

#[inline]
pub fn lightness_hsl_from_linear(r: f64, g: f64, b: f64) -> f64 {
    let maxc = r.max(g.max(b));
    let minc = r.min(g.min(b));
    0.5 * (maxc + minc)
}

#[inline]
pub fn chroma_from_norm(r: f64, g: f64, b: f64) -> f64 {
    let maxc = r.max(g.max(b));
    let minc = r.min(g.min(b));
    maxc - minc
}

#[inline]
pub fn chroma_from_linear(r: f64, g: f64, b: f64) -> f64 {
    let maxc = r.max(g.max(b));
    let minc = r.min(g.min(b));
    maxc - minc
}

// ----------------------------- Saturations -----------------------------

#[inline]
pub fn saturation_hsv_from_norm(r: f64, g: f64, b: f64) -> f64 {
    let v = value_from_norm(r, g, b);
    if v <= 1e-12 {
        0.0
    } else {
        chroma_from_norm(r, g, b) / v
    }
}

#[inline]
pub fn saturation_hsv_from_linear(r: f64, g: f64, b: f64) -> f64 {
    let v = value_from_linear(r, g, b);
    if v <= 1e-12 {
        0.0
    } else {
        chroma_from_linear(r, g, b) / v
    }
}

#[inline]
pub fn saturation_hsl_from_norm(r: f64, g: f64, b: f64) -> f64 {
    let l = lightness_hsl_from_norm(r, g, b);
    let denom = 1.0 - (2.0 * l - 1.0).abs();
    if denom <= 1e-12 {
        0.0
    } else {
        chroma_from_norm(r, g, b) / denom
    }
}

#[inline]
pub fn saturation_hsl_from_linear(r: f64, g: f64, b: f64) -> f64 {
    let l = lightness_hsl_from_linear(r, g, b);
    let denom = 1.0 - (2.0 * l - 1.0).abs();
    if denom <= 1e-12 {
        0.0
    } else {
        chroma_from_linear(r, g, b) / denom
    }
}

// ----------------------- Colorfulness / Achromaticity -----------------------

#[inline]
pub fn colorfulness_from_norm(r: f64, g: f64, b: f64) -> f64 {
    chroma_from_norm(r, g, b)
}

#[inline]
pub fn colorfulness_from_linear(r: f64, g: f64, b: f64) -> f64 {
    chroma_from_linear(r, g, b)
}

#[inline]
pub fn achromaticity_from_norm(r: f64, g: f64, b: f64) -> f64 {
    let v = value_from_norm(r, g, b);
    if v <= 1e-12 {
        1.0
    } else {
        1.0 - chroma_from_norm(r, g, b) / v
    }
}

#[inline]
pub fn achromaticity_from_linear(r: f64, g: f64, b: f64) -> f64 {
    let v = value_from_linear(r, g, b);
    if v <= 1e-12 {
        1.0
    } else {
        1.0 - chroma_from_linear(r, g, b) / v
    }
}

// --------------------------- Channel spread ---------------------------

#[inline]
pub fn channel_stddev_from_norm(r: f64, g: f64, b: f64) -> f64 {
    let m = (r + g + b) / 3.0;
    let rv = (r - m).powi(2);
    let gv = (g - m).powi(2);
    let bv = (b - m).powi(2);
    ((rv + gv + bv) / 3.0).sqrt()
}

#[inline]
pub fn channel_stddev_from_linear(r: f64, g: f64, b: f64) -> f64 {
    let m = (r + g + b) / 3.0;
    let rv = (r - m).powi(2);
    let gv = (g - m).powi(2);
    let bv = (b - m).powi(2);
    ((rv + gv + bv) / 3.0).sqrt()
}

// ---------------------------- Chromaticity xy ----------------------------

#[inline]
pub fn chromaticity_xy_from_norm(r: f64, g: f64, b: f64) -> (f64, f64) {
    let x = 0.4124564 * r + 0.3575761 * g + 0.1804375 * b;
    let y = 0.2126729 * r + 0.7151522 * g + 0.0721750 * b;
    let z = 0.0193339 * r + 0.1191920 * g + 0.9503041 * b;
    let s = x + y + z;
    if s <= 1e-12 {
        (0.0, 0.0)
    } else {
        (x / s, y / s)
    }
}

#[inline]
pub fn chromaticity_xy_from_linear(r: f64, g: f64, b: f64) -> (f64, f64) {
    let x = 0.4124564 * r + 0.3575761 * g + 0.1804375 * b;
    let y = 0.2126729 * r + 0.7151522 * g + 0.0721750 * b;
    let z = 0.0193339 * r + 0.1191920 * g + 0.9503041 * b;
    let s = x + y + z;
    if s <= 1e-12 {
        (0.0, 0.0)
    } else {
        (x / s, y / s)
    }
}
