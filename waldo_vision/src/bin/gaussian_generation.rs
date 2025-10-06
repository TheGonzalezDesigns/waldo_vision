use std::env;
use std::path::PathBuf;

use image::{ImageBuffer, Rgba};
use waldo_vision::core_modules::D1::pixel::pixel::{Pixel, Pixels};
use waldo_vision::core_modules::gaussian_engine::gaussian_engine::gaussian_blur_pixels_rgba_bytes;

fn main() {
    // Args: [input_path] [sigma] [output_path]
    let args: Vec<String> = env::args().collect();

    let default_input =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../assets/marilyn_monroe.jpg");
    let input_path = args.get(1).map(PathBuf::from).unwrap_or(default_input);

    let sigma: f32 = args
        .get(2)
        .and_then(|s| s.parse::<f32>().ok())
        .unwrap_or(3.0);

    let default_output =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../assets/marilyn_monroe_gaussian.png");
    let output_path = args.get(3).map(PathBuf::from).unwrap_or(default_output);

    println!("Loading image: {}", input_path.display());
    let img = match image::open(&input_path) {
        Ok(i) => i,
        Err(e) => {
            eprintln!("Failed to load image: {}", e);
            std::process::exit(1);
        }
    };

    let rgba = img.to_rgba8();
    let (w, h) = rgba.dimensions();
    let w_us = w as usize;
    let h_us = h as usize;

    // Convert RGBA bytes -> Pixels
    let rgba_bytes = rgba.into_raw();
    let pixels: Vec<Pixel> = match Pixels::try_from(rgba_bytes) {
        Ok(pv) => pv.into(),
        Err(e) => {
            eprintln!("Invalid RGBA buffer: {}", e);
            std::process::exit(1);
        }
    };

    println!("Running Gaussian blur (color) on {}x{} image...", w, h);
    let out_bytes = gaussian_blur_pixels_rgba_bytes(&pixels, w_us, h_us, sigma);

    let out_img: ImageBuffer<Rgba<u8>, Vec<u8>> =
        ImageBuffer::from_vec(w, h, out_bytes).expect("buffer size mismatch");

    if let Err(e) = out_img.save(&output_path) {
        eprintln!("Failed to save output image: {}", e);
        std::process::exit(1);
    }
    println!("Saved color blur result to {}", output_path.display());
}
