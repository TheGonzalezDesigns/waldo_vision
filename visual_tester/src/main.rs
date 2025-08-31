use opencv::{
    core::{self, Mat, Rect, Scalar},
    imgproc,
    prelude::*,
    videoio::{self, VideoCapture, VideoWriter},
};
use std::env;
use waldo_vision::pipeline::{ChunkStatus, PipelineConfig, VisionPipeline};

fn main() -> opencv::Result<()> {
    // --- 1. Argument Parsing & Setup ---
    let args: Vec<String> = env::args().collect();
    if args.len() < 3 {
        println!("Usage: visual_tester <input_video_path> <output_video_path>");
        return Ok(());
    }
    let input_path = &args[1];
    let output_path = &args[2];

    // --- 2. Video I/O Initialization ---
    let mut cap = VideoCapture::from_file(input_path, videoio::CAP_ANY)?;
    if !cap.is_opened()? {
        panic!("Error opening video file");
    }

    let frame_width = cap.get(videoio::CAP_PROP_FRAME_WIDTH)? as u32;
    let frame_height = cap.get(videoio::CAP_PROP_FRAME_HEIGHT)? as u32;
    let fps = cap.get(videoio::CAP_PROP_FPS)?;

    let fourcc = VideoWriter::fourcc('m', 'p', '4', 'v')?;
    let mut writer = VideoWriter::new(
        output_path,
        fourcc,
        fps,
        core::Size::new(frame_width as i32, frame_height as i32),
        true,
    )?;

    // --- 3. Vision Pipeline Initialization ---
    let config = PipelineConfig {
        image_width: frame_width,
        image_height: frame_height,
        chunk_width: 10,
        chunk_height: 10,
        significance_age_threshold: 5, // Report moments that are 5 frames old or younger.
        absolute_min_blob_size: 2, // A blob must be at least 2 chunks to be considered.
        blob_size_std_dev_filter: 2.0, // Filter blobs 2 std devs below the mean size.
    };
    let mut pipeline = VisionPipeline::new(config.clone());

    // --- 4. Main Processing Loop ---
    let mut frame = Mat::default();
    loop {
        match cap.read(&mut frame) {
            Ok(true) => {
                if frame.empty() {
                    break;
                }

                // --- 5. Frame Conversion & Pipeline Processing ---
                // Convert the OpenCV Mat (BGR) to an RGBA buffer for our pipeline.
                let mut rgba_frame = Mat::default();
                imgproc::cvt_color(&frame, &mut rgba_frame, imgproc::COLOR_BGR2RGBA, 0)?;
                let frame_buffer: Vec<u8> = rgba_frame.data_bytes()?.to_vec();

                let _report = pipeline.generate_report(&frame_buffer);

                // --- 6. Visualization ---
                // Create a semi-transparent black overlay to dim the static parts of the scene.
                let mut dimming_overlay = Mat::new_size_with_default(
                    frame.size()?,
                    core::CV_8UC4,
                    Scalar::new(0.0, 0.0, 0.0, 100.0), // Semi-transparent black
                )?;

                let status_map = pipeline.get_last_status_map();
                
                // Create a second overlay for the colorful heatmap.
                let mut heatmap_overlay = Mat::new_size_with_default(
                    frame.size()?,
                    core::CV_8UC4,
                    Scalar::new(0.0, 0.0, 0.0, 0.0), // Fully transparent
                )?;

                draw_heat_and_cutouts(
                    &mut dimming_overlay,
                    &mut heatmap_overlay,
                    status_map,
                    frame_width,
                    config.chunk_width,
                    config.chunk_height,
                );

                // --- 7. Blending ---
                // First, convert the original frame to BGRA to blend with overlays.
                let mut bgra_frame = Mat::default();
                imgproc::cvt_color(&frame, &mut bgra_frame, imgproc::COLOR_BGR2BGRA, 0)?;

                // Blend the dimming layer onto the frame.
                let mut dimmed_frame = Mat::default();
                core::add_weighted(&bgra_frame, 1.0, &dimming_overlay, 1.0, 0.0, &mut dimmed_frame, -1)?;

                // Blend the heatmap layer on top of the dimmed frame.
                let mut final_frame_bgra = Mat::default();
                core::add_weighted(&dimmed_frame, 1.0, &heatmap_overlay, 1.0, 0.0, &mut final_frame_bgra, -1)?;

                // Convert back to BGR for the video writer.
                let mut final_frame_bgr = Mat::default();
                imgproc::cvt_color(&final_frame_bgra, &mut final_frame_bgr, imgproc::COLOR_BGRA2BGR, 0)?;

                // --- 8. Write Output Frame ---
                writer.write(&final_frame_bgr)?;
            }
            Ok(false) => {
                // End of video
                break;
            }
            Err(e) => {
                println!("Error reading frame: {:?}", e);
                break;
            }
        }
    }

    println!("Processing complete. Output saved to {}", output_path);
    Ok(())
}

/// Draws the heatmap and cuts out active regions from the dimming overlay.
fn draw_heat_and_cutouts(
    dimming_overlay: &mut Mat,
    heatmap_overlay: &mut Mat,
    status_map: &[ChunkStatus],
    width: u32,
    chunk_w: u32,
    chunk_h: u32,
) {
    let grid_w = width / chunk_w;
    for (i, status) in status_map.iter().enumerate() {
        if matches!(status, ChunkStatus::Stable | ChunkStatus::Learning) {
            continue; // Leave these areas dimmed.
        }

        let y = i as u32 / grid_w;
        let x = i as u32 % grid_w;

        let top_left = core::Point::new((x * chunk_w) as i32, (y * chunk_h) as i32);
        let rect = Rect::new(top_left.x, top_left.y, chunk_w as i32, chunk_h as i32);

        // "Cut out" the active region from the dimming overlay by making it transparent.
        imgproc::rectangle(dimming_overlay, rect, Scalar::new(0.0, 0.0, 0.0, 0.0), -1, imgproc::LINE_8, 0).unwrap();

        // Draw the corresponding heatmap color on the separate heatmap overlay.
        let color = match status {
            ChunkStatus::PredictableMotion => Scalar::new(255.0, 0.0, 0.0, 150.0), // Semi-transparent Blue
            ChunkStatus::AnomalousEvent(details) => {
                // Map the significance score to a Blue -> Yellow -> Red gradient.
                let score = details.luminance_score.clamp(0.0, 10.0);
                let r: f64;
                let g: f64;
                let b: f64;

                if score <= 5.0 {
                    let ratio = score / 5.0;
                    b = 255.0 * (1.0 - ratio);
                    g = 255.0 * ratio;
                    r = 0.0;
                } else {
                    let ratio = (score - 5.0) / 5.0;
                    b = 0.0;
                    g = 255.0 * (1.0 - ratio);
                    r = 255.0 * ratio;
                }
                Scalar::new(b, g, r, 150.0) // Semi-transparent
            }
            _ => Scalar::new(0.0, 0.0, 0.0, 0.0), // Should not happen
        };

        imgproc::rectangle(heatmap_overlay, rect, color, -1, imgproc::LINE_8, 0).unwrap();
    }
}