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
                // Start with a mutable clone of the original frame.
                let mut output_frame = frame.clone();

                let status_map = pipeline.get_last_status_map();

                // Create the colorful heatmap overlay.
                let mut heatmap_overlay = Mat::new_size_with_default(
                    frame.size()?,
                    core::CV_8UC3, // BGR, no alpha needed for this part
                    Scalar::new(0.0, 0.0, 0.0, 0.0),
                )?;

                // --- 7. Apply Dimming and Draw Heatmap ---
                apply_dimming_and_heat(
                    &mut output_frame,
                    &mut heatmap_overlay,
                    status_map,
                    frame_width,
                    config.chunk_width,
                    config.chunk_height,
                );

                // --- 8. Final Blend ---
                // Blend the heatmap colors onto the now partially-dimmed output frame.
                core::add_weighted(&output_frame, 1.0, &heatmap_overlay, 0.8, 0.0, &mut output_frame, -1)?;

                // --- 9. Write Output Frame ---
                writer.write(&output_frame)?;
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

/// Applies dimming to inactive chunks and draws heatmap colors for active chunks.
fn apply_dimming_and_heat(
    frame: &mut Mat,
    heatmap: &mut Mat,
    status_map: &[ChunkStatus],
    width: u32,
    chunk_w: u32,
    chunk_h: u32,
) {
    let grid_w = width / chunk_w;
    for (i, status) in status_map.iter().enumerate() {
        let y = i as u32 / grid_w;
        let x = i as u32 % grid_w;

        let top_left = core::Point::new((x * chunk_w) as i32, (y * chunk_h) as i32);
        let rect = Rect::new(top_left.x, top_left.y, chunk_w as i32, chunk_h as i32);

        match status {
            ChunkStatus::Stable | ChunkStatus::Learning => {
                // This is an inactive chunk, so we dim it.
                let mut roi = Mat::roi(frame, rect).unwrap();
                // Multiply the region by 0.4 to make it darker.
                core::multiply(&roi, &Scalar::all(0.4), &mut roi, 1.0, -1).unwrap();
            }
            ChunkStatus::PredictableMotion => {
                // This is an active chunk, so we draw its heatmap color.
                let color = Scalar::new(255.0, 0.0, 0.0, 0.0); // Blue
                imgproc::rectangle(heatmap, rect, color, -1, imgproc::LINE_8, 0).unwrap();
            }
            ChunkStatus::AnomalousEvent(details) => {
                // This is a highly active chunk, so we draw its heatmap color.
                let score = details.luminance_score.clamp(0.0, 10.0);
                let r: f64;
                let g: f64;
                let b: f64;

                if score <= 5.0 { // Blue to Yellow
                    let ratio = score / 5.0;
                    b = 255.0 * (1.0 - ratio);
                    g = 255.0 * ratio;
                    r = 0.0;
                } else { // Yellow to Red
                    let ratio = (score - 5.0) / 5.0;
                    b = 0.0;
                    g = 255.0 * (1.0 - ratio);
                    r = 255.0 * ratio;
                }
                let color = Scalar::new(b, g, r, 0.0);
                imgproc::rectangle(heatmap, rect, color, -1, imgproc::LINE_8, 0).unwrap();
            }
        }
    }
}