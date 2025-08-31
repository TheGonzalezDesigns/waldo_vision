use opencv::{
    core::{self, Mat, Rect, Scalar},
    imgproc,
    prelude::*,
    videoio::{self, VideoCapture, VideoWriter},
};
use std::env;
use waldo_vision::pipeline::{ChunkStatus, PipelineConfig, TrackedBlob, VisionPipeline};

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
                    core::CV_8UC3, // BGR
                    Scalar::all(0.0),
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

                let report = pipeline.generate_report(&frame_buffer);

                // --- 6. Visualization ---
                // Start with a mutable clone of the original frame.
                let mut output_frame = frame.clone();

                let status_map = pipeline.get_last_status_map();

                // Create the colorful heatmap overlay.
                let mut heatmap_overlay = Mat::new_size_with_default(
                    frame.size()?,
                    core::CV_8UC3, // BGR
                    Scalar::all(0.0),
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

                // --- 8. Draw Tracked Blob Information ---
                draw_tracked_blobs(&mut output_frame, pipeline.get_tracked_blobs(), config.chunk_width, config.chunk_height);


                // --- 9. Final Blend ---
                let mut final_frame = Mat::default();
                core::add_weighted(&output_frame, 1.0, &heatmap_overlay, 0.8, 0.0, &mut final_frame, -1)?;

                // --- 10. Write Output Frame ---
                writer.write(&final_frame)?;
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

/// Draws color-coded bounding boxes and text labels for each tracked blob.
fn draw_tracked_blobs(frame: &mut Mat, tracked_blobs: &[TrackedBlob], chunk_w: u32, chunk_h: u32) {
    for blob in tracked_blobs {
        let (top_left_p, bottom_right_p) = blob.latest_blob.bounding_box;
        let top_left = core::Point::new(top_left_p.x as i32 * chunk_w as i32, top_left_p.y as i32 * chunk_h as i32);
        let bottom_right = core::Point::new((bottom_right_p.x + 1) as i32 * chunk_w as i32, (bottom_right_p.y + 1) as i32 * chunk_h as i32);
        let rect = Rect::new(top_left.x, top_left.y, bottom_right.x - top_left.x, bottom_right.y - top_left.y);

        // Assign a consistent color based on the blob's ID.
        let color = id_to_color(blob.id);

        // Draw the bounding box.
        imgproc::rectangle(frame, rect, color, 2, imgproc::LINE_8, 0).unwrap();

        // Prepare and draw the text label.
        let label = format!("ID: {} | Age: {}", blob.id, blob.age);
        let text_pos = core::Point::new(rect.x, rect.y - 10);
        imgproc::put_text(frame, &label, text_pos, imgproc::FONT_HERSHEY_SIMPLEX, 0.5, color, 2, imgproc::LINE_AA, false).unwrap();
    }
}

/// Generates a consistent, pseudo-random color for a given ID.
fn id_to_color(id: u64) -> Scalar {
    let mut r = (id * 123456789) % 256;
    let mut g = (id * 987654321) % 256;
    let mut b = (id * 456789123) % 256;
    // Ensure the color is bright and visible.
    r = (r + 100).min(255);
    g = (g + 100).min(255);
    b = (b + 100).min(255);
    Scalar::new(b as f64, g as f64, r as f64, 0.0)
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
                let roi = Mat::roi(frame, rect).unwrap();
                let mut dimmed_roi = Mat::default();
                core::multiply(&roi, &Scalar::all(0.4), &mut dimmed_roi, 1.0, -1).unwrap();
                dimmed_roi.copy_to(&mut Mat::roi(frame, rect).unwrap()).unwrap();
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