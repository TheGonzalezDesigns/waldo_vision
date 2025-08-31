use opencv::{
    core::{self, Mat, Rect, Scalar},
    imgproc,
    prelude::*,
    videoio::{self, VideoCapture, VideoWriter},
};
use std::env;
use std::sync::{Arc, Mutex};
use tokio::task::JoinSet;
use waldo_vision::pipeline::{ChunkStatus, FrameAnalysis, PipelineConfig, TrackedBlob, TrackedState, VisionPipeline};

#[tokio::main]
async fn main() -> opencv::Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 3 {
        println!("Usage: visual_tester <input_video_path> <output_video_path>");
        return Ok(());
    }
    let input_path = &args[1];
    let output_path = &args[2];

    let mut cap = VideoCapture::from_file(input_path, videoio::CAP_ANY)?;
    if !cap.is_opened()? {
        panic!("Error opening video file");
    }

    let frame_width = cap.get(videoio::CAP_PROP_FRAME_WIDTH)? as u32;
    let frame_height = cap.get(videoio::CAP_PROP_FRAME_HEIGHT)? as u32;
    let fps = cap.get(videoio::CAP_PROP_FPS)?;

    println!("Reading all frames into memory...");
    let mut frames = Vec::new();
    let mut frame = Mat::default();
    while cap.read(&mut frame)? {
        if frame.empty() { break; }
        frames.push(frame.clone());
    }
    println!("{} frames read.", frames.len());

    let config = Arc::new(PipelineConfig {
        image_width: frame_width,
        image_height: frame_height,
        chunk_width: 10,
        chunk_height: 10,
        new_age_threshold: 5,
        behavioral_anomaly_threshold: 3.0,
        absolute_min_blob_size: 2,
        blob_size_std_dev_filter: 2.0,
        global_disturbance_threshold: 0.25,
    });
    let pipeline = Arc::new(Mutex::new(VisionPipeline::new((*config).clone())));

    println!("Processing frames in parallel...");
    let mut join_set = JoinSet::new();
    for (i, frame) in frames.into_iter().enumerate() {
        let pipeline_clone = Arc::clone(&pipeline);
        let config_clone = Arc::clone(&config);
        join_set.spawn(async move {
            let mut pipeline = pipeline_clone.lock().unwrap();
            
            let mut rgba_frame = Mat::default();
            imgproc::cvt_color(&frame, &mut rgba_frame, imgproc::COLOR_BGR2RGBA, 0).unwrap();
            let frame_buffer: Vec<u8> = rgba_frame.data_bytes().unwrap().to_vec();

            let analysis = pipeline.process_frame(&frame_buffer);

            let mut output_frame = frame.clone();
            let mut heatmap_overlay = Mat::new_size_with_default(frame.size().unwrap(), core::CV_8UC3, Scalar::all(0.0)).unwrap();
            
            apply_dimming_and_heat(&mut output_frame, &mut heatmap_overlay, &analysis.status_map, frame_width, config_clone.chunk_width, config_clone.chunk_height);
            draw_tracked_blobs(&mut output_frame, &analysis.tracked_blobs, config_clone.chunk_width, config_clone.chunk_height);
            draw_header(&mut output_frame, i, &analysis);

            let mut final_frame = Mat::default();
            core::add_weighted(&output_frame, 1.0, &heatmap_overlay, 0.8, 0.0, &mut final_frame, -1).unwrap();
            
            (i, final_frame)
        });
    }

    println!("Collecting and re-ordering processed frames...");
    let mut processed_frames = Vec::with_capacity(join_set.len());
    while let Some(res) = join_set.join_next().await {
        processed_frames.push(res.unwrap());
    }
    processed_frames.sort_by_key(|k| k.0);

    println!("Writing output video...");
    let mut writer = VideoWriter::new(output_path, VideoWriter::fourcc('m', 'p', '4', 'v')?, fps, core::Size::new(frame_width as i32, frame_height as i32), true)?;
    for (_, frame) in processed_frames {
        writer.write(&frame)?;
    }

    println!("Processing complete. Output saved to {}", output_path);
    Ok(())
}

fn draw_header(frame: &mut Mat, frame_index: usize, analysis: &FrameAnalysis) {
    let header_height = 40;
    let rect = Rect::new(0, 0, frame.cols(), header_height);
    imgproc::rectangle(frame, rect, Scalar::new(0.0, 0.0, 0.0, 0.0), -1, imgproc::LINE_8, 0).unwrap();

    let status_text = if analysis.scene_is_stable { "STABLE" } else { "GLOBAL DISTURBANCE" };
    let event_text = format!("Frame: {} | Scene: {} | Significant Events: {}", frame_index, status_text, analysis.significant_event_count);
    
    let text_pos = core::Point::new(10, 25);
    imgproc::put_text(frame, &event_text, text_pos, imgproc::FONT_HERSHEY_SIMPLEX, 0.7, Scalar::new(255.0, 255.0, 255.0, 0.0), 2, imgproc::LINE_AA, false).unwrap();
}

// Other helper functions remain the same...
fn draw_tracked_blobs(frame: &mut Mat, tracked_blobs: &[TrackedBlob], chunk_w: u32, chunk_h: u32) {
    for blob in tracked_blobs {
        let color = state_to_color(&blob.state);
        for point in &blob.latest_blob.chunk_coords {
            let top_left = core::Point::new(point.x as i32 * chunk_w as i32, point.y as i32 * chunk_h as i32);
            let rect = Rect::new(top_left.x, top_left.y, chunk_w as i32, chunk_h as i32);
            let roi = Mat::roi(frame, rect).unwrap();
            let mut colored_roi = Mat::default();
            let color_mat = Mat::new_size_with_default(roi.size().unwrap(), roi.typ(), color).unwrap();
            core::add_weighted(&roi, 0.5, &color_mat, 0.5, 0.0, &mut colored_roi, -1).unwrap();
            colored_roi.copy_to(&mut Mat::roi(frame, rect).unwrap()).unwrap();
        }
        let (top_left_p, bottom_right_p) = blob.latest_blob.bounding_box;
        let top_left = core::Point::new(top_left_p.x as i32 * chunk_w as i32, top_left_p.y as i32 * chunk_h as i32);
        let bottom_right = core::Point::new((bottom_right_p.x + 1) as i32 * chunk_w as i32, (bottom_right_p.y + 1) as i32 * chunk_h as i32);
        let rect = Rect::new(top_left.x, top_left.y, bottom_right.x - top_left.x, bottom_right.y - top_left.y);
        imgproc::rectangle(frame, rect, color, 2, imgproc::LINE_8, 0).unwrap();
        let label = format!("ID: {} | S: {:?} | A: {}", blob.id, blob.state, blob.age);
        let text_pos = core::Point::new(rect.x, rect.y - 10);
        imgproc::put_text(frame, &label, text_pos, imgproc::FONT_HERSHEY_SIMPLEX, 0.5, color, 2, imgproc::LINE_AA, false).unwrap();
    }
}

fn state_to_color(state: &TrackedState) -> Scalar {
    match state {
        TrackedState::New => Scalar::new(0.0, 255.0, 255.0, 0.0),
        TrackedState::Tracking => Scalar::new(255.0, 100.0, 0.0, 0.0),
        TrackedState::Lost => Scalar::new(128.0, 128.0, 128.0, 0.0),
        TrackedState::Anomalous => Scalar::new(0.0, 0.0, 255.0, 0.0),
    }
}

fn apply_dimming_and_heat(frame: &mut Mat, heatmap: &mut Mat, status_map: &[ChunkStatus], width: u32, chunk_w: u32, chunk_h: u32) {
    let grid_w = width / chunk_w;
    for (i, status) in status_map.iter().enumerate() {
        let y = i as u32 / grid_w;
        let x = i as u32 % grid_w;
        let top_left = core::Point::new((x * chunk_w) as i32, (y * chunk_h) as i32);
        let rect = Rect::new(top_left.x, top_left.y, chunk_w as i32, chunk_h as i32);
        match status {
            ChunkStatus::Stable | ChunkStatus::Learning => {
                let roi = Mat::roi(frame, rect).unwrap();
                let mut dimmed_roi = Mat::default();
                core::multiply(&roi, &Scalar::all(0.4), &mut dimmed_roi, 1.0, -1).unwrap();
                dimmed_roi.copy_to(&mut Mat::roi(frame, rect).unwrap()).unwrap();
            }
            ChunkStatus::PredictableMotion => {
                let color = Scalar::new(255.0, 0.0, 0.0, 0.0);
                imgproc::rectangle(heatmap, rect, color, -1, imgproc::LINE_8, 0).unwrap();
            }
            ChunkStatus::AnomalousEvent(details) => {
                let score = details.luminance_score.clamp(0.0, 10.0);
                let (r, g, b) = if score <= 5.0 {
                    let ratio = score / 5.0;
                    (0.0, 255.0 * ratio, 255.0 * (1.0 - ratio))
                } else {
                    let ratio = (score - 5.0) / 5.0;
                    (255.0 * ratio, 255.0 * (1.0 - ratio), 0.0)
                };
                imgproc::rectangle(heatmap, rect, Scalar::new(b, g, r, 0.0), -1, imgproc::LINE_8, 0).unwrap();
            }
        }
    }
}
