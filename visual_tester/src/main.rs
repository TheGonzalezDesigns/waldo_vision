use opencv::{
    core::{self, Mat, Rect, Scalar},
    imgproc,
    prelude::*,
    videoio::{self, VideoCapture, VideoWriter},
};
use std::env;
use std::sync::Arc;
use tokio::sync::Mutex;
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
        disturbance_entry_threshold: 0.25,
        disturbance_exit_threshold: 0.15,
        disturbance_confirmation_frames: 5,
    });
    
    // Create a single pipeline instance
    let mut pipeline = VisionPipeline::new((*config).clone());

    println!("Processing frames with controlled parallelism...");
    
    // Process frames in small batches to avoid memory explosion
    const BATCH_SIZE: usize = 8; // Process 8 frames at a time
    let total_frames = frames.len();
    let mut all_processed_frames = Vec::with_capacity(total_frames);
    
    for batch_start in (0..total_frames).step_by(BATCH_SIZE) {
        let batch_end = (batch_start + BATCH_SIZE).min(total_frames);
        let batch_frames = &frames[batch_start..batch_end];
        
        // Process this batch in parallel
        let mut batch_futures = JoinSet::new();
        
        for (local_idx, frame) in batch_frames.iter().enumerate() {
            let frame_idx = batch_start + local_idx;
            let frame_clone = frame.clone();
            let config_clone = Arc::clone(&config);
            
            // Convert frame to RGBA buffer
            let mut rgba_frame = Mat::default();
            imgproc::cvt_color(&frame_clone, &mut rgba_frame, imgproc::COLOR_BGR2RGBA, 0).unwrap();
            let frame_buffer: Vec<u8> = rgba_frame.data_bytes().unwrap().to_vec();
            
            batch_futures.spawn(async move {
                // Create a temporary pipeline for parallel processing
                // This avoids lock contention
                let mut temp_pipeline = VisionPipeline::new((*config_clone).clone());
                let analysis = temp_pipeline.process_frame(&frame_buffer).await;
                
                // Process visualization
                let mut output_frame = frame_clone.clone();
                let mut heatmap_overlay = Mat::new_size_with_default(
                    frame_clone.size().unwrap(), 
                    core::CV_8UC3, 
                    Scalar::all(0.0)
                ).unwrap();
                
                apply_dimming_and_heat(
                    &mut output_frame, 
                    &mut heatmap_overlay, 
                    &analysis.status_map, 
                    config_clone.image_width, 
                    config_clone.chunk_width, 
                    config_clone.chunk_height
                );
                draw_tracked_blobs(
                    &mut output_frame, 
                    &analysis.tracked_blobs, 
                    config_clone.chunk_width, 
                    config_clone.chunk_height
                );
                draw_header(&mut output_frame, frame_idx, &analysis);

                let mut final_frame = Mat::default();
                core::add_weighted(&output_frame, 1.0, &heatmap_overlay, 0.8, 0.0, &mut final_frame, -1).unwrap();
                
                (frame_idx, final_frame)
            });
        }
        
        // Collect this batch's results
        let mut batch_results = Vec::new();
        while let Some(res) = batch_futures.join_next().await {
            batch_results.push(res.unwrap());
        }
        
        // Sort batch results by frame index
        batch_results.sort_by_key(|k| k.0);
        all_processed_frames.extend(batch_results);
        
        // Progress indicator
        println!("Processed frames {}-{} of {}", batch_start, batch_end.min(total_frames), total_frames);
    }

    println!("Writing output video...");
    let mut writer = VideoWriter::new(
        output_path, 
        VideoWriter::fourcc('m', 'p', '4', 'v')?, 
        fps, 
        core::Size::new(frame_width as i32, frame_height as i32), 
        true
    )?;
    
    for (_, frame) in all_processed_frames {
        writer.write(&frame)?;
    }

    println!("Processing complete. Output saved to {}", output_path);
    Ok(())
}

fn draw_header(frame: &mut Mat, frame_index: usize, analysis: &FrameAnalysis) {
    let text = format!("Frame {} | State: {:?} | Events: {}", 
        frame_index, 
        analysis.scene_state,
        analysis.significant_event_count
    );
    imgproc::put_text(
        frame, 
        &text, 
        core::Point::new(10, 30), 
        imgproc::FONT_HERSHEY_SIMPLEX, 
        0.6, 
        Scalar::new(255.0, 255.0, 255.0, 0.0), 
        2, 
        imgproc::LINE_AA, 
        false
    ).unwrap();
}

fn apply_dimming_and_heat(frame: &mut Mat, heat_overlay: &mut Mat, status_map: &[ChunkStatus], frame_width: u32, chunk_width: u32, chunk_height: u32) {
    let grid_width = frame_width / chunk_width;
    // O(n) where n = number of chunks
    for (i, status) in status_map.iter().enumerate() {
        let chunk_y = i as u32 / grid_width;
        let chunk_x = i as u32 % grid_width;
        let rect = Rect::new(
            (chunk_x * chunk_width) as i32, 
            (chunk_y * chunk_height) as i32, 
            chunk_width as i32, 
            chunk_height as i32
        );
        
        match status {
            ChunkStatus::Stable => {
                let mut roi = Mat::roi(frame, rect).unwrap();
                let mut dimmed = Mat::default();
                core::multiply(&roi, &Scalar::all(0.7), &mut dimmed, 1.0, -1).unwrap();
                dimmed.copy_to(&mut roi).unwrap();
            },
            ChunkStatus::AnomalousEvent(details) => {
                let intensity = ((details.luminance_score * 20.0).min(255.0)) as f64;
                let heat_color = Scalar::new(0.0, intensity * 0.5, intensity, 0.0);
                imgproc::rectangle(heat_overlay, rect, heat_color, -1, imgproc::LINE_8, 0).unwrap();
            },
            _ => {}
        }
    }
}

fn draw_tracked_blobs(frame: &mut Mat, tracked_blobs: &[TrackedBlob], chunk_width: u32, chunk_height: u32) {
    // O(n*m) where n = number of blobs, m = chunks per blob
    for blob in tracked_blobs {
        let color = state_to_color(&blob.state);
        
        // Fill the blob area with a semi-transparent colored overlay
        for chunk_coord in &blob.latest_blob.chunk_coords {
            let chunk_rect = Rect::new(
                (chunk_coord.x * chunk_width) as i32,
                (chunk_coord.y * chunk_height) as i32,
                chunk_width as i32,
                chunk_height as i32
            );
            let colored_overlay = Mat::new_size_with_default(
                core::Size::new(chunk_width as i32, chunk_height as i32),
                core::CV_8UC3,
                color
            ).unwrap();
            let mut roi = Mat::roi(frame, chunk_rect).unwrap();
            let mut blended = Mat::default();
            core::add_weighted(&roi, 0.6, &colored_overlay, 0.4, 0.0, &mut blended, -1).unwrap();
            blended.copy_to(&mut roi).unwrap();
        }
        
        // Draw bounding box
        let (min_point, max_point) = &blob.latest_blob.bounding_box;
        let rect = Rect::new(
            (min_point.x * chunk_width) as i32,
            (min_point.y * chunk_height) as i32,
            ((max_point.x - min_point.x + 1) * chunk_width) as i32,
            ((max_point.y - min_point.y + 1) * chunk_height) as i32
        );
        imgproc::rectangle(frame, rect, color, 2, imgproc::LINE_8, 0).unwrap();
        
        // Draw detailed label
        let label = format!("ID: {} | S: {:?} | A: {}", blob.id, blob.state, blob.age);
        let text_pos = core::Point::new(rect.x, rect.y - 10);
        imgproc::put_text(frame, &label, text_pos, imgproc::FONT_HERSHEY_SIMPLEX, 0.5, color, 2, imgproc::LINE_AA, false).unwrap();
    }
}

fn state_to_color(state: &TrackedState) -> Scalar {
    match state {
        TrackedState::New => Scalar::new(0.0, 255.0, 255.0, 0.0),      // Cyan
        TrackedState::Tracking => Scalar::new(255.0, 100.0, 0.0, 0.0),  // Orange
        TrackedState::Lost => Scalar::new(128.0, 128.0, 128.0, 0.0),    // Gray
        TrackedState::Anomalous => Scalar::new(0.0, 0.0, 255.0, 0.0),   // Red
    }
}