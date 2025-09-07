use opencv::{
    core::{self, Mat, Rect, Scalar},
    imgproc,
    prelude::*,
    videoio::{self, VideoCapture, VideoWriter},
};
use std::env;
use std::sync::{Arc, Mutex};
use tokio::task::JoinSet;
use waldo_vision::pipeline::{
    ChunkStatus, FrameAnalysis, PipelineConfig, TrackedBlob, TrackedState, VisionPipeline,
};
#[cfg(feature = "web")]
use waldo_vision_visualizer::{FrameBus, FrameFormat, FramePacket, Meta, MetaBlobSummary, ServerConfig};
use std::time::{SystemTime, UNIX_EPOCH};

#[tokio::main]
async fn main() -> opencv::Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 3 {
        println!("Usage: visual_tester [--serve <addr>] <input_video_path> <output_video_path>");
        return Ok(());
    }
    #[cfg(feature = "web")]
    let mut serve_addr: Option<String> = None;
    let (input_path, output_path) = if args.len() >= 5 && args[1] == "--serve" {
        #[cfg(feature = "web")]
        {
            serve_addr = Some(args[2].clone());
        }
        (&args[3], &args[4])
    } else {
        (&args[1], &args[2])
    };

    let mut cap = VideoCapture::from_file(input_path, videoio::CAP_ANY)?;
    if !cap.is_opened()? {
        panic!("Error opening video file");
    }

    let frame_width = cap.get(videoio::CAP_PROP_FRAME_WIDTH)? as u32;
    let frame_height = cap.get(videoio::CAP_PROP_FRAME_HEIGHT)? as u32;
    let fps = cap.get(videoio::CAP_PROP_FPS)?;

    #[cfg(feature = "web")]
    let (frame_bus, mut play_rx): (Option<FrameBus>, Option<tokio::sync::watch::Receiver<bool>>) = serve_addr.as_ref().map(|addr| {
        println!("Starting visualizer server at {}...", addr);
        let bus = FrameBus::new(2);
        let cfg = ServerConfig { bind_addr: addr.clone(), nat_public_ip: None, udp_port_start: None, udp_port_end: None };
        let (play_tx, play_rx) = tokio::sync::watch::channel(false);
        let control = waldo_vision_visualizer::ControlHandle { play_tx };
        tokio::spawn({
            let bus_clone = bus.clone();
            let control_clone = control.clone();
            async move { let _ = waldo_vision_visualizer::start_server(bus_clone, cfg, control_clone).await; }
        });
        (Some(bus), Some(play_rx))
    }).unwrap_or((None, None));

    #[cfg(feature = "web")]
    if let Some(addr) = &serve_addr {
        println!("Visualizer ready. Open http://{} and press Play to start.", addr);
    }

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
    let pipeline = Arc::new(Mutex::new(VisionPipeline::new((*config).clone())));

    // Sequential processing with play/pause control
    println!("Waiting for Play...");
    let mut writer = VideoWriter::new(
        output_path,
        VideoWriter::fourcc('m', 'p', '4', 'v')?,
        fps,
        core::Size::new(frame_width as i32, frame_height as i32),
        true,
    )?;
    let mut frame_index: usize = 0;
    let mut frame = Mat::default();
    #[cfg(feature = "web")]
    let stream_enabled = serve_addr.is_some();
    use tokio::time::{sleep, Duration};
    let frame_delay = if fps > 0.0 { Duration::from_secs_f64(1.0 / fps.max(1.0)) } else { Duration::from_millis(33) };
    loop {
        // Wait for play command from UI
        #[cfg(feature = "web")]
        if let Some(rx) = &mut play_rx {
            use tokio::time::{sleep, Duration};
            while !*rx.borrow() { sleep(Duration::from_millis(50)).await; }
        }
        if !cap.read(&mut frame)? || frame.empty() { break; }

        let mut pipeline = pipeline.lock().unwrap();
        let mut rgba_frame = Mat::default();
        imgproc::cvt_color(&frame, &mut rgba_frame, imgproc::COLOR_BGR2RGBA, 0).unwrap();
        let frame_buffer: Vec<u8> = rgba_frame.data_bytes().unwrap().to_vec();

        let analysis = pipeline.process_frame(&frame_buffer);

        let mut output_frame = frame.clone();
        let mut heatmap_overlay = Mat::new_size_with_default(frame.size().unwrap(), core::CV_8UC3, Scalar::all(0.0)).unwrap();
        apply_dimming_and_heat(&mut output_frame, &mut heatmap_overlay, &analysis.status_map, frame_width, config.chunk_width, config.chunk_height);
        draw_tracked_blobs(&mut output_frame, &analysis.tracked_blobs, config.chunk_width, config.chunk_height);
        draw_header(&mut output_frame, frame_index, &analysis);

        let mut final_frame = Mat::default();
        core::add_weighted(&output_frame, 1.0, &heatmap_overlay, 0.8, 0.0, &mut final_frame, -1).unwrap();

        #[cfg(feature = "web")]
        if stream_enabled {
            if let Some(bus) = &frame_bus {
                let mut rgba_out = Mat::default();
                imgproc::cvt_color(&final_frame, &mut rgba_out, imgproc::COLOR_BGR2RGBA, 0).unwrap();
                let bytes = rgba_out.data_bytes().unwrap().to_vec();
                if let Some(pkt) = encode_jpeg(&bytes, frame_width, frame_height, 60) {
                    let ts = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_millis() as u64;
                    let _ = bus.frames_tx.send(FramePacket { ts_millis: ts, width: frame_width, height: frame_height, format: FrameFormat::Jpeg, data: pkt.into() });
                    let meta = Meta { scene_state: format!("{:?}", analysis.scene_state), event_count: analysis.significant_event_count, blobs: analysis.tracked_blobs.iter().map(|b| { let (tl, br) = b.latest_blob.bounding_box; MetaBlobSummary { id: b.id, x0: tl.x, y0: tl.y, x1: br.x, y1: br.y, state: format!("{:?}", b.state) } }).collect() };
                    let _ = bus.meta_tx.send(meta);
                }
            }
        }

        writer.write(&final_frame)?;
        // Pace processing to near real-time so the browser can keep up
        sleep(frame_delay).await;
        frame_index += 1;
    }

    println!("Processing complete. Output saved to {}", output_path);
    Ok(())
}

#[cfg(feature = "web")]
fn encode_jpeg(rgba_bytes: &[u8], width: u32, height: u32, quality: u8) -> Option<Vec<u8>> {
    use image::{codecs::jpeg::JpegEncoder, ColorType, DynamicImage, ImageBuffer, Rgba};
    // Wrap incoming RGBA buffer, then convert to RGB for JPEG encoding
    let rgba: ImageBuffer<Rgba<u8>, _> = ImageBuffer::from_raw(width, height, rgba_bytes.to_vec())?;
    let rgb = DynamicImage::ImageRgba8(rgba).to_rgb8();
    let (w, h) = rgb.dimensions();
    let mut out = Vec::new();
    let mut enc = JpegEncoder::new_with_quality(&mut out, quality);
    enc.encode(&rgb, w, h, ColorType::Rgb8.into()).ok()?;
    Some(out)
}

fn draw_header(frame: &mut Mat, frame_index: usize, analysis: &FrameAnalysis) {
    let header_height = 40;
    let rect = Rect::new(0, 0, frame.cols(), header_height);
    imgproc::rectangle(
        frame,
        rect,
        Scalar::new(0.0, 0.0, 0.0, 0.0),
        -1,
        imgproc::LINE_8,
        0,
    )
    .unwrap();

    let status_text = format!("{:?}", analysis.scene_state);
    let event_text = format!(
        "Frame: {} | Scene: {} | Events: {}",
        frame_index, status_text, analysis.significant_event_count
    );

    let text_pos = core::Point::new(10, 25);
    imgproc::put_text(
        frame,
        &event_text,
        text_pos,
        imgproc::FONT_HERSHEY_SIMPLEX,
        0.7,
        Scalar::new(255.0, 255.0, 255.0, 0.0),
        2,
        imgproc::LINE_AA,
        false,
    )
    .unwrap();
}

// Other helper functions remain the same...
fn draw_tracked_blobs(frame: &mut Mat, tracked_blobs: &[TrackedBlob], chunk_w: u32, chunk_h: u32) {
    for blob in tracked_blobs {
        let color = state_to_color(&blob.state);
        for point in &blob.latest_blob.chunk_coords {
            let top_left = core::Point::new(
                point.x as i32 * chunk_w as i32,
                point.y as i32 * chunk_h as i32,
            );
            let rect = Rect::new(top_left.x, top_left.y, chunk_w as i32, chunk_h as i32);
            let roi = Mat::roi(frame, rect).unwrap();
            let mut colored_roi = Mat::default();
            let color_mat =
                Mat::new_size_with_default(roi.size().unwrap(), roi.typ(), color).unwrap();
            core::add_weighted(&roi, 0.5, &color_mat, 0.5, 0.0, &mut colored_roi, -1).unwrap();
            colored_roi
                .copy_to(&mut Mat::roi(frame, rect).unwrap())
                .unwrap();
        }
        let (top_left_p, bottom_right_p) = blob.latest_blob.bounding_box;
        let top_left = core::Point::new(
            top_left_p.x as i32 * chunk_w as i32,
            top_left_p.y as i32 * chunk_h as i32,
        );
        let bottom_right = core::Point::new(
            (bottom_right_p.x + 1) as i32 * chunk_w as i32,
            (bottom_right_p.y + 1) as i32 * chunk_h as i32,
        );
        let rect = Rect::new(
            top_left.x,
            top_left.y,
            bottom_right.x - top_left.x,
            bottom_right.y - top_left.y,
        );
        imgproc::rectangle(frame, rect, color, 2, imgproc::LINE_8, 0).unwrap();
        let label = format!("ID: {} | S: {:?} | A: {}", blob.id, blob.state, blob.age);
        let text_pos = core::Point::new(rect.x, rect.y - 10);
        imgproc::put_text(
            frame,
            &label,
            text_pos,
            imgproc::FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
            imgproc::LINE_AA,
            false,
        )
        .unwrap();
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
                let roi = Mat::roi(frame, rect).unwrap();
                let mut dimmed_roi = Mat::default();
                core::multiply(&roi, &Scalar::all(0.4), &mut dimmed_roi, 1.0, -1).unwrap();
                dimmed_roi
                    .copy_to(&mut Mat::roi(frame, rect).unwrap())
                    .unwrap();
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
                imgproc::rectangle(
                    heatmap,
                    rect,
                    Scalar::new(b, g, r, 0.0),
                    -1,
                    imgproc::LINE_8,
                    0,
                )
                .unwrap();
            }
        }
    }
}
