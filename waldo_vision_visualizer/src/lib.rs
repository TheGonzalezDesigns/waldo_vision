use std::sync::Arc;

use tokio::sync::broadcast;

#[derive(Debug, Clone, Copy)]
pub enum FrameFormat {
    Jpeg,
    Webp,
}

#[derive(Debug, Clone)]
pub struct FramePacket {
    pub ts_millis: u64,
    pub width: u32,
    pub height: u32,
    pub format: FrameFormat,
    pub data: Arc<[u8]>,
}

#[derive(Debug, Clone, Default)]
#[cfg_attr(feature = "web", derive(serde::Serialize, serde::Deserialize))]
pub struct MetaBlobSummary {
    pub id: u64,
    pub x0: u32,
    pub y0: u32,
    pub x1: u32,
    pub y1: u32,
    pub state: String,
}

#[derive(Debug, Clone, Default)]
#[cfg_attr(feature = "web", derive(serde::Serialize, serde::Deserialize))]
pub struct Meta {
    pub scene_state: String,
    pub event_count: u64,
    pub blobs: Vec<MetaBlobSummary>,
}

#[derive(Clone)]
pub struct FrameBus {
    pub frames_tx: broadcast::Sender<FramePacket>,
    pub meta_tx: broadcast::Sender<Meta>,
}

impl FrameBus {
    pub fn new(capacity: usize) -> Self {
        let (frames_tx, _) = broadcast::channel::<FramePacket>(capacity.max(1));
        let (meta_tx, _) = broadcast::channel::<Meta>(capacity.max(1));
        Self { frames_tx, meta_tx }
    }
}

#[derive(Debug, Clone)]
pub struct ServerConfig {
    pub bind_addr: String,
    pub nat_public_ip: Option<String>,
    pub udp_port_start: Option<u16>,
    pub udp_port_end: Option<u16>,
}

#[derive(Clone)]
pub struct ControlHandle {
    pub play_tx: tokio::sync::watch::Sender<bool>,
}

#[cfg(feature = "web")]
pub async fn start_server(
    bus: FrameBus,
    mut cfg: ServerConfig,
    control: ControlHandle,
) -> anyhow::Result<tokio::task::JoinHandle<()>> {
    use axum::extract::ws::{Message, WebSocket, WebSocketUpgrade};
    use axum::{Router, response::IntoResponse, routing::get};
    use leptos::*;
    use leptos_axum::LeptosRoutes;
    use serde::{Deserialize, Serialize};
    use webrtc::api::APIBuilder;
    use webrtc::ice_transport::ice_candidate::{RTCIceCandidate, RTCIceCandidateInit};
    use webrtc::ice_transport::ice_server::RTCIceServer;
    use webrtc::peer_connection::configuration::RTCConfiguration;
    use webrtc::peer_connection::sdp::session_description::RTCSessionDescription;
    // use webrtc::peer_connection::sdp::sdp_type::RTCSdpType;

    // Leptos SSR app with external client-side WebRTC code
    #[component]
    fn App() -> impl IntoView {
        view! {
            <main>
                <h2>Waldo Vision Visualizer</h2>
                <div style="margin: 8px 0; display:flex; gap:12px; align-items:center;">
                    <button id="btn-play" style="padding:6px 12px;">Play</button>
                    <button id="btn-pause" style="padding:6px 12px;">Pause</button>
                    <span id="status" style="font-family:monospace; font-size:12px; color:#777">idle</span>
                </div>
                <canvas id="preview" width="1280" height="720" style="border:1px solid #444"></canvas>
                <script src="/client.js"></script>
            </main>
        }
    }

    // Serve a small client script for WebRTC DataChannels
    const CLIENT_JS: &str = r#"(function(){
        const status = (t)=>{ const el=document.getElementById('status'); if(el) el.textContent=t; };
        const btnPlay = document.getElementById('btn-play');
        const btnPause = document.getElementById('btn-pause');
        if(btnPlay){ btnPlay.onclick = ()=> fetch('/control/play', { method:'POST' }).then(()=>status('playing')); }
        if(btnPause){ btnPause.onclick = ()=> fetch('/control/pause', { method:'POST' }).then(()=>status('paused')); }

        const params = new URLSearchParams(location.search);
        const relayOnly = params.get('relay') === '1' || true; // default to relay for reliability
        const TURN_HOST = (params.get('turn') || (location.hostname));
        const TURN_USER = (params.get('tu') || 'waldo');
        const TURN_PASS = (params.get('tp') || 'SuperSecretPassword');
        const iceServers = [
            { urls: ['stun:stun.l.google.com:19302'] },
            { urls: [`turn:${TURN_HOST}:3478?transport=udp`], username: TURN_USER, credential: TURN_PASS },
            { urls: [`turn:${TURN_HOST}:3478?transport=tcp`], username: TURN_USER, credential: TURN_PASS },
        ];
        const pc = new RTCPeerConnection({ iceServers, iceTransportPolicy: relayOnly ? 'relay' : 'all' });
        const frames = pc.createDataChannel('frames', {ordered:false, maxRetransmits:0});
        const meta = pc.createDataChannel('meta', {ordered:true});
        const canvas = document.getElementById('preview');
        const ctx = canvas.getContext('2d');
        pc.oniceconnectionstatechange = ()=> status('ice: ' + pc.iceConnectionState);
        pc.onconnectionstatechange = ()=> status('pc: ' + pc.connectionState);
        frames.binaryType='arraybuffer';
        frames.onopen = ()=> status('frames: open');
        frames.onclose = ()=> status('frames: closed');
        meta.onopen = ()=> status('meta: open');
        meta.onclose = ()=> status('meta: closed');
        frames.onmessage = async (ev)=>{
            if(!(ev.data instanceof ArrayBuffer)) return;
            const blob = new Blob([ev.data], {type:'image/jpeg'});
            const bmp = await createImageBitmap(blob);
            ctx.drawImage(bmp, 0, 0, canvas.width, canvas.height);
        };
        meta.onmessage = (ev)=>{
            try {
              const o = JSON.parse(ev.data);
              if(o && o.type==='server_info') status('server: nat_ip='+o.nat_ip+' ice='+o.ice_range);
            } catch(_) { /* ignore non-json meta */ }
        };

        // Setup signaling with buffering until WS opens
        const ws = new WebSocket((location.protocol==='https:'?'wss://':'ws://')+location.host+'/ws/signaling');
        let wsOpen = false;
        const iceQueue = [];
        ws.addEventListener('open', async ()=>{
            wsOpen = true;
            // Create and send offer once WS is ready
            const offer = await pc.createOffer({});
            await pc.setLocalDescription(offer);
            ws.send(JSON.stringify({type:'offer', sdp: offer.sdp}));
            // Flush queued ICE
            while(iceQueue.length){ ws.send(JSON.stringify({type:'ice', candidate: iceQueue.shift()})); }
        });
        ws.addEventListener('close', ()=> status('ws closed'));
        ws.addEventListener('error', ()=> status('ws error'));

        pc.onicecandidate = (ev)=>{
            if(!ev.candidate) return;
            if(wsOpen) {
                try { ws.send(JSON.stringify({type:'ice', candidate: ev.candidate})); } catch(e) { iceQueue.push(ev.candidate); }
            } else {
                iceQueue.push(ev.candidate);
            }
        };

        ws.onmessage = async (ev)=>{
            const msg = JSON.parse(ev.data);
            if(msg.type==='answer'){
                await pc.setRemoteDescription({type:'answer', sdp: msg.sdp});
            } else if(msg.type==='ice'){
                try{ await pc.addIceCandidate(msg.candidate); }catch(e){ console.warn(e); }
            }
        };
    })();"#;

    #[derive(Debug, Serialize, Deserialize)]
    #[serde(tag = "type")]
    enum SigMsg {
        #[serde(rename = "offer")]
        Offer { sdp: String },
        #[serde(rename = "answer")]
        Answer { sdp: String },
        #[serde(rename = "ice")]
        Ice { candidate: RTCIceCandidateInit },
    }

    fn ws_handler_with_bus(
        ws: WebSocketUpgrade,
        bus: FrameBus,
        cfg: ServerConfig,
        control: ControlHandle,
    ) -> impl IntoResponse {
        ws.on_upgrade(move |socket| ws_conn(socket, bus, cfg, control))
    }

    async fn ws_conn(
        mut socket: WebSocket,
        bus: FrameBus,
        cfg: ServerConfig,
        control: ControlHandle,
    ) {
        // Build a PeerConnection with optional NAT 1:1 IP if provided
        use webrtc::api::setting_engine::SettingEngine;
        use webrtc::ice_transport::ice_candidate_type::RTCIceCandidateType;
        let mut se = SettingEngine::default();
        if let Some(ip) = cfg.nat_public_ip.clone() {
            se.set_nat_1to1_ips(vec![ip], RTCIceCandidateType::Host);
        }
        let api = APIBuilder::new().with_setting_engine(se).build();
        // Configure STUN + TURN; TURN host/user/pass from env, fallback to public IP on 3478
        let turn_host = std::env::var("WV_TURN_HOST")
            .unwrap_or_else(|_| cfg.nat_public_ip.clone().unwrap_or_else(|| "".into()));
        let turn_user = std::env::var("WV_TURN_USER").unwrap_or_else(|_| "waldo".into());
        let turn_pass =
            std::env::var("WV_TURN_PASS").unwrap_or_else(|_| "SuperSecretPassword".into());
        let mut ice_servers = vec![RTCIceServer {
            urls: vec!["stun:stun.l.google.com:19302".to_string()],
            ..Default::default()
        }];
        if !turn_host.is_empty() {
            ice_servers.push(RTCIceServer {
                urls: vec![format!("turn:{}:3478?transport=udp", turn_host)],
                username: turn_user.clone(),
                credential: turn_pass.clone(),
                ..Default::default()
            });
            ice_servers.push(RTCIceServer {
                urls: vec![format!("turn:{}:3478?transport=tcp", turn_host)],
                username: turn_user.clone(),
                credential: turn_pass.clone(),
                ..Default::default()
            });
        }
        let config = RTCConfiguration {
            ice_servers,
            ..Default::default()
        };
        let pc = match api.new_peer_connection(config).await {
            Ok(pc) => pc,
            Err(e) => {
                let _=socket.send(Message::Text(serde_json::json!({"type":"error","message":format!("peer_connection: {}", e)}).to_string())).await;
                return;
            }
        };

        // Hook DataChannels created by the browser and forward frames/meta
        let frames_tx = bus.frames_tx.clone();
        let meta_tx = bus.meta_tx.clone();

        let play_tx_for_frames = control.play_tx.clone();
        let nat_ip_str = cfg
            .nat_public_ip
            .clone()
            .unwrap_or_else(|| std::env::var("WV_NAT_IP").unwrap_or_else(|_| "unset".into()));
        let ice_range_str = if let (Some(s), Some(e)) = (cfg.udp_port_start, cfg.udp_port_end) {
            format!("{}-{}", s, e)
        } else {
            std::env::var("WV_ICE_PORT_RANGE").unwrap_or_else(|_| "unset".into())
        };

        pc.on_data_channel(Box::new(move |dc| {
            let label = dc.label().to_string();
            if label == "frames" {
                let mut rx = frames_tx.subscribe();
                let tx_clone = play_tx_for_frames.clone();
                Box::pin(async move {
                    // When frames DataChannel opens, auto-play so streaming starts only after peer is ready
                    dc.on_open(Box::new(move || {
                        let tx2 = tx_clone.clone();
                        Box::pin(async move {
                            let _ = tx2.send(true);
                        })
                    }));
                    loop {
                        match rx.recv().await {
                            Ok(pkt) => {
                                // Send as binary (JPEG/WebP) blob
                                let _ = dc
                                    .send(&bytes::Bytes::from(pkt.data.as_ref().to_vec()))
                                    .await;
                            }
                            Err(tokio::sync::broadcast::error::RecvError::Lagged(_)) => {
                                continue;
                            }
                            Err(_) => break,
                        }
                    }
                })
            } else if label == "meta" {
                let mut rxm = meta_tx.subscribe();
                // Prepare server info message and use a clone of the data channel for the on_open callback
                let info_nat = nat_ip_str.clone();
                let info_range = ice_range_str.clone();
                let dc_for_open = dc.clone();
                Box::pin(async move {
                    let info = format!(
                        "{{\"type\":\"server_info\",\"nat_ip\":\"{}\",\"ice_range\":\"{}\"}}",
                        info_nat, info_range
                    );
                    let msg = info.clone();
                    dc.on_open(Box::new(move || {
                        let msg2 = msg.clone();
                        let dc2 = dc_for_open.clone();
                        Box::pin(async move {
                            let _ = dc2.send_text(msg2).await;
                        })
                    }));
                    loop {
                        match rxm.recv().await {
                            Ok(meta) => {
                                if let Ok(s) = serde_json::to_string(&meta) {
                                    let _ = dc.send_text(s).await;
                                }
                            }
                            Err(tokio::sync::broadcast::error::RecvError::Lagged(_)) => {
                                continue;
                            }
                            Err(_) => break,
                        }
                    }
                })
            } else {
                Box::pin(async {})
            }
        }));

        // Split socket so ICE can be sent from callback
        use futures_util::{SinkExt, StreamExt};
        let (ws_tx, mut ws_rx) = socket.split();
        let ws_tx_arc = std::sync::Arc::new(tokio::sync::Mutex::new(ws_tx));
        // Trickle ICE to client
        let ws_tx_clone = ws_tx_arc.clone();
        pc.on_ice_candidate(Box::new(move |cand: Option<RTCIceCandidate>| {
            let ws_tx_clone = ws_tx_clone.clone();
            Box::pin(async move {
                if let Some(c) = cand {
                    if let Ok(init) = c.to_json() {
                        if let Ok(txt) = serde_json::to_string(&SigMsg::Ice { candidate: init }) {
                            let _ = ws_tx_clone.lock().await.send(Message::Text(txt)).await;
                        }
                    }
                }
            })
        }));

        // Handle signaling exchange
        while let Some(Ok(msg)) = ws_rx.next().await {
            match msg {
                Message::Text(txt) => {
                    if let Ok(sig) = serde_json::from_str::<SigMsg>(&txt) {
                        match sig {
                            SigMsg::Offer { sdp } => {
                                let offer = RTCSessionDescription::offer(sdp).unwrap_or_default();
                                if let Err(e) = pc.set_remote_description(offer).await {
                                    let _=ws_tx_arc.lock().await.send(Message::Text(serde_json::json!({"type":"error","message":format!("set_remote_description: {}", e)}).to_string())).await;
                                    return;
                                }
                                let answer = match pc.create_answer(None).await {
                                    Ok(a) => a,
                                    Err(e) => {
                                        let _=ws_tx_arc.lock().await.send(Message::Text(serde_json::json!({"type":"error","message":format!("create_answer: {}", e)}).to_string())).await;
                                        return;
                                    }
                                };
                                if let Err(e) = pc.set_local_description(answer.clone()).await {
                                    let _=ws_tx_arc.lock().await.send(Message::Text(serde_json::json!({"type":"error","message":format!("set_local_description: {}", e)}).to_string())).await;
                                    return;
                                }
                                let _ = ws_tx_arc
                                    .lock()
                                    .await
                                    .send(Message::Text(
                                        serde_json::to_string(&SigMsg::Answer { sdp: answer.sdp })
                                            .unwrap(),
                                    ))
                                    .await;
                            }
                            SigMsg::Ice { candidate } => {
                                let _ = pc.add_ice_candidate(candidate).await;
                            }
                            _ => {}
                        }
                    }
                }
                Message::Close(_) => break,
                _ => {}
            }
        }
        let _ = pc.close().await;
    }

    let leptos_options = LeptosOptions::builder()
        .output_name("waldo_vision_visualizer")
        .site_root(".")
        .build();
    use axum::extract::{Path, Query};
    use axum::response::Html;
    use axum::{response::IntoResponse as _, routing::post};
    use std::collections::HashMap;
    use tokio::sync::mpsc;
    const INDEX_HTML: &str = r#"<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>Waldo Vision Visualizer</title>
  <style>
    body{background:#111;color:#ddd;font-family:system-ui, sans-serif;margin:0;padding:16px}
    .bar{margin:8px 0;display:flex;gap:12px;align-items:center}
    button{padding:6px 12px;background:#333;border:1px solid #555;color:#fff;border-radius:6px;cursor:pointer}
    canvas{border:1px solid #444;max-width:100%;height:auto}
    #status{font:12px monospace;color:#999}
    details{margin-top:12px}
    #diag-log{background:#0b0b0b;border:1px solid #333;padding:8px;border-radius:6px;max-height:280px;overflow:auto;font:12px/1.4 monospace}
  </style>
  </head>
  <body>
   <h2>Waldo Vision Visualizer</h2>
   <div class=\"bar\">
     <button id=\"btn-play\">Play</button>
     <button id=\"btn-pause\">Pause</button>
     <span id=\"status\">idle</span>
   </div>
   <canvas id=\"preview\" width=\"1280\" height=\"720\"></canvas>
   <details open>
     <summary>Diagnostics</summary>
     <div id=\"diag\">
       <div style=\"margin:6px 0;\">NAT/ICE reported by server will appear here.</div>
       <pre id=\"diag-log\"></pre>
     </div>
   </details>
   <script src=\"/client.js\"></script>
  </body>
</html>"#;

    let mut base = Router::new()
        .route("/", get(|| async { Html(INDEX_HTML) }))
        .route("/healthz", get(|| async { "ok" }))
        .leptos_routes(&leptos_options, Vec::new(), || view! { <App/> })
        .with_state(leptos_options);

    // Merge env into cfg if present
    // Read env for future use (NAT / ICE port range)
    if cfg.nat_public_ip.is_none() {
        if let Ok(ip) = std::env::var("WV_NAT_IP") {
            if !ip.is_empty() {
                cfg.nat_public_ip = Some(ip);
            }
        }
    }
    if cfg.udp_port_start.is_none() || cfg.udp_port_end.is_none() {
        if let Ok(r) = std::env::var("WV_ICE_PORT_RANGE") {
            let parts: Vec<&str> = r.split('-').collect();
            if parts.len() == 2 {
                if let (Ok(s), Ok(e)) = (parts[0].parse::<u16>(), parts[1].parse::<u16>()) {
                    cfg.udp_port_start = Some(s);
                    cfg.udp_port_end = Some(e);
                }
            }
        }
    }

    let bus_ws = bus.clone();
    let cfg_ws = cfg.clone();
    let control_ws = control.clone();
    // Ingest channel: single-source ("pi"). Frames received here will be processed centrally.
    let (ingest_tx, ingest_rx) = mpsc::channel::<Vec<u8>>(64);

    // Spawn ingest worker: decode JPEG -> run VisionPipeline -> draw overlays -> publish to FrameBus.
    tokio::spawn(ingest_worker(ingest_rx, bus.clone()));

    base = base
        .route("/ws/signaling", get(move |ws: WebSocketUpgrade| {
            let bus = bus_ws.clone();
            let cfg = cfg_ws.clone();
            let control = control_ws.clone();
            async move { ws_handler_with_bus(ws, bus, cfg, control) }
        }))
        .route("/ws/ingest/pi", get({
            let ingest_tx = ingest_tx.clone();
            move |ws: WebSocketUpgrade| async move {
                ws.on_upgrade(move |mut socket| async move {
                    use axum::extract::ws::Message;
                    println!("ingest: WS connected (source=pi)");
                    let mut saw_first = false;
                    loop {
                        match socket.recv().await {
                            Some(Ok(Message::Binary(b))) => {
                                if !saw_first { println!("ingest: first frame ({} bytes)", b.len()); saw_first = true; }
                                if let Err(_e) = ingest_tx.send(b).await {
                                    println!("ingest: channel closed while forwarding frame");
                                    break;
                                }
                            }
                            Some(Ok(Message::Text(t))) => {
                                // Optional hello JSON; log and ignore
                                let preview = if t.len() > 200 { format!("{}...", &t[..200]) } else { t };
                                println!("ingest: text msg: {}", preview);
                            }
                            Some(Ok(Message::Close(cf))) => {
                                if let Some(frame) = cf { println!("ingest: WS close code={} reason={}", frame.code, frame.reason); }
                                else { println!("ingest: WS closed by peer"); }
                                break;
                            }
                            Some(Ok(_other)) => { /* ping/pong ignored */ }
                            Some(Err(e)) => {
                                println!("ingest: WS error: {}", e);
                                break;
                            }
                            None => {
                                println!("ingest: WS disconnected");
                                break;
                            }
                        }
                    }
                })
            }
        }))
        .route("/stream.mjpeg", get({
            let bus_stream = bus.clone();
            let control_stream = control.clone();
            move || async move {
                use axum::response::IntoResponse;
                use axum::http::{HeaderMap, HeaderValue};
                use bytes::Bytes;
                use async_stream::stream;
                let boundary = "frame";
                let mut headers = HeaderMap::new();
                headers.insert(
                    axum::http::header::CONTENT_TYPE,
                    HeaderValue::from_static("multipart/x-mixed-replace;boundary=frame"),
                );
                // Auto-start processing when MJPEG stream is requested
                let _ = control_stream.play_tx.send(true);
                let mut rx = bus_stream.frames_tx.subscribe();
                let s = stream! {
                    loop {
                        match rx.recv().await {
                            Ok(pkt) => {
                                println!("mjpeg: streaming frame {} bytes ({}x{})", pkt.data.len(), pkt.width, pkt.height);
                                let header = format!("--{}\r\nContent-Type: image/jpeg\r\nContent-Length: {}\r\n\r\n", boundary, pkt.data.len());
                                yield Ok::<Bytes, axum::Error>(Bytes::from(header));
                                yield Ok::<Bytes, axum::Error>(Bytes::from(pkt.data.to_vec()));
                                yield Ok::<Bytes, axum::Error>(Bytes::from("\r\n"));
                            }
                            Err(_) => { break; }
                        }
                    }
                };
                (headers, axum::body::Body::from_stream(s)).into_response()
            }
        }))
        .route("/client.js", get(|| async {
            let mut resp = axum::response::Response::new(axum::body::Body::from(CLIENT_JS));
            resp.headers_mut().insert(axum::http::header::CONTENT_TYPE, axum::http::HeaderValue::from_static("application/javascript"));
            resp
        }))
        .route("/control/play", post({
            let tx = control.play_tx.clone();
            move || {
                let tx = tx.clone();
                async move { let _ = tx.send(true); axum::http::StatusCode::NO_CONTENT }
            }
        }))
        .route("/control/pause", post({
            let tx = control.play_tx.clone();
            move || {
                let tx = tx.clone();
                async move { let _ = tx.send(false); axum::http::StatusCode::NO_CONTENT }
            }
        }));

    let app = base;

    let bind_addr = cfg.bind_addr.clone();
    let server = tokio::spawn(async move {
        let listener = tokio::net::TcpListener::bind(&bind_addr)
            .await
            .expect("bind addr");
        println!(
            "Visualizer server listening on http://{} (NAT_IP={}, UDP_RANGE={}..{})",
            bind_addr,
            std::env::var("WV_NAT_IP").unwrap_or_else(|_| "unset".into()),
            std::env::var("WV_ICE_PORT_RANGE").unwrap_or_else(|_| "unset".into()),
            ""
        );
        axum::serve(listener, app).await.expect("server run");
    });

    Ok(server)
}

#[cfg(not(feature = "web"))]
pub async fn start_server(
    _bus: FrameBus,
    _cfg: ServerConfig,
) -> anyhow::Result<tokio::task::JoinHandle<()>> {
    Err(anyhow::anyhow!(
        "web feature not enabled for waldo_vision_visualizer"
    ))
}

// Ingest worker: decode JPEG -> run waldo_vision pipeline centrally -> draw overlays -> publish to FrameBus
#[cfg(feature = "web")]
async fn ingest_worker(mut rx: tokio::sync::mpsc::Receiver<Vec<u8>>, bus: FrameBus) {
    use image::{ColorType, DynamicImage};
    use waldo_vision::pipeline::{PipelineConfig, VisionPipeline};

    // Initialize when first frame arrives (dimensions unknown beforehand)
    let mut pipeline: Option<(VisionPipeline, u32, u32, u32, u32)> = None; // (pipe, w, h, chunk_w, chunk_h)

    while let Some(jpeg) = rx.recv().await {
        if jpeg.is_empty() {
            println!("ingest_worker: received empty frame; skipping");
            continue;
        }
        let img = match image::load_from_memory(&jpeg) {
            Ok(i) => i,
            Err(e) => {
                println!(
                    "ingest_worker: JPEG decode error: {} ({} bytes)",
                    e,
                    jpeg.len()
                );
                continue;
            }
        };
        let rgba = img.to_rgba8();
        let (w, h) = rgba.dimensions();
        if pipeline.is_none() {
            let cfg = PipelineConfig {
                image_width: w,
                image_height: h,
                chunk_width: 10,
                chunk_height: 10,
                new_age_threshold: 5,
                behavioral_anomaly_threshold: 3.0,
                absolute_min_blob_size: 2,
                blob_size_std_dev_filter: 2.0,
                disturbance_entry_threshold: 0.25,
                disturbance_exit_threshold: 0.15,
                disturbance_confirmation_frames: 5,
            };
            println!(
                "ingest_worker: initializing pipeline w={} h={} chunk={}x{}",
                w, h, cfg.chunk_width, cfg.chunk_height
            );
            pipeline = Some((
                VisionPipeline::new(cfg.clone()),
                w,
                h,
                cfg.chunk_width,
                cfg.chunk_height,
            ));
        }
        let (pipe, _pw, _ph, chunk_w, chunk_h) = pipeline.as_mut().unwrap();
        let analysis = pipe.process_frame(rgba.as_raw());
        let overlaid = draw_overlays(rgba, &analysis, *chunk_w, *chunk_h);

        // JPEG does not support alpha; convert to RGB before encoding
        let rgb = DynamicImage::ImageRgba8(overlaid).to_rgb8();
        let (w, h) = rgb.dimensions();
        let mut out = Vec::new();
        let mut enc = image::codecs::jpeg::JpegEncoder::new_with_quality(&mut out, 75);
        match enc.encode(&rgb, w, h, ColorType::Rgb8.into()) {
            Ok(_) => {
                println!("ingest_worker: published {} bytes ({}x{})", out.len(), w, h);
                let _ = bus.frames_tx.send(FramePacket {
                    ts_millis: std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .map(|d| d.as_millis() as u64)
                        .unwrap_or(0),
                    width: w,
                    height: h,
                    format: FrameFormat::Jpeg,
                    data: out.into(),
                });
            }
            Err(e) => {
                println!("ingest_worker: JPEG encode error: {}", e);
            }
        }
    }
}

#[cfg(feature = "web")]
fn draw_overlays(
    mut rgba: image::RgbaImage,
    analysis: &waldo_vision::pipeline::FrameAnalysis,
    chunk_w: u32,
    chunk_h: u32,
) -> image::RgbaImage {
    use waldo_vision::pipeline::TrackedState;
    // Draw simple hollow rectangles for tracked blobs; do not change existing drawing styles beyond this minimal overlay
    for tb in &analysis.tracked_blobs {
        let (tl, br) = tb.latest_blob.bounding_box;
        let mut x0 = (tl.x * chunk_w) as i32;
        let mut y0 = (tl.y * chunk_h) as i32;
        let mut x1 = ((br.x + 1) * chunk_w) as i32 - 1;
        let mut y1 = ((br.y + 1) * chunk_h) as i32 - 1;
        if x1 <= x0 || y1 <= y0 {
            continue;
        }
        let (iw, ih) = rgba.dimensions();
        x0 = x0.clamp(0, iw.saturating_sub(1) as i32);
        y0 = y0.clamp(0, ih.saturating_sub(1) as i32);
        x1 = x1.clamp(0, iw.saturating_sub(1) as i32);
        y1 = y1.clamp(0, ih.saturating_sub(1) as i32);

        let color = match tb.state {
            TrackedState::New => image::Rgba([0, 255, 255, 255]),
            TrackedState::Tracking => image::Rgba([255, 100, 0, 255]),
            TrackedState::Lost => image::Rgba([128, 128, 128, 255]),
            TrackedState::Anomalous => image::Rgba([0, 0, 255, 255]),
        };
        for x in x0..=x1 {
            if y0 >= 0 {
                rgba.put_pixel(x as u32, y0 as u32, color);
            }
            if y1 >= 0 {
                rgba.put_pixel(x as u32, y1 as u32, color);
            }
        }
        for y in y0..=y1 {
            if x0 >= 0 {
                rgba.put_pixel(x0 as u32, y as u32, color);
            }
            if x1 >= 0 {
                rgba.put_pixel(x1 as u32, y as u32, color);
            }
        }
    }
    rgba
}
