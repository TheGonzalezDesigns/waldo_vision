use waldo_vision_visualizer::{start_server, ControlHandle, FrameBus, ServerConfig};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Bind address from env or default
    let bind = std::env::var("WV_BIND").unwrap_or_else(|_| "127.0.0.1:3001".to_string());

    let bus = FrameBus::new(2);
    let cfg = ServerConfig {
        bind_addr: bind,
        nat_public_ip: std::env::var("WV_NAT_IP").ok(),
        udp_port_start: None,
        udp_port_end: None,
    };
    let (play_tx, mut _play_rx) = tokio::sync::watch::channel(false);
    let control = ControlHandle { play_tx };

    let handle = start_server(bus, cfg, control).await?;
    // Park forever
    handle.await.ok();
    Ok(())
}

