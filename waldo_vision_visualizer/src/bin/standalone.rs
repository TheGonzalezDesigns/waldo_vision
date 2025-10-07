use waldo_vision_visualizer::{ControlHandle, FrameBus, ServerConfig, start_server};

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
    // Control handle is only used when the `web` feature is enabled
    #[cfg(feature = "web")]
    let control = ControlHandle { play_tx };
    #[cfg(not(feature = "web"))]
    let _control = ControlHandle { play_tx };

    // Call the appropriate start_server signature based on feature flag
    #[cfg(feature = "web")]
    let handle = start_server(bus, cfg, control).await?;
    #[cfg(not(feature = "web"))]
    let handle = start_server(bus, cfg).await?;
    // Park forever
    handle.await.ok();
    Ok(())
}
