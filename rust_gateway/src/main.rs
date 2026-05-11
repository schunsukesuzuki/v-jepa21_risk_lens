use axum::{extract::State, http::StatusCode, response::IntoResponse, routing::get, Json, Router};
use serde_json::Value;
use std::{env, net::SocketAddr, sync::Arc};
use tower_http::cors::CorsLayer;

#[derive(Clone)]
struct AppState {
    backend_url: Arc<String>,
    client: reqwest::Client,
}

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt::init();
    let backend_url = env::var("BACKEND_URL").unwrap_or_else(|_| "http://backend:8000".to_string());
    let state = AppState { backend_url: Arc::new(backend_url), client: reqwest::Client::new() };
    let app = Router::new()
        .route("/health", get(gateway_health))
        .route("/api/health", get(proxy_health))
        .with_state(state)
        .layer(CorsLayer::permissive());
    let addr = SocketAddr::from(([0, 0, 0, 0], 8080));
    tracing::info!("Rust gateway listening on {}", addr);
    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}

async fn gateway_health() -> impl IntoResponse {
    Json(serde_json::json!({"ok": true, "service": "vjepa-risk-gateway"}))
}

async fn proxy_health(State(state): State<AppState>) -> impl IntoResponse {
    let url = format!("{}/api/health", state.backend_url.trim_end_matches('/'));
    match state.client.get(url).send().await {
        Ok(resp) => {
            let status = StatusCode::from_u16(resp.status().as_u16()).unwrap_or(StatusCode::BAD_GATEWAY);
            match resp.json::<Value>().await {
                Ok(json) => (status, Json(json)).into_response(),
                Err(err) => (StatusCode::BAD_GATEWAY, Json(serde_json::json!({"error": err.to_string()}))).into_response(),
            }
        }
        Err(err) => (StatusCode::BAD_GATEWAY, Json(serde_json::json!({"error": err.to_string()}))).into_response(),
    }
}
