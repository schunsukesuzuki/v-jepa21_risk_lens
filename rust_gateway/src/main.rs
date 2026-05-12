// This file implements a lightweight Rust API gateway using Axum.
//
// The gateway exposes two health endpoints:
// - GET /health: checks whether the Rust gateway itself is alive.
// - GET /api/health: forwards the request to the backend service's /api/health endpoint.
//
// Tokio is used as the asynchronous runtime. Rust provides async/await syntax,
// but an async runtime is required to actually execute asynchronous tasks.
// Tokio supplies that runtime, including async TCP networking, task scheduling,
// and efficient non-blocking I/O handling.
use axum::{extract::State, http::StatusCode, response::IntoResponse, routing::get, Json, Router};
use serde_json::Value;
use std::{env, net::SocketAddr, sync::Arc};
use tower_http::cors::CorsLayer;

// Shared application state passed to Axum request handlers.
//
// backend_url stores the target backend service URL, such as http://backend:8000.
// Arc<String> allows the URL string to be cheaply and safely shared across
// multiple asynchronous request handlers without copying the whole string each time.
//
// reqwest::Client is an asynchronous HTTP client. It is stored in the shared state
// because reqwest clients are designed to be reused across requests.
#[derive(Clone)]
struct AppState {
    backend_url: Arc<String>,
    client: reqwest::Client,
}

// #[tokio::main] starts a Tokio async runtime and runs this async main function on it.
//
// This is necessary because the gateway uses async operations such as:
// - binding an async TCP listener,
// - serving HTTP requests with Axum,
// - sending asynchronous HTTP requests to the backend via reqwest.
//
// Without Tokio, the .await expressions in this function and in the handlers
// would not have an executor to drive them forward.
#[tokio::main]
async fn main() {
    // Initialize tracing-based logging so tracing::info! and related logs are emitted.
    tracing_subscriber::fmt::init();

    // Read the backend service URL from BACKEND_URL.
    // If the environment variable is not set, default to http://backend:8000,
    // which is a typical Docker Compose service URL when the backend service is named "backend".
    let backend_url = env::var("BACKEND_URL").unwrap_or_else(|_| "http://backend:8000".to_string());

    // Build the shared application state.
    // The backend URL and reusable HTTP client are made available to request handlers.
    let state = AppState { backend_url: Arc::new(backend_url), client: reqwest::Client::new() };

    // Define the Axum router.
    //
    // /health is handled locally by the Rust gateway.
    // /api/health is proxied to the backend service.
    //
    // with_state(state) attaches the shared AppState so handlers can extract it.
    // CorsLayer::permissive() allows broad cross-origin access, which is convenient
    // for development and demos, though production systems should usually restrict it.
    let app = Router::new()
        .route("/health", get(gateway_health))
        .route("/api/health", get(proxy_health))
        .with_state(state)
        .layer(CorsLayer::permissive());

    // Bind the server to 0.0.0.0:8080.
    // 0.0.0.0 means the server listens on all network interfaces,
    // which is useful when running inside Docker and exposing the port to the host.
    let addr = SocketAddr::from(([0, 0, 0, 0], 8080));
    tracing::info!("Rust gateway listening on {}", addr);

    // Create an asynchronous TCP listener using Tokio's networking API.
    // This does not block an operating-system thread while waiting for network events.
    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();

    // Start serving the Axum application on the listener.
    // This future runs until the server stops or an unrecoverable error occurs.
    axum::serve(listener, app).await.unwrap();
}

// Local health check for the Rust gateway itself.
//
// This endpoint does not call the backend. If this returns ok=true,
// it only proves that the gateway process is running and can respond to requests.
async fn gateway_health() -> impl IntoResponse {
    Json(serde_json::json!({"ok": true, "service": "vjepa-risk-gateway"}))
}

// Backend health proxy endpoint.
//
// This handler receives the shared AppState through Axum's State extractor,
// builds the backend /api/health URL, sends an async GET request via reqwest,
// and returns the backend's JSON response and status code when possible.
//
// If the backend cannot be reached, or if the backend returns a body that cannot
// be parsed as JSON, the gateway returns 502 Bad Gateway with an error message.
async fn proxy_health(State(state): State<AppState>) -> impl IntoResponse {
    // Build the backend health URL safely even when BACKEND_URL has a trailing slash.
    // For example, both http://backend:8000 and http://backend:8000/ become
    // http://backend:8000/api/health rather than producing a double slash.
    let url = format!("{}/api/health", state.backend_url.trim_end_matches('/'));

    // Send the request to the backend asynchronously.
    // While waiting for the backend response, Tokio can continue processing other tasks.
    match state.client.get(url).send().await {
        Ok(resp) => {
            // Preserve the backend's HTTP status code when forwarding the response.
            // If conversion somehow fails, return 502 because the gateway received
            // an invalid or unusable upstream response.
            let status = StatusCode::from_u16(resp.status().as_u16()).unwrap_or(StatusCode::BAD_GATEWAY);

            // Parse the backend response body as arbitrary JSON.
            // serde_json::Value is used because the exact JSON schema is not fixed here.
            match resp.json::<Value>().await {
                Ok(json) => (status, Json(json)).into_response(),
                Err(err) => (StatusCode::BAD_GATEWAY, Json(serde_json::json!({"error": err.to_string()}))).into_response(),
            }
        }
        Err(err) => (StatusCode::BAD_GATEWAY, Json(serde_json::json!({"error": err.to_string()}))).into_response(),
    }
}
