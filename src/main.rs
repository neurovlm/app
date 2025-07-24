use axum::{
    extract::{Json, Request},
    http::Method,
    middleware::{self, Next},
    response::{Html, Response as AxumResponse},
    routing::{get, post},
    Router,
};
use serde::{Deserialize, Serialize};
use std::{net::SocketAddr, time::Instant};
use tokio::task;
use rayon::prelude::*;
use crate::forward::{load_constants, text_query};

#[derive(Deserialize)]
struct Query {
    input: String,
}

#[derive(Serialize)]
struct Response {
    surface: Vec<f32>,     // Surface map data
    volume: Vec<f32>,      // Volume data
    puborder: Vec<u32>,    // Publication order indices
    duration_ms: u128,
}

// Logging middleware to see all requests
async fn log_requests(req: Request, next: Next) -> AxumResponse {
    let method = req.method().clone();
    let uri = req.uri().clone();
    println!("Incoming request: {} {}", method, uri);

    let response = next.run(req).await;
    println!("Response status: {}", response.status());
    response
}
// Serve your index.html page (you can expand to serve other static files)
async fn serve_index() -> Html<&'static str> {
    println!("Serving index page");
    Html(include_str!("../static/index.html"))
}

async fn handle_query(Json(payload): Json<Query>) -> Json<Response> {
    println!("Received query: {}", payload.input);
    let start = Instant::now();

    // Offload CPU-heavy task to blocking thread
    let result = task::spawn_blocking(move || {
        // TODO: Replace this with your actual neural network inference
        // For now, returning dummy data in the expected format

        // Generate dummy surface data (327684 values for fsaverage)
        let surface: Vec<f32> = (0..327684).map(|i| (i as f32 * 0.001) % 1.0).collect();

        // Generate dummy volume data (46*55*46 = 116,140 values)
        let volume: Vec<f32> = (0..116140).map(|i| (i as f32 * 0.0001) % 0.5 + 0.5).collect();

        // Generate dummy publication order (100 publications)
        let puborder: Vec<u32> = (0..100).collect();

        (surface, volume, puborder)
    })
    .await
    .expect("Task panicked");

    let duration_ms = start.elapsed().as_millis();
    println!("Query processed in {}ms", duration_ms);

    Json(Response {
        surface: result.0,
        volume: result.1,
        puborder: result.2,
        duration_ms
    })
}

#[tokio::main]
async fn main() {
    let app = Router::new()
        .route("/", get(serve_index))
        .route("/api/query", post(handle_query))
        .layer(middleware::from_fn(log_requests));

    let addr = SocketAddr::from(([0, 0, 0, 0], 8081));
    println!("Listening on http://{}", addr);
    println!("Routes registered:");
    println!("  GET  /");
    println!("  POST /api/query");

    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}