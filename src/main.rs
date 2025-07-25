use axum::{
    extract::{Json, Request, State},
    middleware::{self, Next},
    response::{Html, Response as AxumResponse},
    routing::{get, post},
    Router,
};
use serde::{Deserialize, Serialize};
use std::{net::SocketAddr, sync::Arc, time::Instant};
use tokio::task;
use candle_transformers::models::bert::{BertModel};
use tokenizers::{Tokenizer};
use ndarray::{Array1, Array2};
use candle_core::Tensor;
use neurovlm::forward::{load_constants, text_query};

// Define your constants structure
struct AppConstants {
    model: BertModel,
    tokenizer: Tokenizer,
    mask: Array1<bool>,
    l_reg_fus: Array2<f32>,
    r_reg_fus: Array2<f32>,
    title_embeddings: Tensor,
    aligner_w0: Tensor,
    aligner_b0: Tensor,
    aligner_w1: Tensor,
    aligner_b1: Tensor,
    decoder_w0: Tensor,
    decoder_b0: Tensor,
    decoder_w1: Tensor,
    decoder_b1: Tensor,
    decoder_w2: Tensor,
    decoder_b2: Tensor
}

impl AppConstants {
    fn new() -> Self {
        println!("Loading constants...");
        let Ok((
            model, tokenizer, mask, l_reg_fus, r_reg_fus, title_embeddings,
            aligner_w0, aligner_b0, aligner_w1, aligner_b1,
            decoder_w0, decoder_b0, decoder_w1, decoder_b1, decoder_w2, decoder_b2
        )) = load_constants() else { todo!() };

        println!("Constants loaded successfully");

        Self {
            model,
            tokenizer,
            mask,
            l_reg_fus,
            r_reg_fus,
            title_embeddings,
            aligner_w0,
            aligner_b0,
            aligner_w1,
            aligner_b1,
            decoder_w0,
            decoder_b0,
            decoder_w1,
            decoder_b1,
            decoder_w2,
            decoder_b2
        }
    }
}

#[derive(Deserialize)]
struct Query {
    input: String,
}

#[derive(Serialize)]
struct Response {
    puborder: Vec<i32>,     // Publication order indices
    surface: Vec<f32>,      // Surface map data
    volume: Vec<f32>,       // Changed from Array3<f32> to Vec<f32>
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

async fn handle_query(
    State(constants): State<Arc<AppConstants>>,
    Json(payload): Json<Query>
) -> Json<Response> {
    println!("Received query: {}", payload.input);
    let start = Instant::now();

    // Clone the Arc to move into the blocking task
    let constants = Arc::clone(&constants);

    // Offload CPU-heavy task to blocking thread
    let result = task::spawn_blocking(move || {

        // Pass query and constants
        let (top_inds, surface_vec, img3d) = match text_query(
            &payload.input,
            &constants.model,
            &constants.tokenizer,
            &constants.title_embeddings,
            &constants.mask,
            &constants.l_reg_fus,
            &constants.r_reg_fus,
            &constants.aligner_w0,
            &constants.aligner_b0,
            &constants.aligner_w1,
            &constants.aligner_b1,
            &constants.decoder_w0,
            &constants.decoder_b0,
            &constants.decoder_w1,
            &constants.decoder_b1,
            &constants.decoder_w2,
            &constants.decoder_b2
        ) {
            Ok(vals) => vals,
            Err(e) => panic!("Query failed: {}", e),
        };

        // Convert Array3<f32> to Vec<f32> (Array3 doesn't implement Serialize)
        let volume_vec = img3d.into_raw_vec_and_offset().0;

        (top_inds, surface_vec, volume_vec)
    })
    .await
    .expect("Task panicked");

    let duration_ms = start.elapsed().as_millis();
    println!("Query processed in {}ms", duration_ms);

    Json(Response {
        puborder: result.0,
        surface: result.1,
        volume: result.2,
        duration_ms
    })
}

#[tokio::main]
async fn main() {
    // Load constants once at startup
    let constants = Arc::new(AppConstants::new());

    let app = Router::new()
        .route("/", get(serve_index))
        .route("/api/query", post(handle_query))
        .layer(middleware::from_fn(log_requests))
        .with_state(constants); // Pass constants to all handlers

    let addr = SocketAddr::from(([0, 0, 0, 0], 8081));
    println!("Listening on http://{}", addr);
    println!("Routes registered:");
    println!("  GET  /");
    println!("  POST /api/query");

    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}