mod routes;
mod utils;
mod logger;
mod config;
mod response;
mod error;
mod models;
mod middleware;
mod state;
mod repository;
mod handler;
mod service;
mod pipeline;

mod tracer;

use std::env;
use std::sync::Arc;
use anyhow::Error;
use axum::{
    routing::get,
    Router,
};
use log::info;
use opentelemetry::global;
use opentelemetry::global::shutdown_tracer_provider;
use tokio::signal;
use crate::logger::logger::setup_logger;
use config::settings::SETTINGS;
use crate::pipeline::general_pipeline::general_pipeline::GeneralPipeline;
use crate::pipeline::antispoofing_pipeline::antispoofing_pipeline::AntiSpoofingPipeline;
use crate::routes::root::{root_routes, RouterState};


#[cfg(not(target_env = "msvc"))]
use tikv_jemallocator::Jemalloc;
use crate::tracer::tracer::init_tracer_provider;

#[cfg(not(target_env = "msvc"))]
#[global_allocator]
static GLOBAL: Jemalloc = Jemalloc;

#[tokio::main]
async fn main() {
    // Setup logger
    setup_logger();
    let addr = format!("0.0.0.0:{}", SETTINGS.server.http_port);

    // Setup pipeline
    let general_pipeline = GeneralPipeline::new(
        SETTINGS.triton.faceid_host.as_str(),
        SETTINGS.triton.faceid_grpc_port.to_string().as_str(),
    )
        .await
        .unwrap_or_else(|e| panic!("Failed to init general pipeline client: {}", e.to_string()));

    let antispoofing_pipeline = AntiSpoofingPipeline::new(
        SETTINGS.triton.faceid_host.as_str(),
        SETTINGS.triton.faceid_grpc_port.to_string().as_str(),
    )
        .await
        .unwrap_or_else(|e| panic!("Failed to init anti-spoofing pipeline client: {}", e.to_string()));
    info!("completed initializing pipelines");

    // Setup tracing
    let tracer_provider = init_tracer_provider().expect("Failed to initialize tracer provider.");
    global::set_tracer_provider(tracer_provider.clone());

    // Init server
    let listener = tokio::net::TcpListener::bind(&addr)
        .await
        .unwrap_or_else(|e| panic!("Failed to create new listener: {}", e.to_string()));
    info!("starting api server on {:?}", addr);
    let router_state = RouterState::new(general_pipeline, antispoofing_pipeline);

    axum::serve(listener, root_routes(router_state))
        .with_graceful_shutdown(shutdown_signal())
        .await
        .unwrap_or_else(|e| panic!("Failed to start api server: {}", e.to_string()));

    shutdown_tracer_provider();
}

async fn shutdown_signal() {
    let ctrl_c = async {
        signal::ctrl_c()
            .await
            .expect("failed to install Ctrl+C handler");
    };

    #[cfg(unix)]
    let terminate = async {
        signal::unix::signal(signal::unix::SignalKind::terminate())
            .expect("failed to install signal handler")
            .recv()
            .await;
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => {},
        _ = terminate => {},
    }
}
