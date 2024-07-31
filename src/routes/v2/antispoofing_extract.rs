use axum::extract::DefaultBodyLimit;
use axum::Router;
use axum::routing::post;
use tower_http::limit::RequestBodyLimitLayer;
use crate::handler::antispoofing_handler::antispoofing_extract;
use crate::state::antispoofing_state::AntiSpoofingState;

pub fn new_antispoofing_extract_route() -> Router<AntiSpoofingState> {

    let router = Router::new()
        .route("/anti-spoofing", post(antispoofing_extract))
        .layer(DefaultBodyLimit::disable())
        .layer(RequestBodyLimitLayer::new(
            250 * 1024 * 1024, /* 250mb */
        ));
    router
}