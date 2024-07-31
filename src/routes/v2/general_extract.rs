use axum::extract::DefaultBodyLimit;
use axum::Router;
use axum::routing::post;
use crate::state::general_state::GeneralState;
use tower_http::limit::RequestBodyLimitLayer;
use crate::handler::general_handler::general_extract;

pub fn new_general_extract_route() -> Router<GeneralState> {

    let router = Router::new()
        .route("/general", post(general_extract))
        .layer(DefaultBodyLimit::disable())
        .layer(RequestBodyLimitLayer::new(
            250 * 1024 * 1024, /* 250mb */
        ));
    router
}