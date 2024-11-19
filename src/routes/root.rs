use std::sync::Arc;
use std::time::Duration;

use axum::{Json, middleware, Router, ServiceExt};
use axum::http::header;
use axum::routing::{get, IntoMakeService};
use http::{HeaderMap, Method, StatusCode, Uri};
use serde::{Deserialize, Serialize};
use tonic_build::Service;
use tower_http::compression::CompressionLayer;
use tower_http::cors::CorsLayer;
use tower_http::propagate_header::PropagateHeaderLayer;
use tower_http::sensitive_headers::SetSensitiveHeadersLayer;
use tower_http::timeout::TimeoutLayer;
use tower_request_id::RequestIdLayer;
use crate::config::settings::SETTINGS;
use crate::error::errors::ResponseCode;
use crate::middleware::api_key_mw::validate_api_key_mw;
use crate::middleware::request_id_mw::generate_request_id_mw;
use crate::models::antispoofing_model::AntiSpoofingExtractionResultOutput;
use crate::pipeline::general_pipeline::general_pipeline::GeneralPipeline;
use crate::pipeline::antispoofing_pipeline::antispoofing_pipeline::AntiSpoofingPipeline;
use crate::response::common_response::{BaseResponse, GeneralResponseBuilder, GeneralResponseResult};
use crate::routes::v2::general_extract::new_general_extract_route;
use crate::routes::v2::antispoofing_extract::new_antispoofing_extract_route;
use crate::state::general_state::GeneralState;
use crate::state::antispoofing_state::AntiSpoofingState;

#[derive(Clone, Serialize, Deserialize)]
struct FallbackResponse {
    message: String,
}


#[derive(Clone)]
pub struct RouterState {
    general_pipeline: Arc<GeneralPipeline>,
    // antispoofing_pipeline: Arc<AntiSpoofingPipeline>,
}

impl RouterState {
    pub fn new(general_pipeline: GeneralPipeline) -> Self { // , antispoofing_pipeline: AntiSpoofingPipeline
         RouterState {
             general_pipeline: Arc::new(general_pipeline),
             // antispoofing_pipeline: Arc::new(antispoofing_pipeline),
        }

    }
}

pub fn root_routes(router_state: RouterState) -> IntoMakeService<Router> {

    let v2_router = {
        let general_state = GeneralState::new(&router_state.general_pipeline);
        let general_route = new_general_extract_route()
            .with_state(general_state);

        // let antispoofing_state = AntiSpoofingState::new(&router_state.antispoofing_pipeline);
        // let antispoofing_route = new_antispoofing_extract_route()
        //     .with_state(antispoofing_state);


        Router::new()
            .nest(
                "/v2",
                Router::new().nest(
                    "/extract",
                    Router::new()
                        .merge(general_route)
                        // .merge(antispoofing_route)


                )
            )
    };

    let mut request_timeout_duration: u64 = 20;
    if let Some(_request_timeout) = SETTINGS.server.request_timeout {
        request_timeout_duration = _request_timeout;
    }

    let app_router = Router::new()
        .nest(
            "/api",
            Router::new()
                .merge(Router::new().route("/health", get(healthcheck)))
                .merge(v2_router)
                .layer(CompressionLayer::new())
                .layer(middleware::from_fn(validate_api_key_mw)),
        )
        .layer(PropagateHeaderLayer::new(header::HeaderName::from_static("x-request-id")))
        .layer(CorsLayer::permissive().allow_methods([Method::GET, Method::POST, Method::HEAD, Method::OPTIONS]))
        .layer(RequestIdLayer)
        .layer(middleware::from_fn(generate_request_id_mw))
        .layer(TimeoutLayer::new(Duration::from_secs(request_timeout_duration)))
        .layer(SetSensitiveHeadersLayer::new(std::iter::once(header::AUTHORIZATION)))
        .fallback(fallback)
        .into_make_service();
    app_router
}

async fn fallback(uri: Uri) -> (StatusCode, Json<FallbackResponse>) {
    (StatusCode::NOT_FOUND, Json(FallbackResponse {
        message: format!("No route for {uri}"),
    }))
}

async fn healthcheck(headers: HeaderMap) -> GeneralResponseResult<BaseResponse<AntiSpoofingExtractionResultOutput>> {
    let request_id_header = headers.get("x-request-id").unwrap().to_str().unwrap();
    let request_id: String = request_id_header.parse().unwrap();

    Ok(GeneralResponseBuilder::new()
        .status_code(StatusCode::OK)
        .body(BaseResponse {
            data: None,
            response_message: "OK".to_string(),
            response_code: ResponseCode::response_code(ResponseCode::CodeOK),
            is_success: true,
            request_id: request_id.clone(),
        })
        .build())
}

