use axum::debug_handler;
use axum::extract::{Multipart, State};
use bytes::Bytes;
use ecs_logger::extra_fields;
use http::{HeaderMap, StatusCode};
use log::{error, info};
use opentelemetry::global;
use opentelemetry::global::ObjectSafeSpan;
use opentelemetry::trace::{TraceContextExt, Tracer};
use crate::config::settings::SETTINGS;
use crate::error::errors::ResponseCode;
use crate::logger::logger::LoggerExtraFields;
use crate::models::antispoofing_model::{AntiSpoofingExtractionInput, AntiSpoofingExtractionResultOutput};
use crate::response::common_response::{BaseResponse, GeneralResponseBuilder, GeneralResponseResult};
use crate::state::antispoofing_state::AntiSpoofingState;

#[debug_handler(state=AntiSpoofingState)]
pub async fn antispoofing_extract(headers: HeaderMap, State(state): State<AntiSpoofingState>, mut payload: Multipart) -> GeneralResponseResult<BaseResponse<AntiSpoofingExtractionResultOutput>> {
    let tracer = global::tracer(SETTINGS.app.name.clone());
    let parent_ctx = opentelemetry::Context::new();
    let span = tracer
        .span_builder("antispoofing-extraction")
        .start_with_context(&tracer, &parent_ctx);

    let request_id_header = headers.get("x-request-id").unwrap().to_str().unwrap();
    let mut im_bytes: Bytes = Bytes::new();
    let request_id: String = request_id_header.parse().unwrap();
    let mut is_enroll: Option<bool> = Some(false);
    let mut spoofing_check: Option<bool> = Some(false);

    extra_fields::set_extra_fields(LoggerExtraFields {
        request_id: request_id.clone(),
    }).unwrap();

    let child_ctx = parent_ctx.with_span(span);
    let mut child = tracer.start_with_context("marshal-request", &child_ctx);
    info!("received anti-spoofing extraction request");
    while let Some(field) = payload.next_field().await.unwrap() {
        let name = field.name().unwrap().to_string();
        match name.as_str() {
            "images" => {
                match field.bytes().await {
                    Ok(data) => {
                        if data.len() == 0 {
                            return Ok(GeneralResponseBuilder::new()
                                .status_code(StatusCode::BAD_REQUEST)
                                .body(BaseResponse {
                                    data: None,
                                    response_message: "image is empty".to_string(),
                                    response_code: ResponseCode::response_code(ResponseCode::ErrorCodeInput),
                                    is_success: false,
                                    request_id: request_id.clone(),
                                })
                                .build()
                            )
                        }
                        im_bytes = data;
                    }
                    Err(e) => {
                        error!("failed to retrieves image from request: {e}");
                        return Ok(GeneralResponseBuilder::new()
                            .status_code(StatusCode::BAD_REQUEST)
                            .body(BaseResponse {
                                data: None,
                                response_message: "failed to process image".to_string(),
                                response_code: ResponseCode::response_code(ResponseCode::ErrorCodeInput),
                                is_success: false,
                                request_id: request_id.clone(),
                            })
                            .build()
                        )
                    }
                };
            }
            "is_enroll" => {
                let value = field.text().await.unwrap();
                match value.parse::<bool>() {
                    Ok(val) => {
                        is_enroll = Some(val);
                    }
                    Err(e) => {
                        error!("failed to retrieves is_enroll value [{value}] from request: {e}");
                        return Ok(GeneralResponseBuilder::new()
                            .status_code(StatusCode::BAD_REQUEST)
                            .body(BaseResponse {
                                data: None,
                                response_message: "invalid boolean value".to_string(),
                                response_code: ResponseCode::response_code(ResponseCode::ErrorCodeInput),
                                is_success: false,
                                request_id: request_id.clone(),
                            })
                            .build()
                        )
                    }
                }
            }
            "spoofing_check" => {
                let value = field.text().await.unwrap();
                match value.parse::<bool>() {
                    Ok(val) => {
                        spoofing_check = Some(val);
                    }
                    Err(e) => {
                        error!("failed to retrieves spoofing_check value [{value}] from request: {e}");
                        return Ok(GeneralResponseBuilder::new()
                            .status_code(StatusCode::BAD_REQUEST)
                            .body(BaseResponse {
                                data: None,
                                response_message: "invalid boolean value".to_string(),
                                response_code: ResponseCode::response_code(ResponseCode::ErrorCodeInput),
                                is_success: false,
                                request_id: request_id.clone(),
                            })
                            .build()
                        )
                    }
                }
            }
            _ => {}
        }
    }
    let input = AntiSpoofingExtractionInput {
        im_bytes,
        is_enroll,
        spoofing_check,
    };
    child.end();

    let mut child = tracer.start_with_context("extract-image", &child_ctx);
    let result = match state.anti_spoofing_service.extract_antispoofing_image(input).await {
        Ok(result) => {result}
        Err(e) => {
            error!("failed to extract face: {e}");
            return Ok(GeneralResponseBuilder::new()
                .status_code(StatusCode::INTERNAL_SERVER_ERROR)
                .body(BaseResponse {
                    data: None,
                    response_message: "internal server error".to_string(),
                    response_code: ResponseCode::response_code(ResponseCode::ErrorCodeServer),
                    is_success: false,
                    request_id: request_id.clone(),
                })
                .build()
            )
        }
    };
    info!("completed extracting image");
    extra_fields::clear_extra_fields();
    child.end();

    return Ok(GeneralResponseBuilder::new()
        .status_code(StatusCode::OK)
        .body(BaseResponse {
            data: Some(result),
            response_message: "OK".to_string(),
            response_code: ResponseCode::response_code(ResponseCode::CodeOK),
            is_success: true,
            request_id: request_id.clone(),
        })
        .build()
    )
}