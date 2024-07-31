use std::str::ParseBoolError;
use anyhow::Error;
use axum::extract::{Multipart, State};
use axum::{debug_handler, Form, Json};
use axum::extract::multipart::MultipartError;
use axum::response::Response;
use bytes::Bytes;
use ecs_logger::extra_fields;
use http::{HeaderMap, Request, StatusCode};
use log::{info, error};
use opencv::calib3d::find_essential_mat;
use serde::{Deserialize, Serialize};
use uuid::Uuid;
use crate::config::settings::SETTINGS;
use crate::error::errors::ResponseCode;
use crate::logger::logger::LoggerExtraFields;
use crate::models::general_model::{GeneralExtractionInput, GeneralExtractionResultOutput};
use crate::pipeline::general_pipeline::general_pipeline::{GeneralPipeline, GeneralFaceExtractionResult};
use crate::response::common_response::{BaseResponse, ResponsePagination, GeneralResponse, GeneralResponseBuilder, GeneralResponseResult};
use crate::state::general_state::GeneralState;

#[debug_handler(state=GeneralState)]
pub async fn general_extract(headers: HeaderMap, State(state): State<GeneralState>, mut payload: Multipart) -> GeneralResponseResult<BaseResponse<GeneralExtractionResultOutput>> {
    let request_id_header = headers.get("x-request-id").unwrap().to_str().unwrap();
    let mut im_bytes: Bytes = Bytes::new();
    let request_id: String = request_id_header.parse().unwrap();
    let mut is_enroll: Option<bool> = Some(false);

    extra_fields::set_extra_fields(LoggerExtraFields {
        request_id: request_id.clone(),
    }).unwrap();

    info!("received general extraction request");

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
            _ => {}
        }
    }
    let input = GeneralExtractionInput {
        im_bytes,
        is_enroll,
    };

    let result = match state.general_service.extract_general_image(input).await {
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