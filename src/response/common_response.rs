use axum::http::{header, HeaderValue, StatusCode};
use axum::response::{IntoResponse, IntoResponseParts, Response, ResponseParts};
use serde::{Deserialize, Serialize};
use bytes::{BufMut, BytesMut};
use log::error;
use uuid::Uuid;
use crate::error::errors::{Error, ResponseCode};
pub type GeneralResponseResult<T> = Result<GeneralResponse<T>, Error>;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BaseResponse<T: Serialize> {
    pub data: Option<T>,
    pub response_message: String,
    pub response_code: u16,
    pub is_success: bool,
    pub request_id: String,
}

impl<T> Default for BaseResponse<T>
    where
        T: Serialize,
{
    fn default() -> Self {
        Self {
            data: None,
            response_message: "OK".to_string(),
            response_code: ResponseCode::response_code(ResponseCode::CodeOK),
            is_success: true,
            request_id: Uuid::new_v4().to_string(),
        }
    }
}

#[derive(Debug)]
pub struct GeneralResponse<T: Serialize> {
    pub data: Option<T>,
    pub status_code: StatusCode,
    pub pagination: Option<ResponsePagination>,
}

pub struct GeneralResponseBuilder<T: Serialize> {
    pub data: Option<T>,
    pub status_code: StatusCode,
    pub pagination: Option<ResponsePagination>,
}

#[derive(Debug)]
pub struct ResponsePagination {
    pub count: u64,
    pub offset: u64,
    pub limit: u32,
}

impl<T> Default for GeneralResponseBuilder<T>
    where
        T: Serialize,
{
    fn default() -> Self {
        Self {
            data: None,
            status_code: StatusCode::OK,
            pagination: None,
        }
    }
}

impl<T> GeneralResponseBuilder<T>
    where
        T: Serialize,
{
    pub fn new() -> Self {
        Self::default()
    }

    pub fn body(mut self, body: T) -> Self {
        self.data = Some(body);
        self
    }

    pub fn status_code(mut self, status_code: StatusCode) -> Self {
        self.status_code = status_code;
        self
    }

    pub fn pagination(mut self, pagination: ResponsePagination) -> Self {
        self.pagination = Some(pagination);
        self
    }

    pub fn build(self) -> GeneralResponse<T> {
        GeneralResponse {
            data: self.data,
            status_code: self.status_code,
            pagination: self.pagination,
        }
    }
}

impl<T> IntoResponse for GeneralResponse<T>
    where
        T: Serialize,
{
    fn into_response(self) -> Response {

        let data = match self.data {
            Some(data) => {data},
            None => return (self.status_code).into_response(),
        };

        let mut bytes = BytesMut::new().writer();
        if let Err(err) = serde_json::to_writer(&mut bytes, &data) {
            error!("Error serializing response body as JSON: {:?}", err);
            return (StatusCode::INTERNAL_SERVER_ERROR).into_response();
        }

        let bytes = bytes.into_inner().freeze();
        let headers = [(
            header::CONTENT_TYPE,
            HeaderValue::from_static(mime::APPLICATION_JSON.as_ref()),
        )];

        match self.pagination {
            Some(pagination) => (self.status_code, pagination, headers, bytes).into_response(),
            None => (self.status_code, headers, bytes).into_response(),
        }
    }
}

impl IntoResponseParts for ResponsePagination {
    type Error = (StatusCode, String);

    fn into_response_parts(self, mut res: ResponseParts) -> Result<ResponseParts, Self::Error> {
        res.headers_mut()
            .insert("x-pagination-count", self.count.into());

        res.headers_mut()
            .insert("x-pagination-offset", self.offset.into());

        res.headers_mut()
            .insert("x-pagination-limit", self.limit.into());

        Ok(res)
    }
}