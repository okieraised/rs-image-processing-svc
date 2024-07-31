use axum::extract::Request;
use axum::middleware::Next;
use axum::response::IntoResponse;
use http::header;
use uuid::Uuid;
use crate::error::errors::{Error};

pub async fn generate_request_id_mw(mut req: Request, next: Next) -> Result<impl IntoResponse, Error> {
    let request_id = Uuid::new_v4().to_string();

    req.headers_mut().insert(
        header::HeaderName::from_static("x-request-id"),
        header::HeaderValue::from_str(&request_id).unwrap(),
    );

    return Ok(next.run(req).await)
}