use axum::extract::Request;
use axum::middleware::Next;
use axum::response::IntoResponse;
use http::header::ToStrError;
use http::HeaderValue;
use log::error;

use crate::config::settings::SETTINGS;
use crate::error::errors::{AuthenticateError, Error};

pub async fn validate_api_key_mw(mut req: Request, next: Next) -> Result<impl IntoResponse, Error> {
    let api_key_header = req.headers_mut().get("x-api-key");
    match api_key_header {
        None => {
            return Err(Error::Authenticate(AuthenticateError::MissingCredentials))
        }
        Some(header) => {
            let api_key_value = match header.to_str() {
                Ok(api_key_value) => {api_key_value}
                Err(e) => {
                    error!("failed to parse token: {e}");
                    return Err(Error::Authenticate(AuthenticateError::InvalidToken))
                }
            };
            if SETTINGS.server.api_key != api_key_value.to_string() {
                return Err(Error::Authenticate(AuthenticateError::MissingCredentials))
            }
        }
    };
    return Ok(next.run(req).await)
}

