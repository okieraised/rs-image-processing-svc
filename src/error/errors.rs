use axum::Json;
use axum::response::{IntoResponse, Response};
use http::StatusCode;
use serde::Serialize;
use serde_json::json;


#[derive(Copy, Clone, Serialize)]
#[repr(u16)]
pub enum ResponseCode {
    CodeOK = 0,
    ErrorCodeAuth = 1,
    ErrorCodeInput = 2,
    ErrorCodeServer = 3,
    ErrorCodeTimeout = 4,
    ErrorCodeDatabase = 5,
    ErrorCodeValidation = 6,
}

impl ResponseCode {
    pub fn response_code(v: ResponseCode) -> u16 {
        v as u16
    }
}

#[derive(thiserror::Error, Debug)]
#[error("...")]
pub enum Error {

    #[error("{0}")]
    OK(#[from] OK),

    #[error("{0}")]
    Authenticate(#[from] AuthenticateError),

    #[error("{0}")]
    BadRequest(#[from] BadRequestError),

    #[error("{0}")]
    NotFound(#[from] NotFoundError),

    #[error("{0}")]
    Timeout(#[from] TimeoutError),

    #[error("{0}")]
    Server(#[from] ServerError),
}

impl Error {
    fn get_codes(&self) -> (StatusCode, u16, bool) {
        match *self {
            // 2XX OK
            Error::OK(_) => (StatusCode::OK, ResponseCode::response_code(ResponseCode::CodeOK), true),

            // 4XX Errors
            Error::BadRequest(_) => (StatusCode::BAD_REQUEST, ResponseCode::response_code(ResponseCode::ErrorCodeInput), false),
            Error::NotFound(_) => (StatusCode::NOT_FOUND,  ResponseCode::response_code(ResponseCode::ErrorCodeInput), false),
            Error::Authenticate(AuthenticateError::MissingCredentials) => (StatusCode::UNAUTHORIZED, ResponseCode::response_code(ResponseCode::ErrorCodeAuth), false),
            Error::Authenticate(AuthenticateError::WrongCredentials) => (StatusCode::FORBIDDEN, ResponseCode::response_code(ResponseCode::ErrorCodeAuth), false),
            Error::Authenticate(AuthenticateError::InvalidToken) => (StatusCode::UNAUTHORIZED, ResponseCode::response_code(ResponseCode::ErrorCodeAuth), false),

            // 5XX Errors
            Error::Server(_) => (StatusCode::INTERNAL_SERVER_ERROR, ResponseCode::response_code(ResponseCode::ErrorCodeServer), false),
            Error::Timeout(_) => (StatusCode::GATEWAY_TIMEOUT, ResponseCode::response_code(ResponseCode::ErrorCodeServer), false),
        }
    }

    pub fn ok() -> Self {
        Error::OK(OK {})
    }

    pub fn bad_request() -> Self {
        Error::BadRequest(BadRequestError {})
    }

    pub fn not_found() -> Self {
        Error::NotFound(NotFoundError {})
    }

    // pub fn authenticate() -> Self {
    //     Error::Authenticate(AuthenticateError {})
    // }

    pub fn server() -> Self {
        Error::Server(ServerError {})
    }

    pub fn timeout() -> Self {
        Error::Timeout(TimeoutError {})
    }
}

impl IntoResponse for Error {
    fn into_response(self) -> Response {
        let (status_code, code, success) = self.get_codes();
        let message = self.to_string();
        let is_success = success.to_string();
        let body = Json(json!({ "code": code, "message": message }));

        (status_code, body).into_response()
    }
}

#[derive(thiserror::Error, Debug)]
#[error("...")]
pub enum AuthenticateError {
    #[error("Wrong authentication credentials")]
    WrongCredentials,
    #[error("Invalid authentication credentials")]
    InvalidToken,
    #[error("Missing authentication credentials")]
    MissingCredentials,

}

#[derive(thiserror::Error, Debug, Serialize)]
#[error("Bad Request")]
pub struct BadRequestError {}

#[derive(thiserror::Error, Debug, Serialize)]
#[error("Not found")]
pub struct NotFoundError {}

#[derive(thiserror::Error, Debug, Serialize)]
#[error("Ok")]
pub struct OK {}

#[derive(thiserror::Error, Debug, Serialize)]
#[error("server error")]
pub struct ServerError {}

#[derive(thiserror::Error, Debug, Serialize)]
#[error("server timeout")]
pub struct TimeoutError {}
