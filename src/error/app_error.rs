use axum::{http::StatusCode, response::IntoResponse, Json};
use thiserror::Error;

use crate::core::index_factory::IndexType;

#[derive(Debug, Error)]
pub enum AppError {
    // 通用错误
    #[error("Validation error: {0}")]
    ValidationError(String),

    // 索引类型特定的错误
    #[error("Faiss error: {0}")]
    FaissError(#[from] faiss::error::Error),
    
    // 索引管理错误
    #[error("Index not found: {0}")]
    IndexNotFound(String),
    
    #[error("Unsupported index type: {0}")]
    UnsupportedIndexType(IndexType),
}

impl IntoResponse for AppError {
    fn into_response(self) -> axum::response::Response {
        let status = match &self {
            AppError::ValidationError(_) => StatusCode::BAD_REQUEST,
            AppError::IndexNotFound(_) |
            AppError::UnsupportedIndexType(_) => StatusCode::NOT_FOUND,
            
            _ => StatusCode::INTERNAL_SERVER_ERROR,
        };
        
        let error_msg = self.to_string();
        
        let body = Json(serde_json::json!({
            "code": -1,
            "error_msg": error_msg
        }));
        
        (status, body).into_response()
    }
}