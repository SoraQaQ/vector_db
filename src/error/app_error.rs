use axum::{Json, http::StatusCode, response::IntoResponse};
use thiserror::Error;

use crate::core::index_factory::IndexKey;

#[derive(Debug, Error)]
pub enum AppError {
    // 通用错误
    #[error("Validation error: {0}")]
    ValidationError(String),

    // 索引类型特定的错误
    #[error("Faiss error: {0}")]
    FaissError(#[from] faiss::error::Error),

    #[error("Hnsw error: {0}")]
    HnswError(String),

    // 索引管理错误
    #[error("Index not found: {0}")]
    IndexNotFound(String),

    #[error("Unsupported index type: {0}")]
    UnsupportedIndexType(IndexKey),

    #[error("Init {0} index error: {1}")]
    InitIndexError(IndexKey, String),

    #[error("Upsert error: {0}")]
    UpsertError(String),

    #[error("Query error: {0}")]
    QueryError(String),
}

impl IntoResponse for AppError {
    fn into_response(self) -> axum::response::Response {
        let status = match &self {
            AppError::ValidationError(_) => StatusCode::BAD_REQUEST,
            AppError::IndexNotFound(_) | AppError::UnsupportedIndexType(_) => StatusCode::NOT_FOUND,
            AppError::InitIndexError(_, _) => StatusCode::INTERNAL_SERVER_ERROR,
            AppError::UpsertError(_) => StatusCode::INTERNAL_SERVER_ERROR,
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
