use axum::{Json, extract::State};
use std::sync::Arc;

use crate::{
    db::vector_database::VectorDatabase,
    error::app_error::AppError,
    models::{request::query::QueryRequest, response::query::QueryResponse},
};
use validator::Validate;

pub async fn search_handle(
    State(vector_database): State<Arc<VectorDatabase>>,
    Json(payload): Json<QueryRequest>,
) -> Result<Json<QueryResponse>, AppError> {
    payload
        .validate()
        .map_err(|e| AppError::ValidationError(e.to_string()))?;

    let id = payload.id.unwrap();

    let data = vector_database
        .query(id)
        .ok_or_else(|| AppError::QueryError(format!("vector database query id {} failed", id)))?;

    Ok(Json(QueryResponse {
        code: 0,
        data: data,
        error_msg: None,
    }))
}
