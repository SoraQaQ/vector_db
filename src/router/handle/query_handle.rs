use axum::{Json, extract::State};
use log::info;
use std::sync::Arc;

use crate::{
    db::vector_database::VectorDatabase,
    error::app_error::AppError,
    models::{request::query::QueryRequest, response::query::QueryResponse},
};
use validator::Validate;

pub async fn query_handle(
    State(vector_database): State<Arc<VectorDatabase>>,
    Json(payload): Json<QueryRequest>,
) -> Result<Json<QueryResponse>, AppError> {
    payload
        .validate()
        .map_err(|e| AppError::ValidationError(e.to_string()))?;

    info!("query_handle: {:?}", payload);

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

#[cfg(test)]
mod tests {
    use axum::{
        Router,
        body::{Body, to_bytes},
        http::{Request, StatusCode},
        routing::post,
    };

    use tower::Service;

    use super::*;

    fn setup_test_app() -> Router {
        let db = Arc::new(VectorDatabase::new("test".to_string()));
        let app = Router::new()
            .route("/query", post(query_handle))
            .with_state(db.clone());
        app
    }

    fn setup_query_json(id: u64) -> Request<Body> {
        Request::builder()
            .uri("/query")
            .method("POST")
            .header("Content-Type", "application/json")
            .body(Body::from(
                serde_json::json!({
                    "id": id,
                })
                .to_string(),
            ))
            .unwrap()
    }

    #[tokio::test]
    async fn test_query_handle() {
        env_logger::Builder::new()
            .filter_level(log::LevelFilter::Debug)
            .init();

        let mut app = setup_test_app();

        let req = setup_query_json(1);

        let res = app.call(req).await.unwrap();

        info!("test_query_handle: {:?}", res);
        assert_eq!(res.status(), StatusCode::OK);

        let body = to_bytes(res.into_body(), 1024).await.unwrap();
        let body_str = String::from_utf8_lossy(&body);

        info!("response body: {}", body_str);
    }
}
