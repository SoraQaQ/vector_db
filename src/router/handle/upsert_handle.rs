use crate::{
    db::vector_database::VectorDatabase,
    error::app_error::AppError,
    models::{request::upsert::UpsertRequest, response::upsert::UpsertResponse},
};
use axum::{Json, extract::State};
use log::info;
use std::sync::Arc;
use validator::Validate;

pub async fn upsert_handle(
    State(vector_database): State<Arc<VectorDatabase>>,
    Json(payload): Json<UpsertRequest>,
) -> Result<Json<UpsertResponse>, AppError> {
    payload
        .validate()
        .map_err(|e| AppError::ValidationError(e.to_string()))?;

    info!("upsert_handle: {:?}", payload);

    let mut data = payload.data;

    if payload.vectors.is_some() {
        data["vectors"] = serde_json::Value::from(
            payload
                .vectors
                .unwrap()
                .into_iter()
                .map(|v| serde_json::Value::from(v))
                .collect::<Vec<_>>(),
        );
    }

    let (id, index_key) = (payload.id.unwrap(), payload.index_key.unwrap());

    vector_database
        .upsert(id, data, index_key)
        .map_err(|e| AppError::UpsertError(e.to_string()))?;

    Ok(Json(UpsertResponse {
        code: 0,
        error_msg: None,
    }))
}

#[cfg(test)]
mod tests {
    use axum::{Router, body::Body, http::Request};
    use std::sync::Arc;

    use crate::core::index_factory::{self, IndexKey, IndexType, MetricType};
    use axum::body::to_bytes;
    use axum::http::StatusCode;
    use log::*;
    use tower::Service;

    use super::*;

    fn setup_test_app() -> Router {
        let vector_database = Arc::new(VectorDatabase::new("your_db_path".to_string()));
        let app = axum::Router::new()
            .route("/upsert", axum::routing::post(upsert_handle))
            .with_state(vector_database.clone());
        app
    }

    fn setup_upsert_json(vectors: Vec<f32>, id: u64, index_key: IndexKey) -> Request<Body> {
        Request::builder()
            .uri("/upsert")
            .method("POST")
            .header("Content-Type", "application/json")
            .body(Body::from(
                serde_json::json!({
                    "vectors": vectors,
                    "id": id,
                    "index_key": index_key,
                    "data": serde_json::json!({"name": "sora", "age": 20})
                })
                .to_string(),
            ))
            .unwrap()
    }

    #[tokio::test]
    async fn test_upsert_handler() {
        env_logger::Builder::new()
            .filter_level(log::LevelFilter::Debug)
            .init();

        index_factory::global_index_factory()
            .init(IndexType::FLAT, 3, 1000, MetricType::L2)
            .unwrap();

        let request = setup_upsert_json(
            vec![1.0, 2.0, 3.0],
            1,
            IndexKey {
                index_type: IndexType::FLAT,
                dim: 3,
                metric_type: MetricType::L2,
            },
        );

        let mut app = setup_test_app();
        let response = app.call(request).await.unwrap();

        info!("response: {:?}", response);
        assert_eq!(response.status(), StatusCode::OK);

        let body = to_bytes(response.into_body(), 1024).await.unwrap();
        let body_str = String::from_utf8_lossy(&body);

        info!("response body: {}", body_str);
    }
}
