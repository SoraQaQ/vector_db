use axum::Json;
use log::info;
use validator::Validate;

use crate::{
    core::{
        index::{faiss_index::FaissIndex, hnsw_index::HnswIndex},
        index_factory::{IndexType, global_index_factory},
    },
    error::app_error::AppError,
    models::{request::insert::InsertRequest, response::insert::InsertResponse},
};

pub async fn insert_handler(
    Json(payload): Json<InsertRequest>,
) -> Result<Json<InsertResponse>, AppError> {
    payload
        .validate()
        .map_err(|e| AppError::ValidationError(e.to_string()))?;

    info!("insert_handler: {:?}", payload);

    let (index_key, vectors, id) = (
        payload.index_key.unwrap(),
        payload.vectors.unwrap(),
        payload.id.unwrap(),
    );

    let index_factory = global_index_factory();

    let index = index_factory
        .get_index(index_key)
        .ok_or_else(|| AppError::UnsupportedIndexType(index_key))?;

    match index_key.index_type {
        IndexType::FLAT => {
            let faiss_index = index.downcast_ref::<FaissIndex>().unwrap();
            faiss_index
                .insert_vectors(&vectors, id)
                .map_err(|e| AppError::FaissError(e))?;
        }
        IndexType::HNSW => {
            let hnsw_index = index.downcast_ref::<HnswIndex<f32>>().unwrap();
            hnsw_index
                .insert_vectors(&vectors, id.try_into().unwrap())
                .map_err(|e| AppError::HnswError(e.to_string()))?;
        }
        _ => return Err(AppError::UnsupportedIndexType(index_key)),
    };

    Ok(Json(InsertResponse {
        code: 0,
        error_msg: None,
    }))
}

#[cfg(test)]
mod tests {
    use crate::core::index_factory::{IndexKey, MetricType};

    use super::*;
    use axum::{
        Router,
        body::{Body, to_bytes},
        http::{Request, StatusCode},
        routing::post,
    };
    use rstest::*;
    use tower::Service;

    fn setup_test_app() -> Router {
        axum::Router::new().route("/insert", post(insert_handler))
    }

    fn setup_insert_json(vectors: Vec<f32>, id: u64, index_key: IndexKey) -> Request<Body> {
        Request::builder()
            .uri("/insert")
            .method("POST")
            .header("Content-Type", "application/json")
            .body(Body::from(
                serde_json::json!({
                    "vectors": vectors,
                    "id": id,
                    "index_key": index_key
                })
                .to_string(),
            ))
            .unwrap()
    }

    #[rstest]
    #[case(IndexKey{index_type: IndexType::FLAT, dim: 3, metric_type: MetricType::L2}, vec![1.0, 2.0, 3.0], 1, StatusCode::OK)]
    #[case(IndexKey{index_type: IndexType::UNKNOWN, dim: 3, metric_type: MetricType::L2}, vec![1.0, 2.0, 3.0], 1, StatusCode::NOT_FOUND)]
    #[tokio::test]
    async fn test_insert_handler(
        #[case] index_key: IndexKey,
        #[case] vectors: Vec<f32>,
        #[case] id: u64,
        #[case] expected_status: StatusCode,
    ) {
        env_logger::Builder::new()
            .filter_level(log::LevelFilter::Debug)
            .init();

        let factory = global_index_factory();
        factory
            .init(
                index_key.index_type,
                index_key.dim,
                1000,
                index_key.metric_type,
            )
            .unwrap();

        let request = setup_insert_json(vectors, id, index_key);

        let mut app = setup_test_app();
        let response = app.call(request).await.unwrap();

        info!("response: {:?}", response);
        assert_eq!(response.status(), expected_status);

        let body = to_bytes(response.into_body(), 1024).await.unwrap();
        let body_str = String::from_utf8_lossy(&body);

        info!("response body: {}", body_str);
    }
}
