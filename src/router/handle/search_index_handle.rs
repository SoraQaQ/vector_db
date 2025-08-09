use axum::Json;
use faiss::Idx;
use log::info;
use validator::Validate;

use crate::{
    core::{
        index::{faiss_index::FaissIndex, hnsw_index::HnswIndex, usearch_index::UsearchIndex},
        index_factory::{IndexType, global_index_factory},
    },
    error::app_error::AppError,
    models::{request::search::SearchRequest, response::search::SearchResponse},
};

struct SearchResult {
    labels: Vec<u64>,
    distances: Vec<f32>,
}

impl SearchResult {
    pub fn from_faiss(result: (Vec<Idx>, Vec<f32>)) -> Result<Self, AppError> {
        let labels = result
            .0
            .into_iter()
            .map(|x| x.get().unwrap())
            .collect::<Vec<u64>>();
        let distances = result.1;
        Ok(SearchResult { labels, distances })
    }

    pub fn from_hnsw(result: (Vec<usize>, Vec<f32>)) -> Result<Self, AppError> {
        let labels = result.0.iter().map(|x| *x as u64).collect::<Vec<u64>>();
        Ok(SearchResult {
            labels,
            distances: result.1,
        })
    }

    pub fn from_usearch(result: (Vec<u64>, Vec<f32>)) -> Result<Self, AppError> {
        let labels = result.0;
        let distances = result.1;
        Ok(SearchResult { labels, distances })
    }
}

pub async fn search_handler(
    Json(payload): Json<SearchRequest>,
) -> Result<Json<SearchResponse>, AppError> {
    payload
        .validate()
        .map_err(|e| AppError::ValidationError(e.to_string()))?;

    info!("search_handler: {:?}", payload);

    let (index_key, vectors, k) = (
        payload.index_key.unwrap(),
        payload.vectors.unwrap(),
        payload.k.unwrap(),
    );

    let index_factory = global_index_factory();

    let index = index_factory
        .get_index(index_key)
        .ok_or_else(|| AppError::IndexNotFound(format!("{:?} index not found", index_key)))?;

    let search_result: SearchResult = match index_key.index_type {
        IndexType::FLAT => {
            let result = index
                .downcast_ref::<FaissIndex>()
                .unwrap()
                .search_vectors(&vectors, k)
                .map_err(|e| AppError::FaissError(format!("faiss search err: {e}")))?;

            SearchResult::from_faiss(result)?
        }
        IndexType::HNSW => {
            let hnsw_index = index.downcast_ref::<HnswIndex<f32>>().unwrap();
            let result = hnsw_index
                .search_vectors(&vectors, k, 200)
                .map_err(|e| AppError::HnswError(e.to_string()))?;

            SearchResult::from_hnsw(result)?
        }

        IndexType::USEARCH => {
            let usearch_index = index.downcast_ref::<UsearchIndex>().unwrap();
            let result = usearch_index
                .search(&vectors, k)
                .map_err(|e| AppError::UsearchError(format!("{e}")))?;
            SearchResult::from_usearch(result)?
        }
        _ => return Err(AppError::UnsupportedIndexType(index_key)),
    };

    Ok(Json(SearchResponse {
        code: 0,
        labels: search_result.labels,
        distances: search_result.distances,
        error_msg: None,
    }))
}

#[cfg(test)]
mod tests {
    use crate::core::index_factory::{IndexKey, MetricType};
    use axum::{
        Router,
        body::{Body, to_bytes},
        http::{Request, StatusCode},
        routing::post,
    };
    use rstest::*;
    use tower::Service;
    use usearch::IndexOptions;

    use super::*;

    fn setup_test_app() -> Router {
        axum::Router::new().route("/search", post(search_handler))
    }

    fn setup_search_json(vectors: Vec<f32>, k: usize, index_key: IndexKey) -> Request<Body> {
        Request::builder()
            .uri("/search")
            .method("POST")
            .header("Content-Type", "application/json")
            .body(Body::from(
                serde_json::json!({
                    "vectors": vectors,
                    "k": k,
                    "index_key": index_key
                })
                .to_string(),
            ))
            .unwrap()
    }

    #[rstest]
    #[case(vec![1.0, 2.0, 3.0], 3, IndexKey{index_type: IndexType::FLAT, dim: 3, metric_type: MetricType::L2}, StatusCode::NOT_FOUND)]
    #[case(vec![0.5, 1.5, 2.5], 3, IndexKey{index_type: IndexType::UNKNOWN, dim: 3, metric_type: MetricType::L2}, StatusCode::NOT_FOUND)]
    #[case(vec![], 1, IndexKey{index_type: IndexType::FLAT, dim: 3, metric_type: MetricType::L2}, StatusCode::BAD_REQUEST)]
    #[tokio::test]
    async fn test_search_handler(
        #[case] vectors: Vec<f32>,
        #[case] k: usize,
        #[case] index_key: IndexKey,
        #[case] expected_status: StatusCode,
    ) {
        env_logger::Builder::new()
            .filter_level(log::LevelFilter::Debug)
            .init();

        let factory = global_index_factory();

        let opt = IndexOptions::default();

        factory
            .init(IndexType::FLAT, 3, 1000, MetricType::L2, opt.clone())
            .unwrap();

        let request = setup_search_json(vectors, k, index_key);

        let mut app = setup_test_app();
        let response = app.call(request).await.unwrap();

        info!("response: {:?}", response);
        assert_eq!(response.status(), expected_status);

        let body = to_bytes(response.into_body(), 1024).await.unwrap();
        let body_str = String::from_utf8_lossy(&body);

        info!("response body: {}", body_str);
    }

    #[tokio::test]
    async fn test_search_success() {
        env_logger::Builder::new()
            .filter_level(log::LevelFilter::Debug)
            .init();

        let opt = IndexOptions::default();

        let factory = global_index_factory();
        factory
            .init(IndexType::HNSW, 3, 1000, MetricType::L2, opt.clone())
            .unwrap();

        factory
            .get_index(IndexKey {
                index_type: IndexType::HNSW,
                dim: 3,
                metric_type: MetricType::L2,
            })
            .unwrap()
            .downcast_ref::<HnswIndex<f32>>()
            .unwrap()
            .insert_vectors(&vec![1.0, 2.0, 3.0], 1)
            .unwrap();

        let request = setup_search_json(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            2,
            IndexKey {
                index_type: IndexType::HNSW,
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
