use axum::Json;
use log::info;
use validator::Validate;

use crate::{core::{index::faiss_index::FaissIndex, index_factory::{global_index_factory, IndexType}}, error::app_error::AppError, models::{request::search::SearchRequest, response::search::SearchResponse}};

struct SearchResult {
    labels: Vec<i64>,
    distances: Vec<f32>,
}

impl SearchResult {
    pub fn from_faiss(
        result: faiss::index::SearchResult 
    ) -> Result<Self, AppError> {
        let labels = result.labels
            .iter()
            .map(|x| x
                .get()
                .map(|id| id as i64).unwrap_or(-1))
            .collect::<Vec<i64>>();
        
        Ok(SearchResult{labels, distances: result.distances})
    }
}


pub async fn search_handler(
    Json(payload): Json<SearchRequest>,
) -> Result<Json<SearchResponse>, AppError> {

    payload
        .validate()
        .map_err(
            |e| AppError::ValidationError(e.to_string())
        )?;
    
    info!("search_handler: {:?}", payload);

    let (index_type, query, k) = (payload.index_type.unwrap(), payload.query.unwrap(), payload.k.unwrap());

    let index_factory = global_index_factory();
    
    let index = index_factory
        .get_index(index_type)
        .ok_or_else(
            || AppError::UnsupportedIndexType(index_type)
        )?; 

    let search_result: SearchResult = match index_type {
        IndexType::FLAT => {
            let result = index.downcast_ref::<FaissIndex>().unwrap()
                .search_vectors(&query, k)
                .map_err(|e| AppError::FaissError(e))?;
            
            let search_result = SearchResult::from_faiss(result)?;

            Ok::<SearchResult, AppError>(search_result)
        }, 
        _ => return Err(AppError::UnsupportedIndexType(index_type)),
    }?;

    Ok(Json(SearchResponse{
        code: 0,
        labels: search_result.labels,
        distances: search_result.distances,
        error_msg: None,    
    }))    
}


#[cfg(test)] 
mod tests {
    use axum::{body::{to_bytes, Body}, http::{Request, StatusCode}, Router};
    use tower::Service;
    use rstest::*;
    use super::*;
    

    fn setup_test_app() -> Router {
       axum::Router::new()
            .route("/search", axum::routing::post(search_handler))
    }

    fn setup_search_json(query: Vec<f32>, k: usize, index_type: IndexType) -> Request<Body> {
        Request::builder()
            .uri("/search")
            .method("POST")
            .header("Content-Type", "application/json")
            .body(Body::from(
                serde_json::json!({
                    "query": query,
                    "k": k,
                    "index_type": index_type
                }).to_string(),
            ))
            .unwrap()
    }
    
    #[rstest]
    #[case(vec![1.0, 2.0, 3.0], 3, IndexType::FLAT, StatusCode::NOT_FOUND)]
    #[case(vec![0.5, 1.5, 2.5], 3, IndexType::UNKNOWN, StatusCode::NOT_FOUND)]
    #[case(vec![], 1, IndexType::FLAT, StatusCode::BAD_REQUEST)]
    #[tokio::test] 
    async fn test_search_handler(
        #[case] query: Vec<f32>,
        #[case] k: usize,
        #[case] index_type: IndexType,
        #[case] expected_status: StatusCode,
    ) {
        env_logger::Builder::new() 
            .filter_level(log::LevelFilter::Debug)
            .init();
        
        let factory = global_index_factory(); 
        factory.init(IndexType::FLAT, 3, faiss::MetricType::L2);

        let request = setup_search_json(query, k, index_type);

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
        
        let factory = global_index_factory(); 
        factory.init(IndexType::FLAT, 3, faiss::MetricType::L2);

        factory.get_index(IndexType::FLAT).unwrap().downcast_ref::<FaissIndex>().unwrap().insert_vectors(&vec![1.0, 2.0, 3.0], 1).unwrap();

        let request = setup_search_json(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, IndexType::FLAT);

        let mut app = setup_test_app();
        let response = app.call(request).await.unwrap(); 

        info!("response: {:?}", response);
        assert_eq!(response.status(), StatusCode::OK);

        let body = to_bytes(response.into_body(), 1024).await.unwrap();
        let body_str = String::from_utf8_lossy(&body);

        info!("response body: {}", body_str);
    }
    
    
}