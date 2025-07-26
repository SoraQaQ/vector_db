use axum::Json;
use log::info;
use validator::Validate;

use crate::{core::{index::faiss_index::FaissIndex, index_factory::{global_index_factory, IndexType}}, error::app_error::AppError, models::{request::insert::InsertRequest, response::insert::InsertResponse}};

pub async fn insert_handler(
    Json(payload): Json<InsertRequest>,
) -> Result<Json<InsertResponse>, AppError> {
    payload
        .validate()
        .map_err(|e| AppError::ValidationError(e.to_string()))?;

    info!("insert_handler: {:?}", payload);

    let (index_type, vector, label) = (payload.index_type.unwrap(), payload.vector.unwrap(), payload.label.unwrap());

    let index_factory = global_index_factory(); 
    
    let index = index_factory
        .get_index(index_type)
        .ok_or_else(|| AppError::UnsupportedIndexType(index_type))?; 

    match index_type {
        IndexType::FLAT => {
            let faiss_index = index.downcast_ref::<FaissIndex>().unwrap();

            faiss_index
                .insert_vectors(&vector, label)
                .map_err(
                    |e| AppError::FaissError(e)
                )?;
        }, 
        _ => return Err(AppError::UnsupportedIndexType(index_type)),
    };

    Ok(Json(InsertResponse{
        code: 0,
        error_msg: None,
    }))    
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{body::{to_bytes, Body}, http::{Request, StatusCode}, Router};
    use rstest::*; 
    use tower::Service;

    fn setup_test_app() -> Router {
        axum::Router::new()
             .route("/insert", axum::routing::post(insert_handler))
     }

     fn setup_insert_json(vector: Vec<f32>, label: u64, index_type: IndexType) -> Request<Body> {
        Request::builder()
            .uri("/insert")
            .method("POST")
            .header("Content-Type", "application/json")
            .body(Body::from(
                serde_json::json!({
                    "vector": vector,
                    "label": label,
                    "index_type": index_type
                }).to_string(),
            ))
            .unwrap()
    }

    #[rstest]
    #[case(IndexType::FLAT, vec![1.0, 2.0, 3.0], 1, StatusCode::OK)]
    #[case(IndexType::UNKNOWN, vec![1.0, 2.0, 3.0], 1, StatusCode::NOT_FOUND)]
    #[tokio::test] 
    async fn test_insert_handler(
        #[case] index_type: IndexType,
        #[case] vector: Vec<f32>,
        #[case] label: u64,
        #[case] expected_status: StatusCode,
    ) {
        env_logger::Builder::new() 
            .filter_level(log::LevelFilter::Debug)
            .init();
        
        let factory = global_index_factory(); 
        factory.init(index_type, 3, faiss::MetricType::L2);

        let request = setup_insert_json(vector, label, index_type);

        let mut app = setup_test_app();
        let response = app.call(request).await.unwrap(); 

        info!("response: {:?}", response);
        assert_eq!(response.status(), expected_status);

        let body = to_bytes(response.into_body(), 1024).await.unwrap();
        let body_str = String::from_utf8_lossy(&body);

        info!("response body: {}", body_str);
    }
}