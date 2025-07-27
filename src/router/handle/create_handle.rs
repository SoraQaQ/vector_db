use axum::Json;
use log::info;
use validator::Validate;

use crate::{core::index_factory::{global_index_factory, IndexKey}, error::app_error::AppError, models::{request::create::CreateRequest, response::create::CreateResponse}};

pub async fn create_handler(
    Json(payload): Json<CreateRequest>,
) -> Result<Json<CreateResponse>, AppError> {
    payload
        .validate()
        .map_err(|e| AppError::ValidationError(e.to_string()))?;

    info!("create_handler: {:?}", payload);

    let (index_type, dim, metric_type) = (payload.index_type.unwrap(), payload.dim.unwrap(), payload.metric_type.unwrap());

    let index_factory = global_index_factory(); 
    
    index_factory.init(index_type, dim, metric_type).map_err(|e| AppError::InitIndexError(IndexKey { index_type, dim, metric_type }, e.to_string()))?;

    Ok(Json(CreateResponse{
        code: 0,
        error_msg: None,
        index_key: Some(IndexKey { index_type, dim, metric_type }),
    }))    
}

#[cfg(test)]
mod tests {
    use axum::{body::{to_bytes, Body}, http::{StatusCode, Request}, routing::Router};

    use crate::{core::index_factory::{IndexType, MyMetricType}, router::handle::create_handle::create_handler};
    use rstest::*;
    use tower::Service;
    
    fn setup_create_json(index_type: IndexType, dim: u32, metric_type: MyMetricType) -> Request<Body> {
        Request::builder()
        .uri("/insert")
        .method("POST")
        .header("Content-Type", "application/json")
        .body(Body::from(
            serde_json::json!({
                "index_type": index_type,
                "dim": dim,
                "metric_type": metric_type,
            }).to_string(),
        ))
        .unwrap()
    } 

    fn app() -> Router {
        axum::Router::new()
            .route("/insert", axum::routing::post(create_handler))
    }

    #[rstest] 
    #[case(IndexType::FLAT, 128, MyMetricType::L2, StatusCode::OK)]
    #[case(IndexType::FLAT, 256, MyMetricType::L2, StatusCode::OK)]
    #[case(IndexType::FLAT, 10, MyMetricType::InnerProduct, StatusCode::OK)]
    #[case(IndexType::UNKNOWN, 128, MyMetricType::L2, StatusCode::INTERNAL_SERVER_ERROR)]
    #[tokio::test] 
    async fn test_create_handler(
        #[case] index_type: IndexType,
        #[case] dim: u32,
        #[case] metric_type: MyMetricType,
        #[case] expected_status: StatusCode,
    ) {
        use log::info;

        env_logger::Builder::new() 
            .filter_level(log::LevelFilter::Debug)
            .init();
        
        let request = setup_create_json(index_type, dim, metric_type);
        
        let mut app = app();
        let response = app.call(request).await.unwrap(); 

        info!("response: {:?}", response);
        assert_eq!(response.status(), expected_status);

        let body = to_bytes(response.into_body(), 1024).await.unwrap();
        let body_str = String::from_utf8_lossy(&body);

        info!("response body: {}", body_str);
    }
}