use serde::Deserialize;
use validator::Validate;
use crate::core::index_factory::IndexType;

#[derive(Debug, Deserialize, Validate)]
pub struct SearchRequest {
    
    #[validate(required(message = "query cannot be empty"))]
    #[validate(length(min = 1, message = "query must contain at least one element"))]
    pub query: Option<Vec<f32>>,

    #[validate(required(message = "k cannot be empty"))]
    #[validate(range(min = 1, message = "k must be at least 1"))]
    pub k: Option<usize>,
    
    #[validate(required(message = "index_type cannot be empty"))]
    pub index_type: Option<IndexType>,
}