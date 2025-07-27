use serde::{Deserialize};
use validator::Validate;

use crate::core::index_factory::{IndexKey};


#[derive(Debug, Deserialize, Validate)] 
pub struct InsertRequest {
    #[validate(required(message = "vector cannot be empty"))]
    #[validate(length(min = 1, message = "vector must contain at least one element"))]
    pub vector: Option<Vec<f32>>,
    
    #[validate(required(message = "label cannot be empty"))]
    #[validate(range(min = 0, message = "label must be at least 0"))]
    pub label: Option<u64>,
    
    #[validate(required(message = "index_key cannot be empty"))]
    pub index_key: Option<IndexKey>,
}