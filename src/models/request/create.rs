use serde::{Deserialize, Serialize};
use validator::Validate;

use crate::core::index_factory::{IndexType, MyMetricType};

#[derive(Debug, Serialize, Deserialize, Validate)]
pub struct CreateRequest {
    #[validate(required(message = "index_type cannot be empty"))]
    pub index_type: Option<IndexType>, 
    
    #[validate(required(message = "dim cannot be empty"))]
    #[validate(range(min = 1, message = "dim must be at least 1"))]
    pub dim: Option<u32>, 
    
    #[validate(required(message = "metric_type cannot be empty"))]
    pub metric_type: Option<MyMetricType>,
}
