use serde::{Deserialize, Serialize};
use validator::{Validate, ValidationError};

use crate::core::index_factory::{IndexType, MetricType};

#[derive(Debug, Serialize, Deserialize, Validate)]
#[validate(schema(function = "validate_create_request"))]
pub struct CreateRequest {
    #[validate(required(message = "index_type cannot be empty"))]
    pub index_type: Option<IndexType>,

    #[validate(required(message = "dim cannot be empty"))]
    #[validate(range(min = 1, message = "dim must be at least 1"))]
    pub dim: Option<u32>,

    #[validate(required(message = "metric_type cannot be empty"))]
    pub metric_type: Option<MetricType>,

    #[serde(skip_serializing_if = "Option::is_none")]
    #[validate(range(min = 1, message = "max_elements must be at least 1"))]
    pub max_elements: Option<usize>,
}

fn validate_create_request(request: &CreateRequest) -> Result<(), ValidationError> {
    match request.index_type {
        Some(IndexType::HNSW) => {
            // For HNSW, max_elements is required
            if request.max_elements.is_none() {
                return Err(ValidationError::new(
                    "max_elements is required for HNSW index type",
                ));
            }
        }
        Some(_) => {
            // For non-HNSW types, max_elements must be None
            if request.max_elements.is_some() {
                return Err(ValidationError::new(
                    "max_elements is only allowed for HNSW index type",
                ));
            }
        }
        None => {
            // index_type is already validated as required, so this case won't happen
        }
    }
    Ok(())
}
