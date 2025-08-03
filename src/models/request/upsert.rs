use serde::Deserialize;
use validator::Validate;

use crate::core::index_factory::IndexKey;

#[derive(Debug, Deserialize, Validate)]
pub struct UpsertRequest {
    #[validate(length(min = 1, message = "vectors must contain at least one element"))]
    pub vectors: Option<Vec<f32>>,

    #[validate(required(message = "id cannot be empty"))]
    #[validate(range(min = 1, message = "id must be at least 1"))]
    pub id: Option<u64>,

    #[validate(required(message = "index_key cannot be empty"))]
    pub index_key: Option<IndexKey>,

    pub data: serde_json::Value,
}
