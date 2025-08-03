use serde::Deserialize;
use validator::Validate;

#[derive(Debug, Deserialize, Validate)]
pub struct QueryRequest {
    #[validate(required(message = "id cannot be empty"))]
    #[validate(range(min = 1, message = "id must be at least 1"))]
    pub id: Option<u64>,
}
