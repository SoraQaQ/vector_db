use serde::Serialize;
use crate::core::index_factory::IndexKey;

#[derive(Debug, Serialize)]
pub struct CreateResponse {
    pub code: i32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error_msg: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub index_key: Option<IndexKey>,
}
