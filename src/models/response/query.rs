use serde::Serialize;

#[derive(Debug, Serialize)]
pub struct QueryResponse {
    pub code: i32,
    pub data: serde_json::Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error_msg: Option<String>,
}
