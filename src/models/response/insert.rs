use serde::Serialize;

#[derive(Debug, Serialize)]
pub struct InsertResponse {
    pub code: i32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error_msg: Option<String>,
}
