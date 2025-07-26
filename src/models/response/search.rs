use serde::Serialize;

#[derive(Debug, Serialize)] 
pub struct SearchResponse {
    pub code: i32, 
    pub labels: Vec<i64>,
    pub distances: Vec<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error_msg: Option<String>,
}
