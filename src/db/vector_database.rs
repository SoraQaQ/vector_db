use crate::{
    core::{
        index::{faiss_index::FaissIndex, hnsw_index::HnswIndex},
        index_factory::{IndexKey, IndexType, global_index_factory},
    },
    db::scalar_storage::ScalarStorage,
};
use anyhow::{Result, anyhow};
use log::info;
use rocksdb::DB;

pub struct VectorDatabase {
    scalar_storage: ScalarStorage,
}

impl VectorDatabase {
    pub fn new(db_path: String) -> Self {
        let db = DB::open_default(db_path).unwrap();
        Self {
            scalar_storage: ScalarStorage { db },
        }
    }

    pub fn upsert(&self, id: u64, data: serde_json::Value, index_key: IndexKey) -> Result<()> {
        info!("upsert data: {:?}", data);
        let index = global_index_factory()
            .get_index(index_key)
            .ok_or_else(|| anyhow!("index not found"))?;

        if self.scalar_storage.get_scalar(id).is_some() {
            match index_key.index_type {
                IndexType::FLAT => {
                    let faiss_index = index.downcast_ref::<FaissIndex>().unwrap();
                    faiss_index.remove_vectors(&[id])?;
                }
                IndexType::HNSW => {
                    // let hnsw_index = index.downcast_ref::<HnswIndex<f32>>().unwrap();
                    info!("unimplemented");
                }

                IndexType::UNKNOWN => {
                    return Err(anyhow!("index type unknown"));
                }
                _ => {}
            }
        }

        let new_vectors = data
            .get("vectors")
            .and_then(|v| v.as_array())
            .ok_or_else(|| anyhow!("vectors field not found or not an array"))?
            .iter()
            .map(|v| {
                v.as_f64()
                    .map(|x| x as f32)
                    .ok_or_else(|| anyhow!("vector element is not a number"))
            })
            .collect::<Result<Vec<f32>>>()?;

        info!("upsert new vectors: {:?}", new_vectors);

        match index_key.index_type {
            IndexType::FLAT => {
                let faiss_index = index.downcast_ref::<FaissIndex>().unwrap();
                faiss_index.insert_vectors(&new_vectors, id.try_into().unwrap())?;
            }
            IndexType::HNSW => {
                let hnsw_index = index.downcast_ref::<HnswIndex<f32>>().unwrap();
                hnsw_index.insert_vectors(&new_vectors, id.try_into().unwrap())?;
            }
            IndexType::UNKNOWN => {
                return Err(anyhow!("index type unknown"));
            }
            _ => {}
        }

        self.scalar_storage.insert_scalar(id, data)?;

        Ok(())
    }

    pub fn query(&self, id: u64) -> Option<serde_json::Value> {
        self.scalar_storage.get_scalar(id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        core::index_factory::MetricType, models::request::create::CreateRequest,
        router::handle::create_index_handle::create_handler,
    };
    use axum::Json;
    use tempfile::TempDir;

    #[tokio::test]
    async fn test_vector_database() {
        let temp_dir = TempDir::new().unwrap();
        let db = DB::open_default(temp_dir.path()).unwrap();
        let vector_database = VectorDatabase {
            scalar_storage: ScalarStorage { db },
        };
        let data = serde_json::json!({"name": "sora", "age": 20});
        let result = vector_database.upsert(
            1,
            data,
            IndexKey {
                index_type: IndexType::FLAT,
                dim: 128,
                metric_type: MetricType::L2,
            },
        );
        assert!(result.is_err());

        let result = create_handler(Json(CreateRequest {
            index_type: Some(IndexType::FLAT),
            dim: Some(128),
            metric_type: Some(MetricType::L2),
            max_elements: None,
        }))
        .await;

        eprintln!("err: {:?}", result.as_ref().err());

        assert!(result.unwrap().code == 0);

        let result = vector_database.upsert(
            1,
            serde_json::json!({"name": "sora", "age": 20, "vectors": [1.0, 2.0, 3.0]}),
            IndexKey {
                index_type: IndexType::FLAT,
                dim: 128,
                metric_type: MetricType::L2,
            },
        );

        assert!(result.is_ok());

        let data = vector_database.query(1);
        assert_eq!(
            data.unwrap(),
            serde_json::json!({"name": "sora", "age": 20, "vectors": [1.0, 2.0, 3.0]})
        );
    }
}
