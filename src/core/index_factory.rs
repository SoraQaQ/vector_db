use std::{collections::HashMap, fmt, sync::{OnceLock, RwLock}};
use faiss::MetricType;
use log::{warn, info};
use anyhow::{Error};
use serde::{Deserialize, Serialize};
use crate::core::builder::{faiss_index_builder::FaissIndexBuilder, index_handle::{IndexBuilder, IndexHandle}};


#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Deserialize, Serialize)]
pub enum IndexType {
    FLAT = 0, 
    UNKNOWN = -1
}

impl fmt::Display for IndexType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            IndexType::FLAT => write!(f, "FLAT"),
            IndexType::UNKNOWN => write!(f, "UNKNOWN"),
        }
    }
}

pub struct IndexFactory {
    index_map: RwLock<HashMap<IndexType, IndexHandle>>,
}

impl IndexFactory {
    pub fn init(&self, index_type: IndexType, dim: u32, metric_type: MetricType) {
        let faiss_metric = if metric_type == MetricType::L2 {
            MetricType::L2 
        } else {
            MetricType::InnerProduct
        };

        info!("init index: {:?}", index_type);
        match index_type { 
            IndexType::FLAT => {
                let builder = FaissIndexBuilder::default()
                    .dim(dim)
                    .description("IDMap,Flat")
                    .metric_type(faiss_metric);

                let index = builder.build();
                
                self.index_map
                    .write()
                    .unwrap()
                    .insert(index_type, index.unwrap()); 
                
            },
            _ => {
                let err = Error::msg(format!("Unknown index type: {:?}", index_type));
                warn!("{}", err);
                // Err(err)
            }
        }
    }

    pub fn get_index(&self, index_type: IndexType) -> Option<IndexHandle> {
        self.index_map
            .read()
            .unwrap()
            .get(&index_type)
            .cloned()
    }
}

pub fn global_index_factory() -> &'static IndexFactory {
    static INDEX_FACTORY: OnceLock<IndexFactory> = OnceLock::new();
    INDEX_FACTORY.get_or_init(|| IndexFactory {
        index_map: RwLock::new(HashMap::new()),
    })
}

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn test_index_factory() {
        env_logger::Builder::new() 
            .filter_level(log::LevelFilter::Debug)
            .init();

        let index_factory = global_index_factory();
        index_factory.init(IndexType::FLAT, 128, MetricType::L2);
       
        let index = index_factory.get_index(IndexType::FLAT);
      
        index_factory.init(IndexType::UNKNOWN, 128, MetricType::L2);

        // assert!(result.is_err());
        // info!("error is {:?}", result.err().unwrap());
        let unknown_index = index_factory.get_index(IndexType::UNKNOWN);

        assert_eq!(index.is_some(), true);
        assert_eq!(unknown_index.is_none(), true);
    }
}