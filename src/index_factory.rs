use std::{collections::HashMap, io::Error, sync::{OnceLock, RwLock}};
use faiss::MetricType;
use log::{warn, info};

use crate::{index_builder::{FaissIndexBuilder, IndexBuilder, IndexHandle}};
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum IndexType {
    FLAT = 0, 
    UNKNOWN = -1
}

pub struct IndexFactory {
    index_map: RwLock<HashMap<IndexType, IndexHandle>>,
}

impl IndexFactory {
    pub fn init(&self, index_type: IndexType, dim: u32, metric_type: MetricType) -> Result<(), Error> {
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
                
                Ok(())
            },
            _ => {
                let err_str = format!("Unknown index type: {:?}", index_type);
                warn!("{}", err_str);
                Err(Error::new(std::io::ErrorKind::NotFound, err_str) )
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
        index_factory.init(IndexType::FLAT, 128, MetricType::L2).expect("Failed to init FLAT index");
       
        let index = index_factory.get_index(IndexType::FLAT);
      
        let result = index_factory.init(IndexType::UNKNOWN, 128, MetricType::L2);

        assert!(result.is_err());
        let unknown_index = index_factory.get_index(IndexType::UNKNOWN);

        assert_eq!(index.is_some(), true);
        assert_eq!(unknown_index.is_none(), true);
    }
}