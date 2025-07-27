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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Deserialize, Serialize)]
pub struct IndexKey {
    pub index_type: IndexType,
    pub dim: u32,
    pub metric_type: MyMetricType,
}

impl fmt::Display for IndexKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({}, {}, {})", self.index_type, self.dim, self.metric_type)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Deserialize, Serialize)]
pub enum MyMetricType {
    /// Inner product, also called cosine distance
    InnerProduct = 0,
    /// Euclidean L2-distance
    L2 = 1,
}

impl fmt::Display for MyMetricType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MyMetricType::InnerProduct => write!(f, "INNER_PRODUCT"),
            MyMetricType::L2 => write!(f, "L2"),
        }
    }
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
    index_map: RwLock<HashMap<IndexKey, IndexHandle>>,
}

impl IndexFactory {
    pub fn init(&self, index_type: IndexType, dim: u32, metric_type: MyMetricType) -> Result<(), Error> {
        info!("init index: {:?}", index_type);
        match index_type { 
            IndexType::FLAT => {
                let faiss_metric = match metric_type {
                    MyMetricType::InnerProduct => MetricType::InnerProduct,
                    MyMetricType::L2 => MetricType::L2,
                };
                let builder = FaissIndexBuilder::default()
                    .dim(dim)
                    .description("IDMap,Flat")
                    .metric_type(faiss_metric);

                let index = builder.build();
                
                self.index_map
                    .write()
                    .unwrap()
                    .insert(IndexKey { index_type, dim, metric_type }, index.unwrap()); 

                Ok(())
            },
            _ => {
                let err = Error::msg(format!("Unknown index type: {:?}", index_type));
                warn!("{}", err);
                Err(err)
            }
        }
    }

    pub fn get_index(&self, index_key: IndexKey) -> Option<IndexHandle> {
        self.index_map
            .read()
            .unwrap()
            .get(&index_key)
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

    use crate::core::index::faiss_index::FaissIndex;

    use super::*;

    #[test]
    fn test_index_factory() {
        env_logger::Builder::new() 
            .filter_level(log::LevelFilter::Debug)
            .init();

        let index_factory = global_index_factory();
        index_factory.init(IndexType::FLAT, 128, MyMetricType::L2).unwrap();

        index_factory.init(IndexType::FLAT, 256, MyMetricType::L2).unwrap();

        index_factory.init(IndexType::FLAT, 10, MyMetricType::InnerProduct).unwrap();
       
        let index = index_factory.get_index(IndexKey { index_type: IndexType::FLAT, dim: 256, metric_type: MyMetricType::L2 });

        assert_eq!(index.unwrap().downcast_ref::<FaissIndex>().unwrap().dim(), 256);

        let index = index_factory.get_index(IndexKey { index_type: IndexType::FLAT, dim: 128, metric_type: MyMetricType::L2 });

        assert_eq!(index.unwrap().downcast_ref::<FaissIndex>().unwrap().dim(), 128);

        let index = index_factory.get_index(IndexKey { index_type: IndexType::FLAT, dim: 10, metric_type: MyMetricType::InnerProduct });

        assert_eq!(index.unwrap().downcast_ref::<FaissIndex>().unwrap().metric_type(), MetricType::InnerProduct);
      
        index_factory.init(IndexType::UNKNOWN, 128, MyMetricType::L2).unwrap();

        // assert!(result.is_err());
        // info!("error is {:?}", result.err().unwrap());
        let unknown_index = index_factory.get_index(IndexKey { index_type: IndexType::UNKNOWN, dim: 128, metric_type: MyMetricType::L2 });

        assert_eq!(unknown_index.is_none(), true);
    }
}