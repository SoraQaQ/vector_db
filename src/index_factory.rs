use std::{collections::HashMap, sync::{OnceLock, RwLock, Arc}};

use faiss::{Index, MetricType};
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum IndexType {
    FLAT = 0, 
    UNKNOWN = -1
}

type IndexPtr = Arc<dyn Index + Send + Sync>;

pub struct IndexFactory {
    index_map: RwLock<HashMap<IndexType, IndexPtr>>,
}

impl IndexFactory {
    pub fn init(&self, index_type: IndexType, dim: u32, metric_type: MetricType) {
        let faiss_metric = if metric_type == MetricType::L2 {
            MetricType::L2 
        } else {
            MetricType::InnerProduct
        };
        match index_type { 
            IndexType::FLAT => {
               if let Ok(index) = faiss::index_factory(dim, "IDMap,Flat", faiss_metric) {
                    self.index_map.write().unwrap().insert(index_type, Arc::new(index));
               } else {
                    eprintln!("Failed to create index: {:?}", index_type);
               }
            },
            _ => {
                eprintln!("Unknown index type: {:?}", index_type);
            }
        }
    }

    pub fn get_index(&self, index_type: IndexType) -> Option<IndexPtr> {
        self.index_map.read().unwrap().get(&index_type).cloned()
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
        let index_factory = global_index_factory();
        index_factory.init(IndexType::FLAT, 128, MetricType::L2);
        let index = index_factory.get_index(IndexType::FLAT);
      
        index_factory.init(IndexType::UNKNOWN, 128, MetricType::L2);
        let unknown_index = index_factory.get_index(IndexType::UNKNOWN);

        assert_eq!(index.is_some(), true);
        assert_eq!(unknown_index.is_none(), true);
        assert_eq!(index.as_ref().unwrap().d(), 128);
    }
}