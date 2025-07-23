use std::fmt::Error;

use faiss::MetricType;

use std::sync::Arc;
use std::any::Any;

use crate::faiss_index::FaissIndex;

#[derive(Clone)]
pub struct IndexHandle {
    inner: Arc<dyn Any + Send + Sync> 
}

impl IndexHandle {
    pub fn new<T: Any + Send + Sync>(inner: T) -> Self {
        Self { inner: Arc::new(inner) }
    }

    pub fn downcast_ref<T: Any>(&self) -> Option<&T> {
        self.inner.as_ref().downcast_ref()
    }
}

pub trait IndexBuilder: Send + Sync {
    fn build(&self) -> Result<IndexHandle, Error>;
}


pub struct FaissIndexBuilder {
    descriptor: String,
    metric_type: MetricType,
    dim: u32, 
}

impl Default for FaissIndexBuilder {
    fn default() -> Self {
        Self {
            descriptor: String::new(), 
            metric_type: MetricType::L2,
            dim: 0,
        }
    }
}

impl IndexBuilder for FaissIndexBuilder {
    fn build(&self) -> Result<IndexHandle, Error> {
        let index = faiss::index_factory(self.dim, self.descriptor.as_str(), self.metric_type);
        if index.is_err() {
            return Err(Error);
        }
        let index = FaissIndex::new(Box::new(index.unwrap()));
        Ok(IndexHandle::new(index))
    }
}

impl FaissIndexBuilder {
    pub fn description(mut self, str: impl Into<String>) -> Self {
        self.descriptor = str.into();
        self
    }

    pub fn metric_type(mut self, metric_type: MetricType) -> Self {
        self.metric_type = metric_type;
        self
    }

    pub fn dim(mut self, dim: u32) -> Self {
        self.dim = dim;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*; 
    
    #[test] 
    fn test_faiss_index_builder() { 
        let builder = FaissIndexBuilder::default()
            .description("IDMap,Flat")
            .metric_type(MetricType::L2)
            .dim(128);

        let index = builder.build();
        assert!(index.is_ok());

        let handler = index.unwrap();
        let faiss_index = handler.downcast_ref::<FaissIndex>().unwrap();
        assert_eq!(faiss_index.dim(), 128);
    }
}