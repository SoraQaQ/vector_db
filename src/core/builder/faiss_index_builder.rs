use faiss::MetricType;

use crate::core::{
    builder::index_handle::{IndexBuilder, IndexHandle},
    index::faiss_index::FaissIndex,
};

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
    fn build(&self) -> anyhow::Result<IndexHandle> {
        let index = faiss::index_factory(self.dim, self.descriptor.as_str(), self.metric_type)
            .expect("failed to create index");

        let index = FaissIndex::new(Box::new(index));

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
