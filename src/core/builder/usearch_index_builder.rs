use anyhow::Result;
use usearch::{Index, IndexOptions};

use crate::core::{
    builder::index_handle::{IndexBuilder, IndexHandle},
    index::usearch_index::UsearchIndex,
};

pub struct UsearchIndexBuilder {
    opt: IndexOptions,
}

impl UsearchIndexBuilder {
    pub fn new(opt: IndexOptions) -> Self {
        Self { opt }
    }
}

impl IndexBuilder for UsearchIndexBuilder {
    fn build(&self) -> Result<IndexHandle> {
        let index = UsearchIndex::new(Index::new(&self.opt).unwrap());
        Ok(IndexHandle::new(index))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new() {
        let usearch_builder = UsearchIndexBuilder::new(IndexOptions::default());
        let builder = usearch_builder.build().unwrap();

        let userach_index = builder.downcast_ref::<UsearchIndex>();
        assert!(userach_index.is_some());
    }
}
