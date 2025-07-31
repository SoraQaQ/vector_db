use crate::core::{builder::index_handle::{IndexBuilder, IndexHandle}, index::hnsw_index::HnswIndex};
use anyhow::Result;
use hnsw_rs::{anndists::dist::{Distance}, hnsw::Hnsw};
use serde::{de::DeserializeOwned, Serialize};

#[derive(Default)]
pub struct HnswIndexBuilder <T: Clone + Send + Sync + Serialize + DeserializeOwned + 'static, D: Distance<T> + 'static> {
    max_nb_connection: usize,
    max_elements: usize,
    max_layer: usize,   
    ef_construction: usize,
    data: T, 
    space: D, 
}

impl<T, D> IndexBuilder for HnswIndexBuilder<T, D> 
where
    T: Clone + Send + Sync + Serialize + DeserializeOwned,
    D: Distance<T> + Send + Sync + Copy 
{
    fn build(&self) -> Result<IndexHandle> {
        let index: Hnsw<T, D> = Hnsw::new(
            self.max_nb_connection,     
            self.max_elements, 
            self.max_layer, 
            self.ef_construction, 
            self.space, 
        ); 
        
        let index = HnswIndex::new(Box::new(index)); 
        Ok(IndexHandle::new(index))
    }
}

impl<T, D> HnswIndexBuilder<T, D>
where
    T: Clone + Send + Sync + Serialize + DeserializeOwned,
    D: Distance<T> + Send + Sync + Copy,
{
    pub fn max_nb_connection(mut self, max_nb_connection: usize) -> Self {
        self.max_nb_connection = max_nb_connection;
        self
    }

    pub fn max_elements(mut self, max_elements: usize) -> Self {
        self.max_elements = max_elements;
        self
    }

    pub fn max_layer(mut self, max_layer: usize) -> Self {
        self.max_layer = max_layer;
        self
    }

    pub fn ef_construction(mut self, ef_construction: usize) -> Self {
        self.ef_construction = ef_construction;
        self
    }

    pub fn space(mut self, space: D) -> Self {
        self.space = space;
        self
    }

    pub fn data(mut self, data: T) -> Self {
        self.data = data;
        self
    }
}



#[cfg(test)]
mod tests {
    use super::*; 
    use hnsw_rs::anndists::dist::DistL2;
    
    #[test] 
    fn test_hnsw_index_builder() { 

        let builder = HnswIndexBuilder::<f32, DistL2>::default()
            .max_nb_connection(16)
            .max_elements(1000)
            .max_layer(16)
            .ef_construction(10);

        let index = builder.build();
        assert!(index.is_ok());

        let handler = index.unwrap();

        let hnsw_index = handler.downcast_ref::<HnswIndex<f32>>().unwrap();

        hnsw_index.insert_vectors(&[1.0; 10], 1);

        let (indices, distances) = hnsw_index.search_vectors(&[1.0; 10], 1, 10);
        
        assert_eq!(indices.len(), 1);
        assert_eq!(distances.len(), 1);
    }
}
