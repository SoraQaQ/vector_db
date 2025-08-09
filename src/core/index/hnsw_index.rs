use anyhow::{Ok, Result};
use hnsw_rs::api::AnnT;
use std::sync::{Arc, Mutex};

pub struct HnswIndex<T: Clone + Send + Sync> {
    index: Arc<Mutex<Box<dyn AnnT<Val = T> + Send>>>,
}

impl<T: Clone + Send + Sync> HnswIndex<T> {
    pub fn new(index: Box<dyn AnnT<Val = T> + Send>) -> Self {
        Self {
            index: Arc::new(Mutex::new(index)),
        }
    }

    pub fn insert_vectors(&self, data: &[T], label: usize) -> Result<()> {
        self.index.lock().unwrap().insert_data(data, label);
        Ok(())
    }

    pub fn search_vectors(
        &self,
        query: &[T],
        k: usize,
        ef_s: usize,
    ) -> Result<(Vec<usize>, Vec<f32>)> {
        let result = self.index.lock().unwrap().search_neighbours(query, k, ef_s);

        let (indices, distances): (Vec<usize>, Vec<f32>) = result
            .into_iter()
            .map(|x| (x.get_origin_id(), x.get_distance()))
            .unzip();

        Ok((indices, distances))
    }

    pub fn search_vectors_filter<F>(
        &self,
        query: &[T],
        k: usize,
        ef_s: usize,
        filter: F,
    ) -> Result<(Vec<usize>, Vec<f32>)>
    where
        F: Fn(u32) -> bool,
    {
        let result = self.index.lock().unwrap().search_neighbours(query, k, ef_s);

        let (indices, distances): (Vec<usize>, Vec<f32>) = result
            .into_iter()
            .map(|x| (x.get_origin_id(), x.get_distance()))
            .unzip();

        let filtered: (Vec<usize>, Vec<f32>) = indices
            .into_iter()
            .zip(distances.into_iter())
            .filter(|(label, _)| filter(*label as u32))
            .unzip();

        Ok(filtered)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hnsw_rs::anndists::dist::DistL2;
    use roaring::RoaringBitmap;
    #[test]
    fn test_hnsw_index() {
        let index = hnsw_rs::hnsw::Hnsw::<f32, DistL2>::new(10, 100, 16, 10, DistL2 {});
        let hnsw_index = HnswIndex::new(Box::new(index));

        hnsw_index.insert_vectors(&[1.0; 10], 1).unwrap();
        hnsw_index.insert_vectors(&[2.0; 30], 2).unwrap();

        let mut bitmap = RoaringBitmap::new();
        bitmap.insert(1);

        let (indices, distances) = hnsw_index
            .search_vectors_filter(&[1.0; 10], 1, 10, |key| bitmap.contains(key))
            .unwrap();

        println!("indices: {:?}", indices);
        println!("distances: {:?}", distances);

        assert_eq!(indices.len(), 1);
        assert_eq!(distances.len(), 1);

        let (indices, distances) = hnsw_index.search_vectors(&[2.0; 10], 1, 10).unwrap();

        println!("not filter indices: {:?}", indices);
        println!("not filter distances: {:?}", distances);
    }
}
