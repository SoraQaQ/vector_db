//! Faiss Index Wrapper Module
//!
//! Provides a thread-safe wrapper around Faiss indices with:
//! - Concurrent access support
//! - Filtered search capabilities
//! - Simplified error handling
use anyhow::{Ok, Result};
use faiss::MetricType;
use faiss::selector::IdSelector;
use faiss::{Idx, Index, error::Result as FaissResult};
use std::sync::Arc;
use std::sync::Mutex;

/// A thread-safe warpper around a Faiss index
///
/// This struct provides synchronized access to a Faiss index using
/// an `Arc<Mutex>` pattern for safe concurrent operations.
#[derive(Clone)]
pub struct FaissIndex {
    index: Arc<Mutex<Box<dyn Index + Send>>>,
}

impl FaissIndex {
    /// Create a new `FaissIndex` from a boxed Faiss index
    ///
    /// # Arguments
    /// * `index` - The Faiss index to wrap
    pub fn new(index: Box<dyn Index + Send>) -> Self {
        Self {
            index: Arc::new(Mutex::new(index)),
        }
    }

    /// Insert vectors into the index with the given labels
    ///
    /// # Arguments
    /// * `data` - The vectors to insert
    /// * `label` - The unique identifier for this vector
    ///
    /// # Errors
    /// Return a `faiss::error::Error` if the insertion fails
    pub fn insert_vectors(&self, data: &[f32], label: u64) -> FaissResult<()> {
        self.index
            .lock()
            .unwrap()
            .add_with_ids(data, &[Idx::new(label)])
    }

    /// Search for the k nearest neighbors of the query vector
    ///
    /// # Arguments
    /// * `query` - The query vector
    /// * `k` - The number of neighbors to return
    ///
    /// # Returns
    /// A tuple containing (labels, distances) of the nearest neighbors
    ///
    /// # Errors
    /// Returns an error if the search operation fails
    pub fn search_vectors(&self, query: &[f32], k: usize) -> Result<(Vec<Idx>, Vec<f32>)> {
        let (labels, distances): (Vec<Idx>, Vec<f32>) = self
            .index
            .lock()
            .unwrap()
            .search(query, k)
            .map(|result| (result.labels, result.distances))?;

        Ok((labels, distances))
    }

    /// Search for nearest neighbors with a filter predicate
    ///
    /// Only vectors whose labels satisfy the predicate `filter` are considered.
    ///
    /// # Arguments
    /// * `query` - The query vector
    /// * `k` - The number of neighbors to return
    /// * `filter` - Predicate function for label filtering
    ///
    /// # Errors
    /// Return a `faiss::error::Error` if the search fails
    ///
    /// # Example
    /// ```
    /// use roaring::RoaringBitmap;
    /// let mut bitmap = RoaringBitmap::new();
    /// bitmap.insert(1);
    ///
    /// let result = index.search_vectors_filter(&query, 10, |label| bitmap.contains(label));
    /// ```
    pub fn search_vectors_filter<F>(
        &self,
        query: &[f32],
        k: usize,
        filter: F,
    ) -> Result<(Vec<Idx>, Vec<f32>)>
    where
        F: Fn(u32) -> bool,
    {
        let (labels, distances): (Vec<Idx>, Vec<f32>) = self
            .index
            .lock()
            .unwrap()
            .search(query, k)
            .map(|result| (result.labels, result.distances))?;

        let filtered: (Vec<Idx>, Vec<f32>) = labels
            .into_iter()
            .zip(distances)
            .filter(|(label, _)| label.get().map(|key| filter(key as u32)).unwrap_or(false))
            .unzip();

        return Ok(filtered);
    }

    /// Get the dimension of the index
    ///
    /// # Returns
    /// Returns the dimension of the index.
    pub fn dim(&self) -> u32 {
        self.index.lock().unwrap().d()
    }

    /// Remove vectors from the faiss index
    ///
    /// # Arguments
    /// * `ids` - The ids of the vectors to remove.
    ///
    /// # Returns
    /// Returns the number of vectors removed.
    pub fn remove_vectors(&self, ids: &[u64]) -> FaissResult<usize> {
        let ids = ids.iter().map(|x| Idx::new(*x)).collect::<Vec<Idx>>();
        self.index.lock().unwrap().remove_ids(
            &IdSelector::batch(&ids)
                .map_err(|e| faiss::error::Error::from(e))
                .unwrap(),
        )
    }

    /// Get the metric type of the index
    ///
    /// # Returns
    /// Returns the metric type of the index.
    pub fn metric_type(&self) -> MetricType {
        self.index.lock().unwrap().metric_type()
    }
}

#[cfg(test)]
mod tests {
    use log::warn;
    use roaring::RoaringBitmap;
    use std::thread::JoinHandle;

    use super::*;
    #[test]
    fn test_faiss_workflow() {
        let index = faiss::index_factory(128, "IDMap,Flat", faiss::MetricType::L2).unwrap();
        let faiss_index = FaissIndex::new(Box::new(index));

        let vectors = vec![1.0; 128];
        let label: u64 = 1;

        faiss_index.insert_vectors(&vectors, label).unwrap();
        faiss_index.insert_vectors(&vectors, label + 1).unwrap();

        let mut bitmap = RoaringBitmap::new();

        bitmap.insert(1);

        let query = vec![1.0; 128];
        let (keys, distances) = faiss_index
            .search_vectors_filter(&query, 2, |key| bitmap.contains(key))
            .unwrap();

        println!("keys: {:?}, distances: {:?}", keys, distances);

        assert_eq!(faiss_index.dim(), 128);

        assert_eq!(keys[0], Idx::new(label));
        assert!(distances[0] < 0.001);

        assert_eq!(keys.len(), 1);

        let (keys, distances) = faiss_index.search_vectors(&query, 2).unwrap();
        println!("keys: {:?}, distances: {:?}", keys, distances);

        assert_eq!(keys.len(), 2);
        assert_eq!(distances.len(), 2);
    }

    #[test]
    fn test_faiss_index_search() {
        env_logger::Builder::new()
            .filter_level(log::LevelFilter::Debug)
            .init();

        let index = faiss::index_factory(128, "IDMap,Flat", faiss::MetricType::L2).unwrap();
        let faiss_index = FaissIndex::new(Box::new(index));

        let mut bitmap = RoaringBitmap::new();
        bitmap.insert(1);

        let query = vec![1.0; 128];
        let search_result = faiss_index.search_vectors(&query, 1);
        warn!("search_result: {:#?}", search_result);
        assert!(search_result.is_ok());
    }

    #[test]
    fn test_faiss_index_search_dim() {
        let index = faiss::index_factory(128, "IDMap,Flat", faiss::MetricType::L2).unwrap();
        let faiss_index = FaissIndex::new(Box::new(index));

        let vectors = vec![1.0; 256];
        let label: u64 = 1;

        eprintln!("dim: {}", faiss_index.dim());
        eprintln!(
            "Error inserting vectors: {:?}",
            faiss_index.insert_vectors(&vectors, label).err()
        );

        // assert!(faiss_index.insert_vectors(&vectors, label).is_err());

        let mut bitmap = RoaringBitmap::new();
        bitmap.insert(1);
        let search_result = faiss_index.search_vectors(&vec![1.0; 128], 2).unwrap();

        eprintln!("search_result: {:#?}", search_result);

        // let query = vec![1.0;128];
        // let search_result = faiss_index.search_vectors(&query, 1).unwrap();

        // assert_eq!(faiss_index.dim(), 128);

        // assert_eq!(search_result.labels[0], Idx::new(label));
        // assert!(search_result.distances[0] < 0.001);
    }

    #[test]
    fn test_concurrent_access() {
        use std::thread;
        use std::time::Duration;
        let index = faiss::index_factory(128, "IDMap,Flat", faiss::MetricType::L2).unwrap();
        let faiss_index = FaissIndex::new(Box::new(index));

        let mut handles: Vec<JoinHandle<u64>> = vec![];

        for i in 0..10 {
            let index_clone = faiss_index.clone();
            let handle = thread::spawn(move || {
                thread::sleep(Duration::from_millis(i * 10));

                let vectors = vec![i as f32; 128];
                let label = i as u64 + 1;

                index_clone.insert_vectors(&vectors, label).unwrap();

                let mut bitmap = RoaringBitmap::new();
                bitmap.insert(label as u32);

                let query = vec![i as f32; 128];
                let search_result = index_clone.search_vectors(&query, 1).unwrap();

                assert_eq!(search_result.0[0], Idx::new(label));
                assert!(search_result.1[0] < 0.001);
                label
            });
            handles.push(handle);
        }

        let mut result: Vec<u64> = vec![];
        for handle in handles {
            result.push(handle.join().unwrap());
        }

        assert_eq!(result.len(), 10);

        for (i, &label) in result.iter().enumerate() {
            let query = vec![i as f32; 128];
            let mut bitmap = RoaringBitmap::new();
            bitmap.insert(label as u32);
            let search_result = faiss_index.search_vectors(&query, 1).unwrap();
            assert_eq!(search_result.0[0], Idx::new(label));
        }
    }
}
