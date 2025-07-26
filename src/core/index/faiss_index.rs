use std::sync::Arc;
use std::sync::Mutex;

use faiss::{index::SearchResult, Idx, Index, error::Result}; 

#[derive(Clone)]
pub struct FaissIndex{
    index: Arc<Mutex<Box<dyn Index + Send>>>
}

impl FaissIndex {
    // return a new FaissIndex 
    pub fn new(
        index: Box<dyn Index + Send> 
    ) -> Self {
        Self { index: Arc::new(Mutex::new(index)) }
    }

    // Use the public constructor for Idx, if available
    pub fn insert_vectors(
        &self,
        data: &[f32],
        label: u64, 
    ) -> Result<()> {
        self.index
            .lock()
            .unwrap()
            .add_with_ids(data, &[Idx::new(label)])
    }

    // Search for the k nearest neighbors of the query vector
    pub fn search_vectors(
        &self, 
        query: &[f32],
        k: usize,   
    ) -> Result<SearchResult>{
        self.index
            .lock()
            .unwrap()
            .search(query, k)
    }

    // Get the dimension of the index
    pub fn dim(&self) -> u32 {
        self.index
            .lock()
            .unwrap()
            .d()
    }
}

#[cfg(test)]
mod tests {
    use std::thread::JoinHandle;

    use log::warn;

    use super::*;
    #[test] 
    fn test_faiss_workflow(){
        let index = faiss::index_factory(128, "IDMap,Flat", faiss::MetricType::L2).unwrap();
        let faiss_index = FaissIndex::new(Box::new(index));

        let vectors = vec![1.0;128]; 
        let label: u64 = 1;

        faiss_index.insert_vectors(&vectors, label).unwrap();

        let query = vec![1.0;128];
        let search_result = faiss_index.search_vectors(&query, 1).unwrap(); 

        assert_eq!(faiss_index.dim(), 128);
        
        assert_eq!(search_result.labels[0], Idx::new(label));
        assert!(search_result.distances[0] < 0.001);
    }

    #[test] 
    fn test_faiss_index_search() {
        env_logger::Builder::new() 
            .filter_level(log::LevelFilter::Debug)
            .init();

        let index = faiss::index_factory(128, "IDMap,Flat", faiss::MetricType::L2).unwrap();
        let faiss_index = FaissIndex::new(Box::new(index));

        let query = vec![1.0;128];
        let search_result = faiss_index.search_vectors(&query, 1);
        warn!("search_result: {:#?}", search_result);
        assert!(search_result.is_err());
    }

    #[test] 
    fn test_faiss_index_search_dim() {
        let index = faiss::index_factory(128, "IDMap,Flat", faiss::MetricType::L2).unwrap();
        let faiss_index = FaissIndex::new(Box::new(index));

        let vectors = vec![1.0;256]; 
        let label: u64 = 1;

        eprintln!("dim: {}", faiss_index.dim());
        eprintln!("Error inserting vectors: {:?}", faiss_index.insert_vectors(&vectors, label).err());

        // assert!(faiss_index.insert_vectors(&vectors, label).is_err());

        let search_result = faiss_index.search_vectors(&vec![1.0;128], 2).unwrap();

        eprintln!("search_result: {:#?}", search_result);



        // let query = vec![1.0;128];
        // let search_result = faiss_index.search_vectors(&query, 1).unwrap(); 

        // assert_eq!(faiss_index.dim(), 128);
        
        // assert_eq!(search_result.labels[0], Idx::new(label));
        // assert!(search_result.distances[0] < 0.001);
    }

    #[test] 
    fn test_concurrent_access() {
        use std::time::Duration;
        use std::thread;
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

                let query = vec![i as f32; 128];
                let search_result = index_clone.search_vectors(&query, 1).unwrap();
                
                assert_eq!(search_result.labels[0], Idx::new(label));
                assert!(search_result.distances[0] < 0.001);
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
            let search_result = faiss_index.search_vectors(&query, 1).unwrap();
            assert_eq!(search_result.labels[0], Idx::new(label));
        }

        
    }
}
