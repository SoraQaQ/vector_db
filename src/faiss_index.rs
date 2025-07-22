use faiss::{index::SearchResult, Idx, Index, error::Result}; 

pub struct FaissIndex{
    index: Box<dyn Index>
}

impl FaissIndex {
    // return a new FaissIndex 
    pub fn new(
        index: Box<dyn Index> 
    ) -> Self {
        Self { index }
    }

    
    pub fn insert_vectors(
        &mut self,
        data: &[f32],
        label: Idx, 
    ) -> Result<()> {
        
        // Use the public constructor for Idx, if available
        self.index.add_with_ids(data, &[label])
    }

    pub fn search_vectors(
        &mut self, 
        query: &[f32],
        k: usize, 
    ) -> Result<SearchResult>{
       self.index.search(query, k)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test] 
    fn test_faiss_workflow(){

        let index = faiss::index_factory(128, "IDMap,Flat", faiss::MetricType::L2).unwrap();
        let mut faiss_index = FaissIndex::new(Box::new(index));

        let vectors = vec![1.0;128]; 
        let label: Idx = Idx::new(1);

        faiss_index.insert_vectors(&vectors, label).unwrap();

        let query = vec![1.0;128];
        let search_result = faiss_index.search_vectors(&query, 1).unwrap();
        
        println!("search_result: {:#?}", search_result);
    
        assert_eq!(search_result.labels[0], label);
        assert!(search_result.distances[0] < 0.001);

    }
}
