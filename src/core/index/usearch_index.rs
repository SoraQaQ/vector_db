use anyhow::{Ok, Result, anyhow};
use usearch::{Index, Key};

pub struct UsearchIndex {
    index: Index,
}

impl UsearchIndex {
    pub fn new(index: Index) -> Self {
        Self { index: index }
    }

    pub fn insert_vectors(&self, label: u64, data: &[f32]) -> Result<()> {
        self.index
            .add(label, data)
            .map_err(|e| anyhow!("insert error: {e}"))
    }

    pub fn filter_exact_search<F>(
        &self,
        query: &[f32],
        count: usize,
        filter: F,
    ) -> Result<(Vec<u64>, Vec<f32>)>
    where
        F: Fn(Key) -> bool,
    {
        let (keys, distances) = self
            .index
            .exact_search(query, count)
            .map(|matches| {
                let labels = matches.keys;
                let distances = matches.distances;
                (labels, distances)
            })
            .map_err(|e| anyhow!("filter_exact_search err: {e}"))?;

        let filtered: (Vec<u64>, Vec<f32>) = keys
            .into_iter()
            .zip(distances.into_iter())
            .filter(|(label, _)| filter(*label))
            .take(count)
            .unzip();

        Ok(filtered)
    }

    pub fn exact_search(&self, query: &[f32], count: usize) -> Result<(Vec<u64>, Vec<f32>)> {
        let result = self
            .index
            .exact_search(query, count)
            .map(|matches| (matches.keys, matches.distances))
            .map_err(|e| anyhow!("exact_search err: {e}"))?;

        Ok(result)
    }

    pub fn search(&self, query: &[f32], count: usize) -> Result<(Vec<u64>, Vec<f32>)> {
        let result = self
            .index
            .search(query, count)
            .map(|matches| (matches.keys, matches.distances))
            .map_err(|e| anyhow!("search err: {e}"))?;

        Ok(result)
    }

    pub fn filtered_search<F>(
        &self,
        query: &[f32],
        count: usize,
        filter: F,
    ) -> Result<(Vec<u64>, Vec<f32>)>
    where
        F: Fn(Key) -> bool,
    {
        self.index
            .filtered_search(query, count, filter)
            .map(|matches| {
                let labels = matches.keys;
                let distances = matches.distances;
                (labels, distances)
            })
            .map_err(|e| anyhow!("filtered_search error: {e}"))
    }

    pub fn remove(&self, label: u64) -> Result<()> {
        self.index
            .remove(label)
            .map_err(|e| anyhow!("usearch remove error: {e}"))?;

        Ok(())
    }

    pub fn reserve(&self, size: usize) -> Result<()> {
        self.index
            .reserve(size)
            .map_err(|e| anyhow!("usearch reserve error: {e}"))?;

        Ok(())
    }

    pub fn dim(&self) -> usize {
        self.index.dimensions()
    }
}

#[cfg(test)]
mod tests {
    use roaring::RoaringBitmap;
    use usearch::{IndexOptions, MetricKind, ScalarKind};

    use super::*;

    #[test]
    fn test_search() {
        let index = UsearchIndex::new(
            Index::new(&IndexOptions {
                dimensions: 3,                  // necessary for most metric kinds
                metric: MetricKind::IP,         // or ::L2sq, ::Cos ...
                quantization: ScalarKind::BF16, // or ::F32, ::F16, ::I8, ::B1x8 ...
                connectivity: 0,                // zero for auto
                expansion_add: 0,               // zero for auto
                expansion_search: 0,            // zero for auto
                multi: false,
            })
            .unwrap(),
        );

        let first: [f32; 3] = [0.2, 0.1, 0.2];
        let second: [f32; 3] = [0.2, 0.1, 0.2];

        assert!(index.reserve(10).is_ok());

        assert!(index.insert_vectors(1, &first).is_ok());
        assert!(index.insert_vectors(2, &second).is_ok());

        let query = [0.2, 0.1, 0.2];
        let result = index.search(&query, 10).unwrap();

        eprintln!("result: {:?}", result);

        assert_eq!(result.0.len(), 2);
    }

    #[test]
    fn test_filtered_search() {
        let index = UsearchIndex::new(
            Index::new(&IndexOptions {
                dimensions: 3,                  // necessary for most metric kinds
                metric: MetricKind::IP,         // or ::L2sq, ::Cos ...
                quantization: ScalarKind::BF16, // or ::F32, ::F16, ::I8, ::B1x8 ...
                connectivity: 0,                // zero for auto
                expansion_add: 0,               // zero for auto
                expansion_search: 0,            // zero for auto
                multi: false,
            })
            .unwrap(),
        );

        let first: [f32; 3] = [0.2, 0.1, 0.2];
        let second: [f32; 3] = [0.2, 0.1, 0.2];

        assert!(index.reserve(10).is_ok());

        assert!(index.insert_vectors(1, &first).is_ok());
        assert!(index.insert_vectors(2, &second).is_ok());

        let query = [0.2, 0.1, 0.2];

        let mut bitmap = RoaringBitmap::new();

        bitmap.insert(1);

        let result = index
            .filtered_search(&query, 10, |f| bitmap.contains(f.try_into().unwrap()))
            .unwrap();

        eprintln!("result: {:?}", result);

        assert_eq!(result.0.len(), 1);
    }

    #[test]
    fn test_exact_search_filtered() {
        let index = UsearchIndex::new(
            Index::new(&IndexOptions {
                dimensions: 3,                  // necessary for most metric kinds
                metric: MetricKind::IP,         // or ::L2sq, ::Cos ...
                quantization: ScalarKind::BF16, // or ::F32, ::F16, ::I8, ::B1x8 ...
                connectivity: 0,                // zero for auto
                expansion_add: 0,               // zero for auto
                expansion_search: 0,            // zero for auto
                multi: false,
            })
            .unwrap(),
        );

        let first: [f32; 3] = [0.2, 0.1, 0.2];
        let second: [f32; 3] = [0.2, 0.1, 0.2];

        assert!(index.reserve(10).is_ok());

        assert!(index.insert_vectors(1, &first).is_ok());
        assert!(index.insert_vectors(2, &second).is_ok());

        let query = [0.2, 0.1, 0.2];

        let mut bitmap = RoaringBitmap::new();

        bitmap.insert(1);

        let result = index
            .filter_exact_search(&query, 10, |f| bitmap.contains(f.try_into().unwrap()))
            .unwrap();

        println!("result: {:?}", result);

        assert_eq!(result.0.len(), 1);
    }

    #[test]
    fn test_remove() {
        let index = UsearchIndex::new(
            Index::new(&IndexOptions {
                dimensions: 3,                  // necessary for most metric kinds
                metric: MetricKind::IP,         // or ::L2sq, ::Cos ...
                quantization: ScalarKind::BF16, // or ::F32, ::F16, ::I8, ::B1x8 ...
                connectivity: 0,                // zero for auto
                expansion_add: 0,               // zero for auto
                expansion_search: 0,            // zero for auto
                multi: false,
            })
            .unwrap(),
        );

        let first: [f32; 3] = [0.2, 0.1, 0.2];
        let second: [f32; 3] = [0.2, 0.1, 0.2];

        assert!(index.reserve(10).is_ok());

        assert!(index.insert_vectors(1, &first).is_ok());
        assert!(index.insert_vectors(2, &second).is_ok());

        let query = [0.2, 0.1, 0.2];

        let mut bitmap = RoaringBitmap::new();

        bitmap.insert(1);

        let r = index.remove(1);
        let result = index.exact_search(&query, 10).unwrap();

        println!("remove result: {:?}", r);
        println!("result: {:?}", result);

        assert_eq!(result.0.len(), 1);
    }
}
