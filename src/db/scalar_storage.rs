use anyhow::Result;
use rocksdb::DB;
pub struct ScalarStorage {
    pub db: DB,
}

impl ScalarStorage {
    pub fn insert_scalar(&self, id: u64, data: serde_json::Value) -> Result<()> {
        let data = serde_json::to_string(&data)?;
        self.db.put(id.to_string(), data)?;
        Ok(())
    }

    pub fn get_scalar(&self, id: u64) -> Option<serde_json::Value> {
        let id = id.to_string();

        self.db.get(&id).ok()?.and_then(|bytes| {
            std::str::from_utf8(&bytes)
                .ok()
                .and_then(|s| serde_json::from_str(s).ok())
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use tempfile::TempDir;

    #[test]
    fn test_scalar_storage() {
        let temp_dir = TempDir::new().unwrap();
        let db = DB::open_default(temp_dir.path()).unwrap();
        let scalar_storage = ScalarStorage { db };
        let data = json!({"name": "sora", "age": 20});
        scalar_storage.insert_scalar(1, data).unwrap();
        let data = scalar_storage.get_scalar(1).unwrap();
        assert_eq!(data, json!({"name": "sora", "age": 20}));
    }
}
