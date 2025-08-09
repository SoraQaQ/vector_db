use std::ops::BitOrAssign;

use anyhow::{Ok, Result, anyhow};
use dashmap::DashMap;
use log::debug;
use roaring::RoaringBitmap;

pub enum Operation {
    Equal,
    NotEqual,
}

impl Operation {
    pub fn symbol(&self) -> &'static str {
        match self {
            Self::Equal => "==",
            Self::NotEqual => "!=",
        }
    }
}

#[derive(Debug)]
pub struct FilterIndex {
    int_field_filter: DashMap<String, DashMap<i64, RoaringBitmap>>,
}

impl FilterIndex {
    pub fn new() -> Self {
        Self {
            int_field_filter: DashMap::new(),
        }
    }

    pub fn get_int_field_filter_bitmap(
        &self,
        field: String,
        op: Operation,
        value: i64,
        result_bitmap: &mut RoaringBitmap,
    ) -> Result<()> {
        let data = self
            .int_field_filter
            .get(&field)
            .ok_or_else(|| anyhow!("int_field_filter not get {}", field))?;

        debug!("get field data {:?}", data);

        match op {
            Operation::Equal => {
                if let Some(entry) = data.get(&value) {
                    result_bitmap.bitor_assign(entry.value());
                }
            }
            Operation::NotEqual => {
                for entry in data.iter() {
                    let key = entry.key();
                    if *key != value {
                        result_bitmap.bitor_assign(entry.value());
                    }
                }
            }
        }

        Ok(())
    }

    pub fn update_int_field_filter(
        &self,
        field: String,
        old_value: Option<i64>,
        new_value: i64,
        id: u32,
    ) -> Result<()> {
        if let Some(old_value) = old_value {
            debug!(
                "Updated int field filter: fieldname={}, old_value={}, new_value={}, id={}",
                field, old_value, new_value, id
            )
        } else {
            debug!(
                "Added int field filter: fieldname={}, value={}, id={}",
                field, new_value, id
            )
        }

        let field_entry = self
            .int_field_filter
            .entry(field)
            .or_insert_with(DashMap::new);

        if let Some(v) = old_value {
            if let Some(mut bitmap) = field_entry.get_mut(&v) {
                let removed = bitmap.remove(id);
                debug!("Remove old value {}: success = {}", v, removed);
            }
        }

        field_entry
            .entry(new_value)
            .or_insert_with(RoaringBitmap::new)
            .insert(id);

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_filter_index() {
        env_logger::Builder::new()
            .filter_level(log::LevelFilter::Debug)
            .init();

        let filter_index = FilterIndex::new();
        let id = 1;
        let field = "age".to_string();
        let old_value = Some(30);
        let new_value = 30;

        filter_index
            .update_int_field_filter(field.clone(), None, new_value, id)
            .unwrap();

        let mut result_bitmap = RoaringBitmap::new();

        println!("int_field_filter: {:?}", filter_index.int_field_filter);

        let result = filter_index.get_int_field_filter_bitmap(
            field.clone(),
            Operation::Equal,
            new_value,
            &mut result_bitmap,
        );

        println!("Result bitmap: {:?}", result_bitmap);
        assert!(result.is_ok());
        assert!(filter_index.int_field_filter.contains_key(&field));

        assert!(
            filter_index
                .int_field_filter
                .get(&field)
                .map(|data| data.contains_key(&new_value))
                .unwrap()
        );

        filter_index
            .update_int_field_filter(field.clone(), old_value, 45, id)
            .unwrap();

        println!("int_field_filter: {:?}", filter_index.int_field_filter);

        filter_index
            .update_int_field_filter(field.clone(), None, 45, 2)
            .unwrap();

        println!("int_field_filter: {:?}", filter_index.int_field_filter);
    }
}
