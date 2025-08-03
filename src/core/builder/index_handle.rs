use anyhow::Result;
use std::any::Any;
use std::sync::Arc;

#[derive(Clone)]
pub struct IndexHandle {
    inner: Arc<dyn Any + Send + Sync>,
}

impl IndexHandle {
    pub fn new<T: Any + Send + Sync>(inner: T) -> Self {
        Self {
            inner: Arc::new(inner),
        }
    }

    pub fn downcast_ref<T: Any>(&self) -> Option<&T> {
        self.inner.as_ref().downcast_ref()
    }
}

pub trait IndexBuilder: Send + Sync {
    fn build(&self) -> Result<IndexHandle>;
}
