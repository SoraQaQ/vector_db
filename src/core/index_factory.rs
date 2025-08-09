use crate::core::builder::{
    faiss_index_builder::FaissIndexBuilder,
    hnsw_index_builder::HnswIndexBuilder,
    index_handle::{IndexBuilder, IndexHandle},
    usearch_index_builder::UsearchIndexBuilder,
};
use anyhow::{Result, anyhow};
use dashmap::DashMap;
use faiss::MetricType as FaissMetricType;
use hnsw_rs::anndists::dist::DistL2;
use log::{debug, info, warn};
use serde::{Deserialize, Serialize};
use std::{fmt, sync::OnceLock};
use usearch::{IndexOptions, MetricKind};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Deserialize, Serialize)]
pub enum IndexType {
    FLAT = 0,
    HNSW = 1,
    UNKNOWN = -1,
    USEARCH = 3,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Deserialize, Serialize)]
pub struct IndexKey {
    pub index_type: IndexType,
    pub dim: u32,
    pub metric_type: MetricType,
}

impl fmt::Display for IndexKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "({}, {}, {})",
            self.index_type, self.dim, self.metric_type
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Deserialize, Serialize, Default)]
pub enum MetricType {
    /// Inner product, also called cosine distance
    InnerProduct = 0,
    /// Euclidean L2-distance
    #[default]
    L2 = 1,
}

impl fmt::Display for MetricType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MetricType::InnerProduct => write!(f, "INNER_PRODUCT"),
            MetricType::L2 => write!(f, "L2"),
        }
    }
}

impl fmt::Display for IndexType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            IndexType::FLAT => write!(f, "FLAT"),
            IndexType::HNSW => write!(f, "HNSW"),
            IndexType::USEARCH => write!(f, "USEARCH"),
            IndexType::UNKNOWN => write!(f, "UNKNOWN"),
        }
    }
}

pub struct IndexFactory {
    index_map: DashMap<IndexKey, IndexHandle>,
}

impl IndexFactory {
    pub fn init(
        &self,
        index_type: IndexType,
        dim: u32,
        max_elements: usize,
        metric_type: MetricType,
        mut usearch_options: IndexOptions,
    ) -> Result<()> {
        info!("init index: {:?}", index_type);
        match index_type {
            IndexType::FLAT => {
                let faiss_metric = match metric_type {
                    MetricType::InnerProduct => FaissMetricType::InnerProduct,
                    MetricType::L2 => FaissMetricType::L2,
                };
                let builder = FaissIndexBuilder::default()
                    .dim(dim)
                    .description("IDMap,Flat")
                    .metric_type(faiss_metric);

                let index = builder.build().unwrap();

                self.index_map.insert(
                    IndexKey {
                        index_type,
                        dim,
                        metric_type,
                    },
                    index,
                );

                Ok(())
            }
            IndexType::HNSW => match metric_type {
                MetricType::L2 => {
                    let builder = HnswIndexBuilder::<f32, DistL2>::default()
                        .max_nb_connection(16)
                        .max_elements(max_elements)
                        .max_layer(16)
                        .ef_construction(200);

                    let index = builder.build().unwrap();

                    self.index_map.insert(
                        IndexKey {
                            index_type,
                            dim,
                            metric_type,
                        },
                        index,
                    );

                    Ok(())
                }

                _ => Err(anyhow!("Unknown metric type: {:?}", metric_type)),
            },
            IndexType::USEARCH => {
                match metric_type {
                    MetricType::InnerProduct => {
                        usearch_options.metric = MetricKind::IP;
                    }
                    MetricType::L2 => {
                        usearch_options.metric = MetricKind::L2sq;
                    }
                }
                usearch_options.dimensions = dim as usize;
                let builder = UsearchIndexBuilder::new(usearch_options);
                let index = builder.build().unwrap();

                let index_key = IndexKey {
                    index_type: index_type,
                    dim: dim,
                    metric_type: metric_type,
                };

                self.index_map.insert(index_key, index);

                debug!("index_key: {:?}", index_key);

                Ok(())
            }
            _ => {
                let err = anyhow!("Unknown index type: {:?}", index_type);
                warn!("{}", err);
                Err(err)
            }
        }
    }

    pub fn get_index(&self, index_key: IndexKey) -> Option<IndexHandle> {
        self.index_map.get(&index_key).map(|v| v.clone())
    }
}

pub fn global_index_factory() -> &'static IndexFactory {
    static INDEX_FACTORY: OnceLock<IndexFactory> = OnceLock::new();
    INDEX_FACTORY.get_or_init(|| IndexFactory {
        index_map: DashMap::new(),
    })
}

#[cfg(test)]
mod tests {

    use usearch::{MetricKind, ScalarKind};

    use crate::core::index::{faiss_index::FaissIndex, usearch_index::UsearchIndex};

    use super::*;

    #[test]
    fn test_index_factory() {
        env_logger::Builder::new()
            .filter_level(log::LevelFilter::Debug)
            .init();

        let opt = IndexOptions {
            dimensions: 3,                 // necessary for most metric kinds
            metric: MetricKind::IP,        // or ::L2sq, ::Cos ...
            quantization: ScalarKind::F32, // or ::F32, ::F16, ::I8, ::B1x8 ...
            connectivity: 0,               // zero for auto
            expansion_add: 0,              // zero for auto
            expansion_search: 0,           // zero for auto
            multi: false,
        };

        let index_factory = global_index_factory();
        index_factory
            .init(IndexType::FLAT, 128, 1000, MetricType::L2, opt.clone())
            .unwrap();

        index_factory
            .init(IndexType::FLAT, 256, 1000, MetricType::L2, opt.clone())
            .unwrap();

        index_factory
            .init(
                IndexType::FLAT,
                10,
                1000,
                MetricType::InnerProduct,
                opt.clone(),
            )
            .unwrap();

        let index = index_factory.get_index(IndexKey {
            index_type: IndexType::FLAT,
            dim: 256,
            metric_type: MetricType::L2,
        });

        assert_eq!(
            index.unwrap().downcast_ref::<FaissIndex>().unwrap().dim(),
            256
        );

        let index = index_factory.get_index(IndexKey {
            index_type: IndexType::FLAT,
            dim: 128,
            metric_type: MetricType::L2,
        });

        assert_eq!(
            index.unwrap().downcast_ref::<FaissIndex>().unwrap().dim(),
            128
        );

        let index = index_factory.get_index(IndexKey {
            index_type: IndexType::FLAT,
            dim: 10,
            metric_type: MetricType::InnerProduct,
        });

        assert_eq!(
            index
                .unwrap()
                .downcast_ref::<FaissIndex>()
                .unwrap()
                .metric_type(),
            FaissMetricType::InnerProduct
        );

        let result = index_factory.init(IndexType::UNKNOWN, 128, 1000, MetricType::L2, opt.clone());
        assert!(result.is_err());

        index_factory
            .init(IndexType::USEARCH, 128, 1000, MetricType::L2, opt.clone())
            .unwrap();

        let index = index_factory.get_index(IndexKey {
            index_type: IndexType::USEARCH,
            dim: 128,
            metric_type: MetricType::L2,
        });

        debug!("usearch index: {:?}", index);

        assert_eq!(
            index.unwrap().downcast_ref::<UsearchIndex>().unwrap().dim(),
            128
        );
    }
}
