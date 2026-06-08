/*
 * `loader`: builds a `DiskIndexSearcher` from a disk-index prefix and reports
 * the index metadata (point count, vector dimension, metric) the server needs.
 */

use anyhow::{Context, Result};

use diskann::utils::VectorRepr;
use diskann_disk::{
    data_model::{AdHoc, CachingStrategy},
    search::{
        provider::{
            disk_provider::DiskIndexSearcher,
            disk_vertex_provider_factory::DiskVertexProviderFactory,
        },
        traits::VertexProviderFactory, // brings get_header() into scope
    },
    storage::disk_index_reader::DiskIndexReader,
    utils::AlignedFileReaderFactory,
};
use diskann_providers::storage::{
    get_compressed_pq_file, get_disk_index_file, get_pq_pivot_file, StorageReadProvider,
};
use diskann_vector::distance::Metric;

use crate::inputs::SimilarityMeasure;

/// Concrete searcher type for a disk index of `T` vectors with u32 ids.
pub type Searcher<T> =
    DiskIndexSearcher<AdHoc<T>, DiskVertexProviderFactory<AdHoc<T>, AlignedFileReaderFactory>>;

/// A loaded, ready-to-query index plus the metadata needed for serving.
pub struct LoadedIndex<T: VectorRepr + bytemuck::Pod> {
    pub searcher: Searcher<T>,
    pub num_points: usize,
    pub dims: usize,
    pub metric: Metric,
}

/// Resolve the three index files from `load_path`, load PQ data, wire the
/// on-disk graph reader, and construct the searcher. `num_threads` is the
/// searcher's concurrency hint (size it to the max number of in-flight queries).
pub fn load_searcher<T, S>(
    load_path: &str,
    distance: SimilarityMeasure,
    num_threads: usize,
    search_io_limit: Option<usize>,
    num_nodes_to_cache: Option<usize>,
    storage: &S,
) -> Result<LoadedIndex<T>>
where
    T: VectorRepr + bytemuck::Pod,
    S: StorageReadProvider,
{
    let pivot_path = get_pq_pivot_file(load_path);
    let pq_data_path = get_compressed_pq_file(load_path);
    let disk_index_path = get_disk_index_file(load_path);

    let index_reader = DiskIndexReader::<T>::new(pivot_path, pq_data_path, storage)
        .context("failed to load PQ pivots / compressed data")?;
    let num_points = index_reader.get_num_points();

    let caching_strategy = match num_nodes_to_cache {
        Some(n) => CachingStrategy::StaticCacheWithBfsNodes(n),
        None => CachingStrategy::None,
    };
    let reader_factory = AlignedFileReaderFactory::new(disk_index_path);
    let vertex_provider_factory = DiskVertexProviderFactory::new(reader_factory, caching_strategy)
        .context("failed to create disk vertex provider factory")?;

    // Read the graph header once for the vector dimension (used to validate
    // incoming query lengths before search).
    let dims = vertex_provider_factory
        .get_header()
        .map_err(|e| anyhow::anyhow!("reading graph header: {e}"))?
        .metadata()
        .dims;

    let metric: Metric = distance.into();
    let searcher = DiskIndexSearcher::<AdHoc<T>, _>::new(
        num_threads,
        search_io_limit.unwrap_or(usize::MAX),
        &index_reader,
        vertex_provider_factory,
        metric,
        None,
    )
    .context("failed to construct DiskIndexSearcher")?;

    Ok(LoadedIndex {
        searcher,
        num_points,
        dims,
        metric,
    })
}