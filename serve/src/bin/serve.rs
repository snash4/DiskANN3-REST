//! REST server for a DiskANN3 disk index.
//!
//! Usage:
//!     serve <serve.json>
//!
//! Example serve.json:
//! {
//!   "load_path": "sample_index_l50_r32",
//!   "data_type": "Float32",
//!   "distance": "SquaredL2",
//!   "addr": "0.0.0.0:8080",
//!   "workers": 8,
//!   "queue_capacity": 256,
//!   "num_nodes_to_cache": null,
//!   "search_io_limit": null,
//!   "defaults": { "k": 10, "search_list": 100, "beam_width": 4, "flat": false }
//! }

use std::env;

use anyhow::{bail, Context, Result};
use diskann_providers::storage::FileStorageProvider;

use DiskANN3_REST::inputs::DataType;
use DiskANN3_REST::loader::load_searcher;
use DiskANN3_REST::serve::{self, ServeConfig};

fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();
    let cfg_path = args.get(1).cloned().unwrap_or_else(|| {
        eprintln!("usage: {} <serve.json>", args[0]);
        std::process::exit(2);
    });

    let json =
        std::fs::read_to_string(&cfg_path).with_context(|| format!("reading {cfg_path}"))?;
    let config: ServeConfig =
        serde_json::from_str(&json).with_context(|| format!("parsing {cfg_path}"))?;

    let storage = FileStorageProvider;

    // workers is also the searcher's thread-hint (the max in-flight query count).
    let loaded = match config.data_type {
        DataType::Float32 => load_searcher::<f32, _>(
            &config.load_path,
            config.distance,
            config.workers,
            config.search_io_limit,
            config.num_nodes_to_cache,
            &storage,
        )?,
        other => bail!("the server only implements Float32 (got {other:?})"),
    };

    println!(
        "loaded index '{}': {} points, dim {}",
        config.load_path, loaded.num_points, loaded.dims
    );

    serve::run(loaded, &config)
}