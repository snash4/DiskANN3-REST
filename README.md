# DiskANN3 REST API

A small REST service for approximate nearest-neighbor (ANN) search over a [DiskANN3](https://github.com/microsoft/DiskANN) **disk index**, plus a HTTP client for querying and load-testing it.

## Overview

The project is a Cargo workspace with two crates:

| Crate | Binary | Purpose | Toolchain |
|-------|--------|---------|-----------|
| `serve` (`diskann3-serve`) | `serve` | Loads a disk index and answers queries over HTTP | DiskANN3 deps |
| `client` (`diskann3-client`) | `diskann3-client` | Reads `.fbin` queries and calls the API | stable Rust |

The client has **no DiskANN dependency** — it only speaks HTTP + JSON and parses the `.fbin` header itself, so it builds in seconds and runs anywhere.

## How it works

- **Load once, share everywhere.** The disk index is opened at startup into a single searcher that all request threads share (it is read-only at serve time).
- **Synchronous server.** `serve` uses `tiny_http` with a fixed worker pool. This is deliberate: DiskANN3's `search()` is blocking and internally drives its own current-thread Tokio runtime, which would panic inside an async handler ("cannot start a runtime from within a runtime"). On plain OS worker threads it just works.
- **Concurrency is across requests, not within a query.** One query is handled by one thread, overlapping its own disk reads via beam search. Throughput comes from many queries running at once.
- **Backpressure.** An acceptor thread feeds a bounded queue; when it is full the server returns `503` immediately instead of letting latency grow unbounded.
- **Horizontally scalable.** Because the index is read-only, replicas are identical and stateless — put N behind a load balancer and throughput scales near-linearly. Each replica needs its own copy of the index files.

## Project layout

```
diskann3/
├── Cargo.toml          # workspace: members = ["serve", "client"]
├── serve/
│   ├── Cargo.toml
│   └── src/
│       ├── lib.rs
│       ├── inputs.rs   # DataType, SimilarityMeasure
│       ├── loader.rs   # load_searcher() + index metadata
│       ├── serve.rs    # HTTP server, routes, worker pool, backpressure
│       └── bin/serve.rs
└── client/
    ├── Cargo.toml
    └── src/main.rs
```

## Requirements

- **Server:** Rust **1.92** (DiskANN3 pins this toolchain) and network access to fetch the `diskann*` git crates. On Linux the disk reader uses `io_uring`, so a reasonably modern kernel is recommended for running search.
- **Client:** stable Rust.
- **A prebuilt disk index.** This service only *loads and serves* an index; it does not build one. An index is a set of three files sharing a prefix:
  - `<prefix>_disk.index` — the graph + full-precision vectors (on disk)
  - `<prefix>_pq_pivots.bin` — PQ pivot table (loaded to RAM)
  - `<prefix>_pq_compressed.bin` — PQ-compressed vectors (loaded to RAM)

## Build

```bash
cd diskann3
cargo build --release                    # both crates
cargo build --release -p diskann3-serve  # server only (needs Rust 1.92)
cargo build --release -p diskann3-client # client only (stable, no DiskANN)
```

## Running the server

The server is configured by a JSON file:

```json
{
  "load_path": "sample_index_l50_r32",
  "data_type": "Float32",
  "distance": "SquaredL2",
  "addr": "0.0.0.0:8080",
  "workers": 8,
  "queue_capacity": 256,
  "num_nodes_to_cache": null,
  "search_io_limit": null,
  "defaults": { "k": 10, "search_list": 100, "beam_width": 4, "flat": false }
}
```

```bash
cargo run --release -p diskann3-serve --bin serve -- serve.json
```

### Configuration fields

| Field | Type | Default | Notes |
|-------|------|---------|-------|
| `load_path` | string | — | Index **prefix** (no `_disk.index` suffix). Required. |
| `data_type` | string | — | `Float32` (only type implemented). Required. |
| `distance` | string | — | `SquaredL2` \| `InnerProduct` \| `Cosine` \| `CosineNormalized`. Must match how the index was built. |
| `addr` | string | `0.0.0.0:8080` | Bind address. |
| `workers` | int | `8` | Worker threads = max concurrent in-flight queries. |
| `queue_capacity` | int | `256` | Bounded request queue; overflow → `503`. |
| `num_nodes_to_cache` | int \| null | `null` | Cache this many hot graph nodes in RAM (fewer disk reads). |
| `search_io_limit` | int \| null | `null` | Cap on IOs per query (`null` = unbounded). |
| `defaults.k` | int | `10` | Default neighbors to return. |
| `defaults.search_list` | int | `100` | Default search-list size `L` (must be ≥ `k`). |
| `defaults.beam_width` | int | `4` | Default beam width. |
| `defaults.flat` | bool | `false` | Default to brute-force search. |

## API reference

All responses are `Content-Type: application/json`. Errors use the shape
`{"error": "<message>"}`.

### `POST /search`

Search a single query vector.

Request:

```json
{
  "query": [0.12, -0.03, 0.88, "..."],
  "k": 10,
  "search_list": 100,
  "beam_width": 4,
  "flat": false
}
```

Only `query` is required; `k`, `search_list` `beam_width`, and `flat` fall back to the server's configured defaults when omitted.

Response (`200`):

```json
{
  "neighbors": [
    { "id": 4823, "distance": 0.142 },
    { "id": 91,   "distance": 0.155 }
  ],
  "comparisons": 312
}
```

`comparisons` is the number of distance computations the search performed (a
useful work/cost signal).

### `POST /search/batch`

Search many query vectors in one request.

Request:

```json
{
  "queries": [[0.1, 0.2, "..."], [0.3, 0.4, "..."]],
  "k": 10,
  "search_list": 100
}
```

Response (`200`):

```json
{ "results": [ { "neighbors": [ "..." ], "comparisons": 312 } ] }
```

Queries in a batch are searched sequentially within the request; concurrency comes from issuing multiple requests at once.

### `GET /info`

Index metadata and configured defaults.

```json
{
  "num_points": 1000000,
  "dim": 128,
  "metric": "L2",
  "defaults": { "k": 10, "search_list": 100, "beam_width": 4, "flat": false }
}
```

### `GET /healthz` and `GET /readyz`

Liveness and readiness. Both return `200` with `{"status":"ok"}` once the index is loaded. (The index loads before the server binds, so readiness is immediate; the first queries warm the OS page cache lazily.)

### Status codes

| Code | When |
|------|------|
| `200` | Success |
| `400` | Query dimension ≠ index dimension, `search_list < k`, or malformed JSON |
| `404` | Unknown route |
| `500` | Search failed internally |
| `503` | Request queue full (retry) |

### Examples

```bash
# index info
curl -s localhost:8080/info

# single search (k defaults to server config if omitted)
curl -s -X POST localhost:8080/search \
  -H 'content-type: application/json' \
  -d '{"query":[0.1,0.2,0.3,0.4],"k":5}'

# batch
curl -s -X POST localhost:8080/search/batch \
  -H 'content-type: application/json' \
  -d '{"queries":[[0.1,0.2,0.3,0.4],[0.5,0.6,0.7,0.8]],"k":5}'

# health
curl -s localhost:8080/healthz
```

## Client

`diskann3-client` reads query vectors from a DiskANN `.fbin` file (`[u32 num][u32 dim][f32 num*dim]`, little-endian) and calls the API, reporting latency and throughput.

```bash
# sequential, verbose per-query output
cargo run --release -p diskann3-client -- \
  --queries queries.fbin --k 10 --search-list 100

# drive the server's worker pool with concurrent requests
cargo run --release -p diskann3-client -- \
  --queries queries.fbin --concurrency 16 --quiet

# one batch request
cargo run --release -p diskann3-client -- \
  --queries queries.fbin --batch
```

### Client options

| Flag | Default | Meaning |
|------|---------|---------|
| `--server <url>` | `http://127.0.0.1:8080` | Base URL of the server. |
| `--queries <file>` | — | `.fbin` query file (required). |
| `--k <n>` | `10` | Neighbors to return. |
| `--search-list <n>` | `100` | Search-list size `L` (must be ≥ `k`). |
| `--beam <n>` | `4` | Beam width. |
| `--flat` | off | Brute-force search. |
| `--limit <n>` | all | Only send the first `n` queries. |
| `--concurrency <n>` | `1` | Parallel client threads (requests in flight at once). |
| `--batch` | off | Send all queries in one `/search/batch` request. |
| `--quiet` | off | Suppress per-query output. |

Example summary output:

```
sent 10000 queries to http://127.0.0.1:8080 (concurrency 16) ok: 10000   failed: 0
latency ms: mean 2.13, p50 1.9, p95 4.2, p99 6.8, max 11.0 
throughput: 7400.0 queries/sec (wall 1.35s)
```

## Tuning & scaling

- **`workers`** is your per-node concurrency. Set it near the storage device's effective queue depth (≈ 8–16 for a single NVMe SSD; lower for spinning or networked storage), then tune in place.
- **Find the saturation point with the client.** Sweep `--concurrency` upward and watch the reported QPS and p95: throughput climbs while there is I/O headroom, then flattens while latency rises. The plateau is the disk-I/O ceiling.
- **Raise the ceiling before adding workers.** `num_nodes_to_cache` (more hot nodes in RAM → fewer reads per query) and faster storage help more than extra threads once the disk is saturated.
- **Scale out by replication.** The index is read-only, so run identical stateless replicas behind a load balancer; throughput scales near-linearly. Update the index offline and swap atomically (blue/green); each replica holds its own copy of the files.

## Notes & limitations

- Only `Float32` indexes are wired up; other element types are rejected with a clear error.
- `distance` in the config must match the metric the index was built with.
- This service loads and serves an existing index.
- The `diskann*` dependencies track `main`; pin a `rev` in `serve/Cargo.toml` for reproducible builds.