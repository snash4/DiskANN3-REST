/*
 * `serve`: a synchronous tiny_http REST server for a loaded disk index.
 *
 * One acceptor thread (this thread) pulls connections and pushes them into a
 * bounded queue; N scoped worker threads each take a request and run the
 * BLOCKING search on a plain OS thread. Because workers are not inside any
 * async runtime, DiskIndexSearcher's internal `block_on` is safe. Workers
 * borrow `&searcher` (shared, needs only `Sync`). When the queue is full the
 * acceptor responds 503 immediately rather than letting latency grow unbounded.
 *
 * Routes:
 *   POST /search        { query: [f32], k?, search_list?, beam_width?, flat? }
 *   POST /search/batch  { queries: [[f32]], k?, ... }
 *   GET  /healthz /readyz   -> {"status":"ok"} (serve immediately, warm lazily)
 *   GET  /info          -> { num_points, dim, metric, defaults }
 */

use std::io::{Cursor, Read};
use std::sync::atomic::{AtomicU64, Ordering};

use anyhow::Result;
use crossbeam_channel::{bounded, TrySendError};
use serde::{Deserialize, Serialize};
use tiny_http::{Header, Method, Request, Response, Server};

use diskann_vector::distance::Metric;

use crate::inputs::{DataType, SimilarityMeasure};
use crate::loader::{LoadedIndex, Searcher};

const MAX_BODY_BYTES: u64 = 16 * 1024 * 1024; // reject absurd request bodies

type Resp = Response<Cursor<Vec<u8>>>;

// ----------------------------- configuration -----------------------------

#[derive(Debug, Deserialize)]
pub struct ServeConfig {
    pub load_path: String,
    pub data_type: DataType,
    pub distance: SimilarityMeasure,
    #[serde(default = "default_addr")]
    pub addr: String,
    #[serde(default = "default_workers")]
    pub workers: usize,
    #[serde(default = "default_queue")]
    pub queue_capacity: usize,
    #[serde(default)]
    pub num_nodes_to_cache: Option<usize>,
    #[serde(default)]
    pub search_io_limit: Option<usize>,
    #[serde(default)]
    pub defaults: SearchDefaults,
}
fn default_addr() -> String {
    "0.0.0.0:8080".to_string()
}
fn default_workers() -> usize {
    8
}
fn default_queue() -> usize {
    256
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct SearchDefaults {
    #[serde(default = "d_k")]
    pub k: u32,
    #[serde(default = "d_l")]
    pub search_list: u32,
    #[serde(default = "d_beam")]
    pub beam_width: usize,
    #[serde(default)]
    pub flat: bool,
}
fn d_k() -> u32 {
    10
}
fn d_l() -> u32 {
    100
}
fn d_beam() -> usize {
    4
}
impl Default for SearchDefaults {
    fn default() -> Self {
        Self { k: d_k(), search_list: d_l(), beam_width: d_beam(), flat: false }
    }
}

// --------------------------- request / response ---------------------------

#[derive(Deserialize)]
struct SearchRequest {
    query: Vec<f32>,
    k: Option<u32>,
    search_list: Option<u32>,
    beam_width: Option<usize>,
    flat: Option<bool>,
}
#[derive(Deserialize)]
struct BatchRequest {
    queries: Vec<Vec<f32>>,
    k: Option<u32>,
    search_list: Option<u32>,
    beam_width: Option<usize>,
    flat: Option<bool>,
}
#[derive(Serialize)]
struct Neighbor {
    id: u32,
    distance: f32,
}
#[derive(Serialize)]
struct SearchResponse {
    neighbors: Vec<Neighbor>,
    comparisons: u32,
    search_ms: f64,
}
#[derive(Serialize)]
struct BatchResponse {
    results: Vec<SearchResponse>,
}
#[derive(Serialize)]
struct Health {
    status: &'static str,
}
#[derive(Serialize)]
struct InfoResponse {
    num_points: usize,
    dim: usize,
    metric: &'static str,
    defaults: SearchDefaults,
}
#[derive(Serialize)]
struct ErrorBody {
    error: String,
}

struct Params {
    k: u32,
    l: u32,
    beam: usize,
    flat: bool,
}
impl SearchDefaults {
    fn resolve(
        &self,
        k: Option<u32>,
        l: Option<u32>,
        beam: Option<usize>,
        flat: Option<bool>,
    ) -> Params {
        Params {
            k: k.unwrap_or(self.k),
            l: l.unwrap_or(self.search_list),
            beam: beam.unwrap_or(self.beam_width),
            flat: flat.unwrap_or(self.flat),
        }
    }
}

// ------------------------------ search core -------------------------------

fn search_one(searcher: &Searcher<f32>, query: &[f32], p: &Params) -> Result<SearchResponse> {
    let t = std::time::Instant::now();
    let res = searcher
        .search(query, p.k, p.l, Some(p.beam), None, p.flat)
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    let search_ms = t.elapsed().as_secs_f64() * 1000.0;
    println!(
        "search: k={}, search_list={}, beam={}, flat={} -> {} neighbors, {} cmps, {:.2} ms",
        p.k,
        p.l,
        p.beam,
        p.flat,
        res.results.len(),
        res.stats.cmps,
        search_ms
    );
    Ok(SearchResponse {
        neighbors: res
            .results
            .iter()
            .map(|n| Neighbor { id: n.vertex_id, distance: n.distance })
            .collect(),
        comparisons: res.stats.cmps,
        search_ms,
    })
}

fn metric_name(m: Metric) -> &'static str {
    match m {
        Metric::L2 => "L2",
        Metric::Cosine => "Cosine",
        Metric::CosineNormalized => "CosineNormalized",
        Metric::InnerProduct => "InnerProduct",
    }
}

// ----------------------------- HTTP plumbing ------------------------------

fn json_response<T: Serialize>(body: &T, status: u16) -> Resp {
    let data = serde_json::to_vec(body).unwrap_or_else(|_| b"{\"error\":\"encode\"}".to_vec());
    let header =
        Header::from_bytes(&b"Content-Type"[..], &b"application/json"[..]).expect("valid header");
    Response::from_data(data).with_status_code(status).with_header(header)
}
fn err(status: u16, msg: impl Into<String>) -> Resp {
    json_response(&ErrorBody { error: msg.into() }, status)
}

fn read_body(req: &mut Request) -> std::io::Result<Vec<u8>> {
    let mut buf = Vec::new();
    req.as_reader().take(MAX_BODY_BYTES).read_to_end(&mut buf)?;
    Ok(buf)
}

struct Ctx<'a> {
    searcher: &'a Searcher<f32>,
    dims: usize,
    num_points: usize,
    metric: Metric,
    defaults: SearchDefaults,
}

impl Ctx<'_> {
    fn info(&self) -> InfoResponse {
        InfoResponse {
            num_points: self.num_points,
            dim: self.dims,
            metric: metric_name(self.metric),
            defaults: self.defaults,
        }
    }

    fn search(&self, r: SearchRequest) -> Resp {
        if r.query.len() != self.dims {
            return err(400, format!("query dim {} != index dim {}", r.query.len(), self.dims));
        }
        let p = self.defaults.resolve(r.k, r.search_list, r.beam_width, r.flat);
        if p.l < p.k {
            return err(400, format!("search_list {} must be >= k {}", p.l, p.k));
        }
        match search_one(self.searcher, &r.query, &p) {
            Ok(out) => json_response(&out, 200),
            Err(e) => err(500, format!("search failed: {e}")),
        }
    }

    fn batch(&self, r: BatchRequest) -> Resp {
        let p = self.defaults.resolve(r.k, r.search_list, r.beam_width, r.flat);
        if p.l < p.k {
            return err(400, format!("search_list {} must be >= k {}", p.l, p.k));
        }
        let mut results = Vec::with_capacity(r.queries.len());
        for (i, q) in r.queries.iter().enumerate() {
            if q.len() != self.dims {
                return err(400, format!("query {i} dim {} != index dim {}", q.len(), self.dims));
            }
            match search_one(self.searcher, q, &p) {
                Ok(out) => results.push(out),
                Err(e) => return err(500, format!("query {i} failed: {e}")),
            }
        }
        json_response(&BatchResponse { results }, 200)
    }
}

fn handle(mut req: Request, ctx: &Ctx) {
    let method = req.method().clone();
    let url = req.url().to_string();
    let resp = match (&method, url.as_str()) {
        (Method::Get, "/healthz") | (Method::Get, "/readyz") => {
            json_response(&Health { status: "ok" }, 200)
        }
        (Method::Get, "/info") => json_response(&ctx.info(), 200),
        (Method::Post, "/search") => match read_body(&mut req) {
            Ok(body) => match serde_json::from_slice::<SearchRequest>(&body) {
                Ok(r) => ctx.search(r),
                Err(e) => err(400, format!("invalid JSON: {e}")),
            },
            Err(e) => err(400, format!("read error: {e}")),
        },
        (Method::Post, "/search/batch") => match read_body(&mut req) {
            Ok(body) => match serde_json::from_slice::<BatchRequest>(&body) {
                Ok(r) => ctx.batch(r),
                Err(e) => err(400, format!("invalid JSON: {e}")),
            },
            Err(e) => err(400, format!("read error: {e}")),
        },
        _ => err(404, "not found"),
    };
    let _ = req.respond(resp);
}

/// Serve the loaded index over HTTP until the process is killed.
pub fn run(loaded: LoadedIndex<f32>, config: &ServeConfig) -> Result<()> {
    let server = Server::http(&config.addr)
        .map_err(|e| anyhow::anyhow!("failed to bind {}: {e}", config.addr))?;

    let ctx = Ctx {
        searcher: &loaded.searcher,
        dims: loaded.dims,
        num_points: loaded.num_points,
        metric: loaded.metric,
        defaults: config.defaults,
    };

    let (tx, rx) = bounded::<Request>(config.queue_capacity);
    let rejected = AtomicU64::new(0);

    println!(
        "serving on http://{} ({} workers, queue capacity {})",
        config.addr, config.workers, config.queue_capacity
    );

    std::thread::scope(|scope| {
        let ctx_ref = &ctx;
        for _ in 0..config.workers {
            let rx = rx.clone();
            scope.spawn(move || {
                while let Ok(req) = rx.recv() {
                    handle(req, ctx_ref);
                }
            });
        }
        drop(rx); // workers hold their own clones; drop this extra handle

        // Acceptor loop (runs on this thread, forever).
        for req in server.incoming_requests() {
            match tx.try_send(req) {
                Ok(()) => {}
                Err(TrySendError::Full(req)) => {
                    rejected.fetch_add(1, Ordering::Relaxed);
                    let _ = req.respond(err(503, "server busy, retry later"));
                }
                Err(TrySendError::Disconnected(req)) => {
                    let _ = req.respond(err(503, "server shutting down"));
                }
            }
        }
    });

    Ok(())
}