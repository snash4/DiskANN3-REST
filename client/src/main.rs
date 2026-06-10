//! HTTP client for the diskann3-serve REST API.
//!
//! Reads query vectors from a DiskANN `.fbin` file and calls the server's
//! `/search` (or `/search/batch`) endpoint, reporting latency and throughput.
//!
//! Usage:
//!     diskann3-client --queries queries.fbin [options]
//!
//! Options:
//!     --server <url>        base URL (default http://127.0.0.1:8080)
//!     --queries <file>      .fbin query file (required)
//!     --k <n>               neighbors to return (default 10)
//!     --search-list <n>     search-list size L (default 100)
//!     --beam <n>            beam width (default 4)
//!     --flat                brute-force search
//!     --limit <n>           only send the first n queries (default all)
//!     --concurrency <n>     parallel client threads (default 1)
//!     --batch               send all queries in one /search/batch request
//!     --quiet               suppress per-query output

use std::fs::File;
use std::io::Read;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};

// ----------------------------- wire types --------------------------------

#[derive(Serialize)]
struct SearchRequest<'a> {
    query: &'a [f32],
    k: u32,
    search_list: u32,
    beam_width: usize,
    flat: bool,
}
#[derive(Serialize)]
struct BatchRequest<'a> {
    queries: &'a [Vec<f32>],
    k: u32,
    search_list: u32,
    beam_width: usize,
    flat: bool,
}
#[derive(Deserialize)]
struct Neighbor {
    id: u32,
    #[allow(dead_code)]
    distance: f32,
}
#[derive(Deserialize)]
struct SearchResponse {
    neighbors: Vec<Neighbor>,
    comparisons: u32,
    search_ms: f64,
}
#[derive(Deserialize)]
struct BatchResponse {
    results: Vec<SearchResponse>,
}

// ------------------------------- config -----------------------------------

struct Args {
    server: String,
    queries: String,
    k: u32,
    search_list: u32,
    beam_width: usize,
    flat: bool,
    limit: Option<usize>,
    concurrency: usize,
    batch: bool,
    quiet: bool,
}

fn parse_args() -> Result<Args> {
    let mut a = Args {
        server: "http://127.0.0.1:8080".to_string(),
        queries: String::new(),
        k: 10,
        search_list: 100,
        beam_width: 4,
        flat: false,
        limit: None,
        concurrency: 1,
        batch: false,
        quiet: false,
    };
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < argv.len() {
        let arg = &argv[i];
        let mut next = || {
            i += 1;
            argv.get(i).cloned().ok_or_else(|| anyhow::anyhow!("missing value after {arg}"))
        };
        match arg.as_str() {
            "--server" => a.server = next()?,
            "--queries" => a.queries = next()?,
            "--k" => a.k = next()?.parse().context("--k")?,
            "--search-list" => a.search_list = next()?.parse().context("--search-list")?,
            "--beam" => a.beam_width = next()?.parse().context("--beam")?,
            "--limit" => a.limit = Some(next()?.parse().context("--limit")?),
            "--concurrency" => a.concurrency = next()?.parse().context("--concurrency")?,
            "--flat" => a.flat = true,
            "--batch" => a.batch = true,
            "--quiet" => a.quiet = true,
            other => bail!("unknown argument: {other}"),
        }
        i += 1;
    }
    if a.queries.is_empty() {
        bail!("--queries <file.fbin> is required");
    }
    if a.search_list < a.k {
        bail!("--search-list ({}) must be >= --k ({})", a.search_list, a.k);
    }
    a.concurrency = a.concurrency.max(1);
    Ok(a)
}

// --------------------------- fbin query reader -----------------------------

/// Read a DiskANN `.fbin`: little-endian `[u32 num][u32 dim][f32 num*dim]`.
fn read_fbin(path: &str) -> Result<(Vec<Vec<f32>>, usize)> {
    let mut buf = Vec::new();
    File::open(path)
        .with_context(|| format!("opening {path}"))?
        .read_to_end(&mut buf)
        .with_context(|| format!("reading {path}"))?;
    if buf.len() < 8 {
        bail!("{path} is too small to be an fbin file");
    }
    let num = u32::from_le_bytes(buf[0..4].try_into().unwrap()) as usize;
    let dim = u32::from_le_bytes(buf[4..8].try_into().unwrap()) as usize;
    let expected = 8 + num * dim * 4;
    if buf.len() < expected {
        bail!("{path}: header says {num}x{dim} ({expected} bytes) but file is {}", buf.len());
    }
    let mut vectors = Vec::with_capacity(num);
    let mut off = 8;
    for _ in 0..num {
        let mut v = Vec::with_capacity(dim);
        for _ in 0..dim {
            v.push(f32::from_le_bytes(buf[off..off + 4].try_into().unwrap()));
            off += 4;
        }
        vectors.push(v);
    }
    Ok((vectors, dim))
}

// ------------------------------- HTTP I/O ---------------------------------

fn send_one(
    agent: &ureq::Agent,
    url: &str,
    args: &Args,
    query: &[f32],
) -> Result<(SearchResponse, f64)> {
    let body = SearchRequest {
        query,
        k: args.k,
        search_list: args.search_list,
        beam_width: args.beam_width,
        flat: args.flat,
    };
    let start = Instant::now();
    let mut resp = agent.post(url).send_json(&body)?;
    let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;

    let status = resp.status();
    if !status.is_success() {
        let msg = resp.body_mut().read_to_string().unwrap_or_default();
        bail!("HTTP {}: {}", status.as_u16(), msg.trim());
    }
    let parsed: SearchResponse = resp.body_mut().read_json().context("decoding response")?;
    Ok((parsed, elapsed_ms))
}

// ------------------------------- summary ----------------------------------

fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = ((p / 100.0) * (sorted.len() as f64 - 1.0)).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

fn print_summary(args: &Args, latencies: &mut Vec<f64>, ok: usize, failed: usize, wall_s: f64) {
    latencies.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mean = if latencies.is_empty() {
        0.0
    } else {
        latencies.iter().sum::<f64>() / latencies.len() as f64
    };
    let qps = if wall_s > 0.0 { ok as f64 / wall_s } else { 0.0 };
    println!("\nsent {} queries to {} (concurrency {})", ok + failed, args.server, args.concurrency);
    println!("  ok: {ok}   failed: {failed}");
    println!(
        "  latency ms: mean {:.2}, p50 {:.2}, p95 {:.2}, p99 {:.2}, max {:.2}",
        mean,
        percentile(latencies, 50.0),
        percentile(latencies, 95.0),
        percentile(latencies, 99.0),
        latencies.last().copied().unwrap_or(0.0),
    );
    println!("  throughput: {qps:.1} queries/sec (wall {wall_s:.2}s)");
}

// --------------------------------- main -----------------------------------

fn main() -> Result<()> {
    let args = parse_args()?;

    let (mut queries, dim) = read_fbin(&args.queries)?;
    if let Some(limit) = args.limit {
        queries.truncate(limit);
    }
    if queries.is_empty() {
        bail!("no queries to send");
    }
    println!("loaded {} queries (dim {}) from {}", queries.len(), dim, args.queries);

    // One shared agent (connection pool); 4xx/5xx come back as Ok so we can
    // read the body, instead of being turned into errors.
    let agent: ureq::Agent = ureq::Agent::config_builder()
        .timeout_global(Some(Duration::from_secs(30)))
        .http_status_as_error(false)
        .build()
        .into();

    if args.batch {
        let url = format!("{}/search/batch", args.server.trim_end_matches('/'));
        let body = BatchRequest {
            queries: &queries,
            k: args.k,
            search_list: args.search_list,
            beam_width: args.beam_width,
            flat: args.flat,
        };
        let start = Instant::now();
        let mut resp = agent.post(&url).send_json(&body)?;
        let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
        if !resp.status().is_success() {
            let msg = resp.body_mut().read_to_string().unwrap_or_default();
            bail!("HTTP {}: {}", resp.status().as_u16(), msg.trim());
        }
        let parsed: BatchResponse = resp.body_mut().read_json().context("decoding batch response")?;
        let total_cmps: u64 = parsed.results.iter().map(|r| r.comparisons as u64).sum();
        println!(
            "\nbatch of {} queries returned in {:.2} ms ({:.1} q/s); total comparisons {}",
            parsed.results.len(),
            elapsed_ms,
            parsed.results.len() as f64 / (elapsed_ms / 1000.0),
            total_cmps,
        );
        return Ok(());
    }

    let url = format!("{}/search", args.server.trim_end_matches('/'));
    let n = queries.len();
    let next = AtomicUsize::new(0);
    let verbose = args.concurrency == 1 && !args.quiet;

    let mut latencies = Vec::with_capacity(n);
    let mut total_ok = 0usize;
    let mut total_failed = 0usize;
    let mut first_error: Option<String> = None;

    let wall = Instant::now();
    std::thread::scope(|scope| {
        let handles: Vec<_> = (0..args.concurrency)
            .map(|_| {
                let (agent, queries, url, args, next) =
                    (&agent, &queries, &url, &args, &next);
                scope.spawn(move || {
                    let mut lats = Vec::new();
                    let (mut ok, mut failed) = (0usize, 0usize);
                    let mut err: Option<String> = None;
                    loop {
                        let i = next.fetch_add(1, Ordering::Relaxed);
                        if i >= n {
                            break;
                        }
                        match send_one(agent, url, args, &queries[i]) {
                            Ok((resp, ms)) => {
                                lats.push(ms);
                                ok += 1;
                                if verbose {
                                    let ids: Vec<u32> =
                                        resp.neighbors.iter().take(5).map(|nb| nb.id).collect();
                                    println!(
                                        "query {i}: top {:?}{} ({} cmps, search {:.2} ms, rest {:.2} ms, total {:.2} ms)",
                                        ids,
                                        if resp.neighbors.len() > 5 { " ..." } else { "" },
                                        resp.comparisons,
                                        resp.search_ms,
                                        ms - resp.search_ms,
                                        ms,
                                    );
                                }
                            }
                            Err(e) => {
                                failed += 1;
                                if err.is_none() {
                                    err = Some(e.to_string());
                                }
                            }
                        }
                    }
                    (lats, ok, failed, err)
                })
            })
            .collect();

        for h in handles {
            let (lats, ok, failed, err) = h.join().expect("worker panicked");
            latencies.extend(lats);
            total_ok += ok;
            total_failed += failed;
            if first_error.is_none() {
                first_error = err;
            }
        }
    });
    let wall_s = wall.elapsed().as_secs_f64();

    if let Some(e) = first_error {
        eprintln!("first error: {e}");
    }
    print_summary(&args, &mut latencies, total_ok, total_failed, wall_s);
    Ok(())
}