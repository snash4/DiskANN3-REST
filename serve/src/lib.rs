/*
 * Library root for the serving-only crate. The `serve` binary
 * (src/bin/serve.rs) builds on these three modules.
 */

pub mod inputs;
pub mod loader;
pub mod serve;