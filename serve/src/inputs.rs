/*
 * `inputs`: the two config enums the server needs.
 *   - SimilarityMeasure mirrors diskann-benchmark's enum and converts to Metric.
 *   - DataType selects the vector element type (only Float32 is wired up).
 */

use diskann_vector::distance::Metric;
use serde::{Deserialize, Serialize};

/// Distance measure, mirroring `diskann_benchmark::utils::SimilarityMeasure`.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum SimilarityMeasure {
    SquaredL2,
    InnerProduct,
    Cosine,
    CosineNormalized,
}

impl From<SimilarityMeasure> for Metric {
    fn from(value: SimilarityMeasure) -> Self {
        match value {
            SimilarityMeasure::SquaredL2 => Metric::L2,
            SimilarityMeasure::InnerProduct => Metric::InnerProduct,
            SimilarityMeasure::Cosine => Metric::Cosine,
            SimilarityMeasure::CosineNormalized => Metric::CosineNormalized,
        }
    }
}

/// Vector element type of the index.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DataType {
    Float32,
    Float16,
    UInt8,
    Int8,
}