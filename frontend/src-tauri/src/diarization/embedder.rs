//! Speaker (voice) embeddings with a WeSpeaker-style ONNX model:
//! input `feats [1, T, 80]` (log-mel fbank, mean-normalized), output `embs [1, D]`.

use super::error::DiarizationError;
use super::fbank::{mean_normalize, FbankExtractor, SAMPLE_RATE};
use ndarray::Array3;
use ort::inputs;
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::value::TensorRef;
use std::path::Path;

/// Shortest audio we embed (0.5 s); shorter segments give unreliable voice vectors.
pub const MIN_EMBED_SAMPLES: usize = SAMPLE_RATE / 2;
/// Longest audio fed to the model at once (10 s); longer segments are center-cropped.
pub const MAX_EMBED_SAMPLES: usize = SAMPLE_RATE * 10;
const NUM_BINS: usize = 80;

pub struct SpeakerEmbedder {
    session: Session,
    extractor: FbankExtractor,
}

/// L2-normalizes in place; returns the original norm.
pub fn l2_normalize(v: &mut [f32]) -> f32 {
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 1e-12 {
        v.iter_mut().for_each(|x| *x /= norm);
    }
    norm
}

impl SpeakerEmbedder {
    pub fn load(model_path: &Path, intra_threads: usize) -> Result<Self, DiarizationError> {
        let session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(intra_threads.max(1))?
            .with_inter_threads(1)?
            .commit_from_file(model_path)?;
        Ok(Self { session, extractor: FbankExtractor::new(NUM_BINS) })
    }

    /// Embeds 16 kHz mono samples in [-1, 1]. Returns an L2-normalized vector.
    pub fn embed(&mut self, samples: &[f32]) -> Result<Vec<f32>, DiarizationError> {
        if samples.len() < MIN_EMBED_SAMPLES {
            return Err(DiarizationError::TooShort { got: samples.len(), min: MIN_EMBED_SAMPLES });
        }
        let samples = if samples.len() > MAX_EMBED_SAMPLES {
            let start = (samples.len() - MAX_EMBED_SAMPLES) / 2;
            &samples[start..start + MAX_EMBED_SAMPLES]
        } else {
            samples
        };
        // WeSpeaker models expect int16-scaled samples (normalize_samples = 0)
        let scaled: Vec<f32> = samples.iter().map(|s| s * 32768.0).collect();
        let mut feats = self.extractor.compute(&scaled);
        mean_normalize(&mut feats, NUM_BINS);
        let frames = feats.len() / NUM_BINS;
        let input = Array3::from_shape_vec((1, frames, NUM_BINS), feats)
            .map_err(|e| DiarizationError::Model(e.to_string()))?;

        let outputs = self.session.run(inputs!["feats" => TensorRef::from_array_view(input.view())?])?;
        let embs = outputs
            .get("embs")
            .ok_or_else(|| DiarizationError::OutputMissing("embs".into()))?
            .try_extract_array::<f32>()?;
        let mut v: Vec<f32> = embs.iter().copied().collect();
        l2_normalize(&mut v);
        Ok(v)
    }
}

pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na < 1e-12 || nb < 1e-12 {
        0.0
    } else {
        dot / (na * nb)
    }
}
