//! Live speaker identification: voice embedding + online clustering for one meeting.

use super::clusterer::{ClustererConfig, OnlineClusterer};
use super::embedder::{SpeakerEmbedder, MIN_EMBED_SAMPLES};
use super::error::DiarizationError;
use super::fbank::SAMPLE_RATE;
use std::path::Path;

/// Result of identifying one speech segment.
#[derive(Debug, Clone, PartialEq)]
pub struct SpeakerTag {
    /// Meeting-local speaker index (0-based, stable for the whole meeting).
    pub index: usize,
    /// 0..1, low = uncertain.
    pub confidence: f32,
    /// True when this segment created the speaker.
    pub is_new: bool,
}

pub struct SpeakerIdentifier {
    embedder: SpeakerEmbedder,
    clusterer: OnlineClusterer,
}

impl SpeakerIdentifier {
    pub fn new(model_path: &Path, config: ClustererConfig, threads: usize) -> Result<Self, DiarizationError> {
        Ok(Self {
            embedder: SpeakerEmbedder::load(model_path, threads)?,
            clusterer: OnlineClusterer::new(config),
        })
    }

    pub fn speaker_count(&self) -> usize {
        self.clusterer.speaker_count()
    }

    /// Identifies the speaker of 16 kHz mono samples in [-1, 1].
    /// Returns `None` for segments too short to embed (< 0.5 s), on model errors, or when the
    /// very first segment is too short to create a speaker — the segment is then left unassigned
    /// (it can be assigned manually).
    pub fn identify(&mut self, samples_16k: &[f32]) -> Option<SpeakerTag> {
        if samples_16k.len() < MIN_EMBED_SAMPLES {
            return None;
        }
        let seconds = samples_16k.len() as f32 / SAMPLE_RATE as f32;
        let embedding = match self.embedder.embed(samples_16k) {
            Ok(e) => e,
            Err(e) => {
                log::warn!("Speaker embedding failed: {}", e);
                return None;
            }
        };
        self.clusterer
            .assign(&embedding, seconds)
            .map(|a| SpeakerTag { index: a.speaker, confidence: a.confidence, is_new: a.is_new })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Needs `DZ_MODEL` (path to the ONNX model) and ONNX Runtime available to `ort`; skipped otherwise.
    #[test]
    fn short_segments_are_left_unassigned() {
        let Ok(model) = std::env::var("DZ_MODEL") else { return };
        let mut id = SpeakerIdentifier::new(Path::new(&model), ClustererConfig::default(), 1).unwrap();
        assert!(id.identify(&vec![0.1; 4000]).is_none()); // 0.25 s
        assert_eq!(id.speaker_count(), 0);
    }
}
