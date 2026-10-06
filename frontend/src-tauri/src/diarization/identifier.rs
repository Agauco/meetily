//! Live speaker identification: voice embedding + online clustering for one meeting.
//!
//! Besides individual speakers, a segment can be classified as **room** ("Sala"): several people
//! talking at once, or a voice that cannot be matched to anyone reliably. Room segments never
//! create or update speakers.

use super::clusterer::{ClustererConfig, OnlineClusterer};
use super::embedder::{cosine, SpeakerEmbedder, MIN_EMBED_SAMPLES};
use super::error::DiarizationError;
use super::fbank::SAMPLE_RATE;
use std::path::Path;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpeakerKind {
    /// Meeting-local speaker index (0-based, stable for the whole meeting).
    Person(usize),
    /// Overlapping / unidentifiable speech ("Sala").
    Room,
}

/// Result of identifying one speech segment.
#[derive(Debug, Clone, PartialEq)]
pub struct SpeakerTag {
    pub kind: SpeakerKind,
    /// 0..1, low = uncertain.
    pub confidence: f32,
    /// True when this segment created a new person.
    pub is_new: bool,
}

#[derive(Debug, Clone)]
pub struct IdentifierConfig {
    pub clusterer: ClustererConfig,
    /// A segment whose sliding windows disagree with each other (mean pairwise cosine below this)
    /// contains more than one voice and is labeled "Sala".
    pub room_consistency_threshold: f32,
    /// A matched person with confidence below this (ambiguous voice that is neither clearly a known
    /// speaker nor allowed to create a new one) is labeled "Sala" instead of being forced onto the nearest speaker.
    pub room_min_confidence: f32,
    /// Consistency is only measured for segments at least this long (seconds).
    pub consistency_min_seconds: f32,
    pub window_seconds: f32,
    pub hop_seconds: f32,
}

impl Default for IdentifierConfig {
    fn default() -> Self {
        Self {
            clusterer: ClustererConfig::default(),
            room_consistency_threshold: 0.40,
            room_min_confidence: 0.15,
            consistency_min_seconds: 3.0,
            window_seconds: 1.5,
            hop_seconds: 0.75,
        }
    }
}

pub struct SpeakerIdentifier {
    embedder: SpeakerEmbedder,
    clusterer: OnlineClusterer,
    cfg: IdentifierConfig,
}

/// Mean pairwise cosine similarity of window embeddings; `None` when fewer than 2 windows.
pub(crate) fn mean_pairwise_cosine(embeddings: &[Vec<f32>]) -> Option<f32> {
    if embeddings.len() < 2 {
        return None;
    }
    let mut sum = 0.0;
    let mut n = 0usize;
    for i in 0..embeddings.len() {
        for j in i + 1..embeddings.len() {
            sum += cosine(&embeddings[i], &embeddings[j]);
            n += 1;
        }
    }
    Some(sum / n as f32)
}

impl SpeakerIdentifier {
    pub fn new(model_path: &Path, cfg: IdentifierConfig, threads: usize) -> Result<Self, DiarizationError> {
        Ok(Self {
            embedder: SpeakerEmbedder::load(model_path, threads)?,
            clusterer: OnlineClusterer::new(cfg.clusterer.clone()),
            cfg,
        })
    }

    pub fn speaker_count(&self) -> usize {
        self.clusterer.speaker_count()
    }

    /// How consistent the voice is across the segment (1 = one voice, low = several voices).
    fn voice_consistency(&mut self, samples: &[f32]) -> Option<f32> {
        let seconds = samples.len() as f32 / SAMPLE_RATE as f32;
        if seconds < self.cfg.consistency_min_seconds {
            return None;
        }
        let win = (self.cfg.window_seconds * SAMPLE_RATE as f32) as usize;
        let hop = ((self.cfg.hop_seconds * SAMPLE_RATE as f32) as usize).max(1);
        let mut embeddings = Vec::new();
        let mut start = 0usize;
        while start + win <= samples.len() {
            if let Ok(e) = self.embedder.embed(&samples[start..start + win]) {
                embeddings.push(e);
            }
            start += hop;
        }
        mean_pairwise_cosine(&embeddings)
    }

    /// Identifies the speaker of 16 kHz mono samples in [-1, 1].
    /// Returns `None` for segments too short to embed (< 0.5 s), on model errors, or when the very
    /// first segment is too short to create a speaker — such a segment is left unassigned.
    pub fn identify(&mut self, samples_16k: &[f32]) -> Option<SpeakerTag> {
        if samples_16k.len() < MIN_EMBED_SAMPLES {
            return None;
        }
        let seconds = samples_16k.len() as f32 / SAMPLE_RATE as f32;

        if let Some(consistency) = self.voice_consistency(samples_16k) {
            if consistency < self.cfg.room_consistency_threshold {
                let confidence = ((self.cfg.room_consistency_threshold - consistency) / self.cfg.room_consistency_threshold)
                    .clamp(0.0, 1.0)
                    .max(0.5);
                return Some(SpeakerTag { kind: SpeakerKind::Room, confidence, is_new: false });
            }
        }

        let embedding = match self.embedder.embed(samples_16k) {
            Ok(e) => e,
            Err(e) => {
                log::warn!("Speaker embedding failed: {}", e);
                return None;
            }
        };
        let a = self.clusterer.assign(&embedding, seconds)?;
        if !a.is_new && a.confidence < self.cfg.room_min_confidence {
            return Some(SpeakerTag { kind: SpeakerKind::Room, confidence: 0.5, is_new: false });
        }
        Some(SpeakerTag { kind: SpeakerKind::Person(a.speaker), confidence: a.confidence, is_new: a.is_new })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mean_pairwise_cosine_handles_one_voice_and_mixed_voices() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        assert_eq!(mean_pairwise_cosine(&[a.clone()]), None);
        assert!((mean_pairwise_cosine(&[a.clone(), a.clone(), a.clone()]).unwrap() - 1.0).abs() < 1e-6);
        // two identical + one orthogonal: pairs (a,a)=1, (a,b)=0, (a,b)=0 -> 1/3
        assert!((mean_pairwise_cosine(&[a.clone(), a.clone(), b]).unwrap() - 1.0 / 3.0).abs() < 1e-6);
    }

    /// Needs `DZ_MODEL` (path to the ONNX model) and ONNX Runtime available to `ort`; skipped otherwise.
    #[test]
    fn short_segments_are_left_unassigned() {
        let Ok(model) = std::env::var("DZ_MODEL") else { return };
        let mut id = SpeakerIdentifier::new(Path::new(&model), IdentifierConfig::default(), 1).unwrap();
        assert!(id.identify(&vec![0.1; 4000]).is_none()); // 0.25 s
        assert_eq!(id.speaker_count(), 0);
    }
}
