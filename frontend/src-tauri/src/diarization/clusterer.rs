//! Online speaker clustering: assigns each voice embedding to a known speaker or creates a new one.
//! Pure logic (no audio, no model), so it is fully unit-testable.
//!
//! Decision for an embedding with best cosine similarity `s` to an existing centroid:
//! * `s >= match_threshold`            -> same speaker; centroid is updated (duration-weighted mean)
//! * `new_threshold <= s < match_threshold` -> ambiguous; best speaker with low confidence, centroid NOT updated
//! * `s < new_threshold`               -> new speaker (if allowed), otherwise best speaker with low confidence

use super::embedder::{cosine, l2_normalize};

#[derive(Debug, Clone)]
pub struct ClustererConfig {
    /// Cosine similarity at/above which a segment is confidently the same speaker.
    pub match_threshold: f32,
    /// Cosine similarity below which a segment is considered a different, new speaker.
    pub new_threshold: f32,
    /// Hard cap on the number of speakers in one meeting.
    pub max_speakers: usize,
    /// Minimum speech (seconds) required to update a centroid (and to enroll the very first speaker).
    pub min_enroll_seconds: f32,
    /// Minimum speech (seconds) required to create an additional speaker. Higher than
    /// `min_enroll_seconds` because short segments of one voice often look like "new" voices.
    pub min_create_seconds: f32,
    /// Cap on the weight (seconds) a single segment adds to a centroid.
    pub max_weight_seconds: f32,
}

impl Default for ClustererConfig {
    fn default() -> Self {
        Self {
            match_threshold: 0.45,
            new_threshold: 0.40,
            max_speakers: 10,
            min_enroll_seconds: 1.5,
            min_create_seconds: 3.0,
            max_weight_seconds: 8.0,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Assignment {
    /// Index of the speaker (creation order, stable for the whole meeting).
    pub speaker: usize,
    /// 0..1; low values mean the decision is uncertain.
    pub confidence: f32,
    /// True when this segment created the speaker.
    pub is_new: bool,
}

#[derive(Debug, Clone)]
struct Cluster {
    sum: Vec<f32>,
    centroid: Vec<f32>,
    weight: f32,
    segments: usize,
}

#[derive(Debug, Clone)]
pub struct OnlineClusterer {
    cfg: ClustererConfig,
    clusters: Vec<Cluster>,
}

impl OnlineClusterer {
    pub fn new(cfg: ClustererConfig) -> Self {
        Self { cfg, clusters: Vec::new() }
    }

    pub fn config(&self) -> &ClustererConfig {
        &self.cfg
    }

    pub fn speaker_count(&self) -> usize {
        self.clusters.len()
    }

    /// Unit-length centroid of a speaker.
    pub fn centroid(&self, speaker: usize) -> Option<&[f32]> {
        self.clusters.get(speaker).map(|c| c.centroid.as_slice())
    }

    /// Restores a known speaker (e.g. when resuming a meeting); returns its index.
    pub fn add_known_speaker(&mut self, centroid: &[f32], weight_seconds: f32) -> usize {
        let mut c = centroid.to_vec();
        l2_normalize(&mut c);
        let weight = weight_seconds.max(self.cfg.min_enroll_seconds);
        self.clusters.push(Cluster {
            sum: c.iter().map(|x| x * weight).collect(),
            centroid: c,
            weight,
            segments: 1,
        });
        self.clusters.len() - 1
    }

    /// Merges speaker `from` into `into` (e.g. after a manual merge). Indices above `from` shift down by one.
    pub fn merge(&mut self, from: usize, into: usize) {
        if from == into || from >= self.clusters.len() || into >= self.clusters.len() {
            return;
        }
        let removed = self.clusters.remove(from);
        let into = if into > from { into - 1 } else { into };
        let target = &mut self.clusters[into];
        for (t, r) in target.sum.iter_mut().zip(&removed.sum) {
            *t += r;
        }
        target.weight += removed.weight;
        target.segments += removed.segments;
        target.centroid = target.sum.clone();
        l2_normalize(&mut target.centroid);
    }

    fn best(&self, emb: &[f32]) -> Option<(usize, f32)> {
        self.clusters
            .iter()
            .enumerate()
            .map(|(i, c)| (i, cosine(emb, &c.centroid)))
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
    }

    /// Maps a similarity to a 0..1 confidence relative to the thresholds.
    fn confidence(&self, s: f32) -> f32 {
        let lo = self.cfg.new_threshold;
        let hi = self.cfg.match_threshold;
        if s >= hi {
            // matched: 0.7 at the threshold up to 1.0 at perfect similarity
            0.7 + 0.3 * ((s - hi) / (1.0 - hi).max(1e-6)).clamp(0.0, 1.0)
        } else {
            // ambiguous / rejected: 0..0.7 across the gray zone
            (0.7 * ((s - lo) / (hi - lo).max(1e-6)).clamp(0.0, 1.0)).min(0.69)
        }
    }

    /// Assigns an embedding (any length; normalized internally) extracted from `seconds` of speech.
    /// Returns `None` when there is no speaker yet and the segment is too short to create one.
    pub fn assign(&mut self, embedding: &[f32], seconds: f32) -> Option<Assignment> {
        let mut emb = embedding.to_vec();
        if l2_normalize(&mut emb) <= 1e-12 {
            return None;
        }
        let long_enough = seconds >= self.cfg.min_enroll_seconds;

        let Some((best, s)) = self.best(&emb) else {
            if !long_enough {
                return None;
            }
            self.push_cluster(&emb, seconds);
            return Some(Assignment { speaker: 0, confidence: 1.0, is_new: true });
        };

        if s >= self.cfg.match_threshold {
            if long_enough {
                self.update(best, &emb, seconds);
            }
            // short segments match but are never fully trusted
            let conf = self.confidence(s) * if long_enough { 1.0 } else { 0.85 };
            return Some(Assignment { speaker: best, confidence: conf, is_new: false });
        }

        if s < self.cfg.new_threshold
            && seconds >= self.cfg.min_create_seconds
            && self.clusters.len() < self.cfg.max_speakers
        {
            self.push_cluster(&emb, seconds);
            return Some(Assignment { speaker: self.clusters.len() - 1, confidence: 0.8, is_new: true });
        }

        Some(Assignment { speaker: best, confidence: self.confidence(s), is_new: false })
    }

    fn push_cluster(&mut self, emb: &[f32], seconds: f32) {
        let weight = seconds.min(self.cfg.max_weight_seconds);
        self.clusters.push(Cluster {
            sum: emb.iter().map(|x| x * weight).collect(),
            centroid: emb.to_vec(),
            weight,
            segments: 1,
        });
    }

    fn update(&mut self, idx: usize, emb: &[f32], seconds: f32) {
        let weight = seconds.min(self.cfg.max_weight_seconds);
        let c = &mut self.clusters[idx];
        for (s, e) in c.sum.iter_mut().zip(emb) {
            *s += e * weight;
        }
        c.weight += weight;
        c.segments += 1;
        c.centroid = c.sum.clone();
        l2_normalize(&mut c.centroid);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Unit vector in `dim` dimensions pointing mostly along axis `axis`, perturbed by `noise` along `other`.
    fn vec_with(dim: usize, axis: usize, other: usize, noise: f32) -> Vec<f32> {
        let mut v = vec![0.0; dim];
        v[axis] = 1.0;
        v[other] = noise;
        v
    }

    #[test]
    fn first_long_segment_creates_speaker_zero() {
        let mut c = OnlineClusterer::new(ClustererConfig::default());
        let a = c.assign(&vec_with(8, 0, 7, 0.0), 5.0).unwrap();
        assert_eq!(a, Assignment { speaker: 0, confidence: 1.0, is_new: true });
        assert_eq!(c.speaker_count(), 1);
    }

    #[test]
    fn short_segment_cannot_create_the_first_speaker() {
        let mut c = OnlineClusterer::new(ClustererConfig::default());
        assert!(c.assign(&vec_with(8, 0, 7, 0.0), 0.8).is_none());
        assert_eq!(c.speaker_count(), 0);
    }

    #[test]
    fn same_voice_matches_and_different_voice_creates_new_speaker() {
        let mut c = OnlineClusterer::new(ClustererConfig::default());
        c.assign(&vec_with(8, 0, 7, 0.0), 5.0).unwrap();
        let same = c.assign(&vec_with(8, 0, 7, 0.2), 5.0).unwrap();
        assert_eq!((same.speaker, same.is_new), (0, false));
        assert!(same.confidence >= 0.7);
        let other = c.assign(&vec_with(8, 1, 7, 0.0), 5.0).unwrap();
        assert_eq!((other.speaker, other.is_new), (1, true));
        // back to the first voice
        let back = c.assign(&vec_with(8, 0, 7, 0.1), 2.0).unwrap();
        assert_eq!((back.speaker, back.is_new), (0, false));
    }

    #[test]
    fn short_different_segment_does_not_create_speaker_and_has_low_confidence() {
        let mut c = OnlineClusterer::new(ClustererConfig::default());
        c.assign(&vec_with(8, 0, 7, 0.0), 5.0).unwrap();
        let a = c.assign(&vec_with(8, 1, 7, 0.0), 0.7).unwrap();
        assert_eq!((a.speaker, a.is_new), (0, false));
        assert!(a.confidence < 0.1, "confidence {}", a.confidence);
        assert_eq!(c.speaker_count(), 1);
    }

    #[test]
    fn ambiguous_segments_do_not_move_the_centroid() {
        let cfg = ClustererConfig::default();
        let mut c = OnlineClusterer::new(cfg);
        c.assign(&vec_with(8, 0, 7, 0.0), 5.0).unwrap();
        let before = c.centroid(0).unwrap().to_vec();
        // cosine to axis 0 is 0.42: inside the gray zone (0.40..0.45)
        let mut amb = vec![0.0; 8];
        amb[0] = 0.42;
        amb[1] = (1.0f32 - 0.42 * 0.42).sqrt();
        let a = c.assign(&amb, 5.0).unwrap();
        assert!(!a.is_new && a.confidence < 0.7);
        assert_eq!(c.centroid(0).unwrap(), before.as_slice());
    }

    #[test]
    fn respects_max_speakers() {
        let mut c = OnlineClusterer::new(ClustererConfig { max_speakers: 2, ..Default::default() });
        for axis in 0..4 {
            c.assign(&vec_with(8, axis, 7, 0.0), 5.0);
        }
        assert_eq!(c.speaker_count(), 2);
    }

    #[test]
    fn merge_combines_speakers_and_shifts_indices() {
        let mut c = OnlineClusterer::new(ClustererConfig::default());
        for axis in 0..3 {
            c.assign(&vec_with(8, axis, 7, 0.0), 5.0);
        }
        c.merge(0, 2); // speaker 0 into 2 -> speakers [1, merged(2+0)]
        assert_eq!(c.speaker_count(), 2);
        // a segment of the old speaker 0's voice now matches the merged speaker (index 1)
        let a = c.assign(&vec_with(8, 0, 7, 0.0), 5.0).unwrap();
        assert!(!a.is_new);
        assert_eq!(a.speaker, 1);
    }

    #[test]
    fn restores_known_speakers() {
        let mut c = OnlineClusterer::new(ClustererConfig::default());
        let idx = c.add_known_speaker(&vec_with(8, 3, 7, 0.0), 10.0);
        assert_eq!(idx, 0);
        let a = c.assign(&vec_with(8, 3, 7, 0.05), 2.0).unwrap();
        assert_eq!((a.speaker, a.is_new), (0, false));
    }

    #[test]
    fn creating_an_extra_speaker_needs_more_speech_than_matching() {
        let mut c = OnlineClusterer::new(ClustererConfig::default());
        c.assign(&vec_with(8, 0, 7, 0.0), 5.0).unwrap();
        // different voice, 2 s: long enough to enroll/update but too short to create a speaker
        let short = c.assign(&vec_with(8, 1, 7, 0.0), 2.0).unwrap();
        assert!(!short.is_new && short.speaker == 0);
        assert_eq!(c.speaker_count(), 1);
        let long = c.assign(&vec_with(8, 1, 7, 0.0), 4.0).unwrap();
        assert!(long.is_new && long.speaker == 1);
    }

    #[test]
    fn zero_vector_is_rejected() {
        let mut c = OnlineClusterer::new(ClustererConfig::default());
        assert!(c.assign(&[0.0; 8], 5.0).is_none());
    }
}
