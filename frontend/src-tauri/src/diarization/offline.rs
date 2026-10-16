//! Offline (after-recording) speaker diarization: clusters the voice embeddings of all transcript
//! segments of a meeting at once and maps the clusters onto the meeting's existing speakers.
//!
//! Unlike the live [`OnlineClusterer`](super::clusterer::OnlineClusterer), the offline pass sees the
//! whole meeting, so early decisions can be corrected: it runs average-linkage agglomerative
//! clustering over the longer segments ("anchors") and then attaches short segments to the nearest
//! cluster. Pure logic (no audio, no model, no database), fully unit-testable.
//!
//! Manual corrections are never decided here — the caller writes only the *automatic* speaker.
//! They are, however, used as evidence when mapping clusters onto existing speakers, so names the
//! user gave stay with the right voice.

use super::embedder::{cosine, l2_normalize};
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct OfflineConfig {
    /// Average-linkage similarity below which two clusters are kept apart.
    pub merge_threshold: f32,
    /// Segments at least this long (seconds) form clusters; shorter ones are attached afterwards.
    pub anchor_min_seconds: f32,
    /// Clusters with less speech than this (seconds) are folded into their nearest neighbour.
    pub min_speaker_seconds: f32,
    /// Hard cap on the number of speakers.
    pub max_speakers: usize,
    /// Known number of participants who speak; when set, clustering stops exactly there (if possible).
    pub expected_speakers: Option<usize>,
    /// A short segment less similar than this to every speaker is labeled room ("Sala").
    pub room_min_similarity: f32,
    /// Cap on the weight (seconds) a single segment adds to a centroid.
    pub max_weight_seconds: f32,
    /// Minimum centroid similarity for a cluster with no shared segments to inherit an existing speaker.
    pub inherit_min_similarity: f32,
}

impl Default for OfflineConfig {
    fn default() -> Self {
        Self {
            merge_threshold: 0.35,
            anchor_min_seconds: 1.5,
            min_speaker_seconds: 3.0,
            max_speakers: 10,
            expected_speakers: None,
            room_min_similarity: 0.15,
            max_weight_seconds: 8.0,
            inherit_min_similarity: 0.45,
        }
    }
}

/// Voice analysis of one transcript segment.
#[derive(Debug, Clone, Default)]
pub struct VoiceSegment {
    /// L2-normalized voice embedding; `None` when the segment could not be embedded (too short, no audio).
    pub embedding: Option<Vec<f32>>,
    /// Speech duration in seconds.
    pub seconds: f32,
    /// Several voices detected inside the segment ("Sala").
    pub is_room: bool,
    /// Confidence of the room decision.
    pub room_confidence: f32,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum OfflineLabel {
    /// Cluster index (0-based, ordered by first appearance in the meeting).
    Person { cluster: usize, confidence: f32 },
    Room { confidence: f32 },
    /// Not enough information; the caller keeps whatever was there before.
    Unknown,
}

#[derive(Debug, Clone)]
pub struct OfflineResult {
    pub labels: Vec<OfflineLabel>,
    /// Unit-length centroid of every cluster.
    pub centroids: Vec<Vec<f32>>,
}

impl OfflineResult {
    pub fn cluster_count(&self) -> usize {
        self.centroids.len()
    }

    pub fn cluster_of(&self, segment: usize) -> Option<usize> {
        match self.labels.get(segment) {
            Some(OfflineLabel::Person { cluster, .. }) => Some(*cluster),
            _ => None,
        }
    }
}

fn weight(cfg: &OfflineConfig, seconds: f32) -> f32 {
    seconds.clamp(0.1, cfg.max_weight_seconds)
}

/// Weighted, normalized mean of the given segments' embeddings.
fn centroid_of(segments: &[VoiceSegment], members: &[usize], cfg: &OfflineConfig) -> Vec<f32> {
    let dim = members
        .iter()
        .find_map(|&i| segments[i].embedding.as_ref().map(|e| e.len()))
        .unwrap_or(0);
    let mut sum = vec![0.0f32; dim];
    for &i in members {
        if let Some(e) = &segments[i].embedding {
            let w = weight(cfg, segments[i].seconds);
            for (s, x) in sum.iter_mut().zip(e) {
                *s += x * w;
            }
        }
    }
    l2_normalize(&mut sum);
    sum
}

/// Average-linkage agglomerative clustering of `anchors` (indices into `segments`).
/// Returns the member lists of the final clusters.
fn agglomerate(segments: &[VoiceSegment], anchors: &[usize], cfg: &OfflineConfig) -> Vec<Vec<usize>> {
    let n = anchors.len();
    if n == 0 {
        return Vec::new();
    }
    let emb = |k: usize| segments[anchors[k]].embedding.as_deref().unwrap_or(&[]);

    // linkage[i][j]: average pairwise similarity between clusters i and j
    let mut linkage = vec![vec![0.0f32; n]; n];
    for i in 0..n {
        for j in i + 1..n {
            let s = cosine(emb(i), emb(j));
            linkage[i][j] = s;
            linkage[j][i] = s;
        }
    }
    let mut members: Vec<Option<Vec<usize>>> = (0..n).map(|k| Some(vec![anchors[k]])).collect();
    let mut seconds: Vec<f32> = anchors.iter().map(|&a| segments[a].seconds).collect();
    let mut count = n;
    let max_speakers = cfg.max_speakers.max(1);
    let target = cfg.expected_speakers.map(|k| k.clamp(1, max_speakers));

    let merge = |a: usize, b: usize, members: &mut Vec<Option<Vec<usize>>>, linkage: &mut Vec<Vec<f32>>, seconds: &mut Vec<f32>| {
        let (na, nb) = (
            members[a].as_ref().map_or(0, |m| m.len()) as f32,
            members[b].as_ref().map_or(0, |m| m.len()) as f32,
        );
        for c in 0..n {
            if c == a || c == b || members[c].is_none() {
                continue;
            }
            let l = (na * linkage[a][c] + nb * linkage[b][c]) / (na + nb);
            linkage[a][c] = l;
            linkage[c][a] = l;
        }
        let moved = members[b].take().unwrap_or_default();
        if let Some(m) = members[a].as_mut() {
            m.extend(moved);
        }
        seconds[a] += seconds[b];
    };

    let best_pair = |members: &Vec<Option<Vec<usize>>>, linkage: &Vec<Vec<f32>>, only: Option<usize>| -> Option<(usize, usize, f32)> {
        let mut best: Option<(usize, usize, f32)> = None;
        for i in 0..n {
            if members[i].is_none() {
                continue;
            }
            for j in i + 1..n {
                if members[j].is_none() || (only.is_some() && only != Some(i) && only != Some(j)) {
                    continue;
                }
                if best.map_or(true, |(_, _, s)| linkage[i][j] > s) {
                    best = Some((i, j, linkage[i][j]));
                }
            }
        }
        best
    };

    // 1. Merge the most similar clusters until they are clearly different voices
    //    (or until the expected / maximum number of speakers is reached).
    while count > 1 {
        let Some((i, j, s)) = best_pair(&members, &linkage, None) else { break };
        let must_merge = match target {
            Some(k) => count > k,
            None => count > max_speakers,
        };
        let may_merge = match target {
            Some(k) => count > k,
            None => s >= cfg.merge_threshold,
        };
        if !(must_merge || may_merge) {
            break;
        }
        merge(i, j, &mut members, &mut linkage, &mut seconds);
        count -= 1;
    }

    // 2. Fold clusters with too little speech into their nearest neighbour: a real participant who
    //    said almost nothing is far less likely than a stray voice vector of a known speaker.
    loop {
        if count <= 1 || target.map_or(false, |k| count <= k) {
            break;
        }
        let smallest = (0..n)
            .filter(|&c| members[c].is_some() && seconds[c] < cfg.min_speaker_seconds)
            .min_by(|&a, &b| seconds[a].partial_cmp(&seconds[b]).unwrap_or(std::cmp::Ordering::Equal));
        let Some(small) = smallest else { break };
        let Some((i, j, _)) = best_pair(&members, &linkage, Some(small)) else { break };
        let (into, from) = if i == small { (j, i) } else { (i, j) };
        merge(into, from, &mut members, &mut linkage, &mut seconds);
        count -= 1;
    }

    members.into_iter().flatten().collect()
}

/// Maps a similarity margin (own cluster vs. best other) to a 0..1 confidence.
fn margin_confidence(margin: f32) -> f32 {
    (0.5 + margin / 0.6).clamp(0.05, 1.0)
}

/// Clusters all segments of a meeting. `labels[i]` corresponds to `segments[i]`.
pub fn cluster_segments(segments: &[VoiceSegment], cfg: &OfflineConfig) -> OfflineResult {
    let usable: Vec<usize> = (0..segments.len())
        .filter(|&i| !segments[i].is_room && segments[i].embedding.is_some())
        .collect();
    let mut anchors: Vec<usize> = usable
        .iter()
        .copied()
        .filter(|&i| segments[i].seconds >= cfg.anchor_min_seconds)
        .collect();
    if anchors.is_empty() {
        // Only short segments: better a rough grouping than nothing.
        anchors = usable.clone();
    }

    let mut clusters = agglomerate(segments, &anchors, cfg);
    // Stable numbering: order clusters by their first segment in the meeting.
    for c in clusters.iter_mut() {
        c.sort_unstable();
    }
    clusters.sort_by_key(|c| c.first().copied().unwrap_or(usize::MAX));

    let centroids: Vec<Vec<f32>> = clusters.iter().map(|m| centroid_of(segments, m, cfg)).collect();
    let mut anchor_cluster: HashMap<usize, usize> = HashMap::new();
    for (ci, m) in clusters.iter().enumerate() {
        for &s in m {
            anchor_cluster.insert(s, ci);
        }
    }

    let labels = segments
        .iter()
        .enumerate()
        .map(|(i, seg)| {
            if seg.is_room {
                return OfflineLabel::Room { confidence: seg.room_confidence };
            }
            let Some(e) = seg.embedding.as_deref() else { return OfflineLabel::Unknown };
            if centroids.is_empty() {
                return OfflineLabel::Unknown;
            }
            let sims: Vec<f32> = centroids.iter().map(|c| cosine(e, c)).collect();
            let best = (0..sims.len())
                .max_by(|&a, &b| sims[a].partial_cmp(&sims[b]).unwrap_or(std::cmp::Ordering::Equal))
                .unwrap_or(0);
            let (cluster, is_anchor) = match anchor_cluster.get(&i) {
                Some(&c) => (c, true),
                None => (best, false),
            };
            if !is_anchor && sims[cluster] < cfg.room_min_similarity {
                return OfflineLabel::Room { confidence: 0.5 };
            }
            let other = (0..sims.len())
                .filter(|&c| c != cluster)
                .map(|c| sims[c])
                .fold(f32::NEG_INFINITY, f32::max);
            let margin = if other.is_finite() { sims[cluster] - other } else { sims[cluster] - cfg.merge_threshold };
            let confidence = margin_confidence(margin) * if is_anchor { 1.0 } else { 0.85 };
            OfflineLabel::Person { cluster, confidence }
        })
        .collect();

    OfflineResult { labels, centroids }
}

/// Decides which existing speaker each new cluster becomes, so names and colors survive.
///
/// * `current[i]` — the speaker currently *displayed* for segment `i` (manual override or automatic,
///   resolved through merges; room excluded), i.e. what the user has seen and possibly corrected.
///
/// Clusters are matched one-to-one, first by the amount of speech they share with an existing
/// speaker, then (for clusters sharing nothing) by voice similarity to the remaining speakers.
/// Returns, per cluster, `Some(existing speaker id)` or `None` when a new speaker is needed.
pub fn map_clusters_to_speakers(
    result: &OfflineResult,
    segments: &[VoiceSegment],
    current: &[Option<String>],
    cfg: &OfflineConfig,
) -> Vec<Option<String>> {
    let k = result.cluster_count();
    let mut mapping: Vec<Option<String>> = vec![None; k];

    // Shared speech (seconds) between every cluster and every existing speaker.
    let mut shared: HashMap<(usize, &str), f32> = HashMap::new();
    for (i, cur) in current.iter().enumerate() {
        if let (Some(c), Some(sp)) = (result.cluster_of(i), cur.as_deref()) {
            *shared.entry((c, sp)).or_insert(0.0) += segments.get(i).map_or(0.0, |s| s.seconds.max(0.1));
        }
    }
    let mut pairs: Vec<((usize, &str), f32)> = shared.into_iter().collect();
    pairs.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.0 .0.cmp(&b.0 .0))
            .then_with(|| a.0 .1.cmp(b.0 .1))
    });
    let mut taken: std::collections::HashSet<String> = std::collections::HashSet::new();
    for ((cluster, speaker), _) in pairs {
        if mapping[cluster].is_none() && !taken.contains(speaker) {
            mapping[cluster] = Some(speaker.to_string());
            taken.insert(speaker.to_string());
        }
    }

    // Voice of every existing speaker that is still free.
    let mut by_speaker: HashMap<&str, Vec<usize>> = HashMap::new();
    for (i, cur) in current.iter().enumerate() {
        if let Some(sp) = cur.as_deref() {
            if !taken.contains(sp) && segments.get(i).map_or(false, |s| s.embedding.is_some() && !s.is_room) {
                by_speaker.entry(sp).or_default().push(i);
            }
        }
    }
    let mut free: Vec<(&str, Vec<f32>)> =
        by_speaker.into_iter().map(|(sp, m)| (sp, centroid_of(segments, &m, cfg))).collect();
    free.sort_by(|a, b| a.0.cmp(b.0));
    for (cluster, slot) in mapping.iter_mut().enumerate() {
        if slot.is_some() {
            continue;
        }
        let best = free
            .iter()
            .enumerate()
            .map(|(fi, (_, c))| (fi, cosine(c, &result.centroids[cluster])))
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        if let Some((fi, s)) = best {
            if s >= cfg.inherit_min_similarity {
                let (sp, _) = free.remove(fi);
                *slot = Some(sp.to_string());
            }
        }
    }
    mapping
}

#[cfg(test)]
mod tests {
    use super::*;

    const DIM: usize = 16;

    /// Unit vector mostly along `axis`, with a small deterministic perturbation.
    fn voice(axis: usize, jitter: usize) -> Vec<f32> {
        let mut v = vec![0.0f32; DIM];
        v[axis] = 1.0;
        v[(axis + 1 + jitter % 3) % DIM] += 0.25;
        v[(axis + 5 + jitter % 4) % DIM] -= 0.15;
        l2_normalize(&mut v);
        v
    }

    fn seg(axis: usize, jitter: usize, seconds: f32) -> VoiceSegment {
        VoiceSegment { embedding: Some(voice(axis, jitter)), seconds, ..Default::default() }
    }

    fn clusters(r: &OfflineResult) -> Vec<Option<usize>> {
        (0..r.labels.len()).map(|i| r.cluster_of(i)).collect()
    }

    #[test]
    fn separates_three_voices_and_numbers_by_first_appearance() {
        let segs = vec![seg(3, 0, 4.0), seg(7, 1, 3.0), seg(3, 2, 5.0), seg(11, 0, 6.0), seg(7, 3, 2.0), seg(3, 1, 2.5)];
        let r = cluster_segments(&segs, &OfflineConfig::default());
        assert_eq!(r.cluster_count(), 3);
        assert_eq!(clusters(&r), vec![Some(0), Some(1), Some(0), Some(2), Some(1), Some(0)]);
        for l in &r.labels {
            if let OfflineLabel::Person { confidence, .. } = l {
                assert!(*confidence > 0.7, "clear voices should be confident: {confidence}");
            }
        }
    }

    #[test]
    fn corrects_an_early_live_mistake_by_seeing_the_whole_meeting() {
        // A live clusterer would split or merge depending on order; offline sees everything at once.
        let segs = vec![seg(2, 0, 2.0), seg(2, 1, 6.0), seg(9, 0, 6.0), seg(2, 2, 6.0), seg(9, 1, 6.0)];
        let r = cluster_segments(&segs, &OfflineConfig::default());
        assert_eq!(clusters(&r), vec![Some(0), Some(0), Some(1), Some(0), Some(1)]);
    }

    #[test]
    fn expected_speakers_forces_the_count() {
        let segs = vec![seg(1, 0, 4.0), seg(5, 0, 4.0), seg(9, 0, 4.0), seg(13, 0, 4.0)];
        let mut cfg = OfflineConfig::default();
        assert_eq!(cluster_segments(&segs, &cfg).cluster_count(), 4);
        cfg.expected_speakers = Some(2);
        assert_eq!(cluster_segments(&segs, &cfg).cluster_count(), 2);
        cfg.expected_speakers = None;
        cfg.max_speakers = 3;
        assert_eq!(cluster_segments(&segs, &cfg).cluster_count(), 3);
    }

    #[test]
    fn tiny_clusters_are_folded_into_a_neighbour() {
        // one stray 1.6 s segment of an odd voice next to two well-established speakers
        let segs = vec![seg(1, 0, 8.0), seg(6, 0, 8.0), seg(12, 0, 1.6), seg(1, 1, 8.0), seg(6, 1, 8.0)];
        let r = cluster_segments(&segs, &OfflineConfig::default());
        assert_eq!(r.cluster_count(), 2);
        assert!(r.cluster_of(2).is_some());
    }

    #[test]
    fn short_segments_join_the_nearest_speaker_and_room_and_unknown_pass_through() {
        let mut segs = vec![seg(4, 0, 5.0), seg(10, 0, 5.0), seg(4, 1, 0.8)];
        segs.push(VoiceSegment { is_room: true, room_confidence: 0.9, seconds: 4.0, ..Default::default() });
        segs.push(VoiceSegment { seconds: 0.3, ..Default::default() });
        // a short segment unlike anyone -> room
        let mut odd = vec![0.0f32; DIM];
        odd[14] = 1.0;
        segs.push(VoiceSegment { embedding: Some(odd), seconds: 0.9, ..Default::default() });
        let r = cluster_segments(&segs, &OfflineConfig::default());
        assert_eq!(r.cluster_count(), 2);
        assert_eq!(r.cluster_of(2), Some(0));
        match r.labels[2] {
            OfflineLabel::Person { confidence, .. } => assert!(confidence <= 0.85 + 1e-6),
            other => panic!("unexpected {other:?}"),
        }
        assert_eq!(r.labels[3], OfflineLabel::Room { confidence: 0.9 });
        assert_eq!(r.labels[4], OfflineLabel::Unknown);
        assert!(matches!(r.labels[5], OfflineLabel::Room { .. }));
    }

    #[test]
    fn empty_and_only_short_meetings() {
        let r = cluster_segments(&[], &OfflineConfig::default());
        assert_eq!(r.cluster_count(), 0);
        let segs = vec![seg(3, 0, 0.8), seg(3, 1, 0.9)];
        let r = cluster_segments(&segs, &OfflineConfig::default());
        assert_eq!(r.cluster_count(), 1);
        assert_eq!(clusters(&r), vec![Some(0), Some(0)]);
    }

    #[test]
    fn mapping_keeps_existing_speakers_by_shared_speech() {
        let segs = vec![seg(1, 0, 4.0), seg(8, 0, 4.0), seg(1, 1, 4.0), seg(8, 1, 4.0), seg(13, 0, 5.0)];
        let r = cluster_segments(&segs, &OfflineConfig::default());
        assert_eq!(r.cluster_count(), 3);
        // live result had voice 1 as "anna" except one segment; voice 8 as "bob"; voice 13 never seen
        let current = vec![Some("anna".to_string()), Some("bob".to_string()), Some("bob".to_string()), Some("bob".to_string()), None];
        let m = map_clusters_to_speakers(&r, &segs, &current, &OfflineConfig::default());
        assert_eq!(m, vec![Some("anna".to_string()), Some("bob".to_string()), None]);
    }

    #[test]
    fn mapping_uses_voice_similarity_when_nothing_is_shared() {
        // Existing speaker "ewa" shares no clustered segment with the new cluster (her only segment
        // stayed unknown offline), but her voice matches it.
        let segs = vec![seg(5, 0, 4.0), seg(5, 1, 4.0), seg(5, 2, 4.0)];
        let r = OfflineResult {
            labels: vec![
                OfflineLabel::Person { cluster: 0, confidence: 1.0 },
                OfflineLabel::Person { cluster: 0, confidence: 1.0 },
                OfflineLabel::Unknown,
            ],
            centroids: vec![voice(5, 0)],
        };
        let current = vec![None, None, Some("ewa".to_string())];
        let m = map_clusters_to_speakers(&r, &segs, &current, &OfflineConfig::default());
        assert_eq!(m, vec![Some("ewa".to_string())]);
        // a different voice does not inherit her
        let r2 = OfflineResult { centroids: vec![voice(12, 0)], ..r };
        assert_eq!(map_clusters_to_speakers(&r2, &segs, &current, &OfflineConfig::default()), vec![None]);
    }

    #[test]
    fn mapping_is_one_to_one() {
        // two clusters both mostly overlapping one existing speaker: only one may take it
        let segs = vec![seg(1, 0, 6.0), seg(9, 0, 4.0)];
        let r = cluster_segments(&segs, &OfflineConfig::default());
        assert_eq!(r.cluster_count(), 2);
        let current = vec![Some("x".to_string()), Some("x".to_string())];
        let m = map_clusters_to_speakers(&r, &segs, &current, &OfflineConfig::default());
        assert_eq!(m, vec![Some("x".to_string()), None]);
    }
}
