//! Kaldi-compatible log-mel filterbank features (80 bins, 25 ms / 10 ms, Hamming window),
//! as expected by WeSpeaker speaker-embedding models. Verified against `kaldi-native-fbank`
//! (see `tests/fixtures`).

use realfft::{RealFftPlanner, RealToComplex};
use std::f32::consts::PI;
use std::sync::Arc;

pub const SAMPLE_RATE: usize = 16_000;
const FRAME_LEN: usize = 400; // 25 ms
const FRAME_SHIFT: usize = 160; // 10 ms
const FFT_SIZE: usize = 512;
const PREEMPH: f32 = 0.97;
const LOW_FREQ: f32 = 20.0;
const EPSILON: f32 = 1.1920929e-7; // f32::EPSILON, Kaldi's floor for log

pub struct FbankExtractor {
    num_bins: usize,
    window: Vec<f32>,
    /// Triangular mel weights per bin over `FFT_SIZE/2` FFT bins (Nyquist bin excluded, as in Kaldi).
    mel_weights: Vec<Vec<f32>>,
    fft: Arc<dyn RealToComplex<f32>>,
}

fn mel(freq: f32) -> f32 {
    1127.0 * (1.0 + freq / 700.0).ln()
}

impl FbankExtractor {
    pub fn new(num_bins: usize) -> Self {
        let window: Vec<f32> = (0..FRAME_LEN)
            .map(|i| 0.54 - 0.46 * (2.0 * PI * i as f32 / (FRAME_LEN as f32 - 1.0)).cos())
            .collect();

        let nyquist = SAMPLE_RATE as f32 / 2.0;
        let fft_bin_width = SAMPLE_RATE as f32 / FFT_SIZE as f32;
        let mel_low = mel(LOW_FREQ);
        let mel_high = mel(nyquist);
        let mel_delta = (mel_high - mel_low) / (num_bins as f32 + 1.0);
        let half = FFT_SIZE / 2;

        let mel_weights = (0..num_bins)
            .map(|b| {
                let left = mel_low + b as f32 * mel_delta;
                let center = mel_low + (b as f32 + 1.0) * mel_delta;
                let right = mel_low + (b as f32 + 2.0) * mel_delta;
                (0..half)
                    .map(|i| {
                        let m = mel(fft_bin_width * i as f32);
                        if m > left && m < right {
                            if m <= center {
                                (m - left) / (center - left)
                            } else {
                                (right - m) / (right - center)
                            }
                        } else {
                            0.0
                        }
                    })
                    .collect()
            })
            .collect();

        let fft = RealFftPlanner::<f32>::new().plan_fft_forward(FFT_SIZE);
        Self { num_bins, window, mel_weights, fft }
    }

    pub fn num_bins(&self) -> usize {
        self.num_bins
    }

    /// Number of frames for `num_samples` samples (Kaldi `snip_edges = true`).
    pub fn num_frames(num_samples: usize) -> usize {
        if num_samples < FRAME_LEN {
            0
        } else {
            1 + (num_samples - FRAME_LEN) / FRAME_SHIFT
        }
    }

    /// Computes log-mel features for 16 kHz mono samples scaled to the int16 range
    /// (WeSpeaker models use `normalize_samples = 0`). Returns `frames x num_bins`, row-major.
    pub fn compute(&self, samples: &[f32]) -> Vec<f32> {
        let frames = Self::num_frames(samples.len());
        let mut out = Vec::with_capacity(frames * self.num_bins);
        let mut frame = vec![0.0f32; FRAME_LEN];
        let mut input = self.fft.make_input_vec();
        let mut spectrum = self.fft.make_output_vec();
        let mut scratch = self.fft.make_scratch_vec();

        for f in 0..frames {
            let start = f * FRAME_SHIFT;
            frame.copy_from_slice(&samples[start..start + FRAME_LEN]);

            // remove DC offset
            let mean = frame.iter().sum::<f32>() / FRAME_LEN as f32;
            frame.iter_mut().for_each(|x| *x -= mean);
            // pre-emphasis (first sample uses itself as the previous one, as in Kaldi)
            for i in (1..FRAME_LEN).rev() {
                frame[i] -= PREEMPH * frame[i - 1];
            }
            frame[0] -= PREEMPH * frame[0];
            // window + zero-pad to FFT size
            for i in 0..FRAME_LEN {
                input[i] = frame[i] * self.window[i];
            }
            input[FRAME_LEN..].iter_mut().for_each(|x| *x = 0.0);

            self.fft
                .process_with_scratch(&mut input, &mut spectrum, &mut scratch)
                .expect("fft size mismatch");
            let power: Vec<f32> = spectrum[..FFT_SIZE / 2].iter().map(|c| c.re * c.re + c.im * c.im).collect();

            for weights in &self.mel_weights {
                let energy: f32 = weights.iter().zip(&power).map(|(w, p)| w * p).sum();
                out.push(energy.max(EPSILON).ln());
            }
        }
        out
    }
}

/// Subtracts the per-dimension mean over time (cepstral mean normalization), in place.
pub fn mean_normalize(features: &mut [f32], num_bins: usize) {
    if num_bins == 0 || features.is_empty() {
        return;
    }
    let frames = features.len() / num_bins;
    for d in 0..num_bins {
        let mean = (0..frames).map(|t| features[t * num_bins + d]).sum::<f32>() / frames as f32;
        for t in 0..frames {
            features[t * num_bins + d] -= mean;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn sines(n: usize) -> Vec<f32> {
        (0..n)
            .map(|i| {
                let t = i as f64;
                (8000.0 * (2.0 * std::f64::consts::PI * 220.0 * t / 16000.0).sin()
                    + 3000.0 * (2.0 * std::f64::consts::PI * 1375.0 * t / 16000.0 + 0.3).sin()
                    + 1500.0 * (2.0 * std::f64::consts::PI * 3100.0 * t / 16000.0).sin()) as f32
            })
            .collect()
    }

    #[test]
    fn matches_kaldi_native_fbank() {
        let expected: Vec<f32> = include_bytes!("testdata/sines_fbank_80.f32")
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();
        let extractor = FbankExtractor::new(80);
        let got = extractor.compute(&sines(8000));
        assert_eq!(got.len(), expected.len());
        let max_err = got.iter().zip(&expected).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        println!("max abs error = {max_err}");
        assert!(max_err < 0.02, "max abs error {max_err}");
    }

    #[test]
    fn frame_count_follows_snip_edges() {
        assert_eq!(FbankExtractor::num_frames(399), 0);
        assert_eq!(FbankExtractor::num_frames(400), 1);
        assert_eq!(FbankExtractor::num_frames(559), 1);
        assert_eq!(FbankExtractor::num_frames(560), 2);
        assert_eq!(FbankExtractor::new(80).compute(&vec![0.0; 300]).len(), 0);
    }
}
