//! Speaker diarization: recognizing different speakers by voice.
//!
//! Pipeline per speech segment: 16 kHz samples -> Kaldi fbank -> WeSpeaker ONNX embedding
//! -> online clustering -> meeting-local speaker index + confidence.
//!
//! Status: core engine (not yet wired into the transcription worker or exposed via commands).

pub mod clusterer;
pub mod embedder;
pub mod error;
pub mod fbank;
pub mod identifier;
pub mod model;

pub use clusterer::{Assignment, ClustererConfig, OnlineClusterer};
pub use error::DiarizationError;
pub use identifier::{SpeakerIdentifier, SpeakerTag};
