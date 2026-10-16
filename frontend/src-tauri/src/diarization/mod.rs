//! Speaker diarization: recognizing different speakers by voice.
//!
//! Pipeline per speech segment: 16 kHz samples -> Kaldi fbank -> WeSpeaker ONNX embedding
//! -> online clustering -> meeting-local speaker index + confidence.
//!
//! After a recording (or on demand) `rediarize` re-clusters the whole meeting offline
//! (`offline`) and rewrites only the automatic speakers.
//!
//! The live identifier is created per recording by `commands::create_identifier` and driven by the
//! transcription worker; settings and model download are exposed as Tauri commands.

pub mod clusterer;
pub mod commands;
pub mod config;
pub mod embedder;
pub mod error;
pub mod fbank;
pub mod identifier;
pub mod model;
pub mod offline;
pub mod rediarize;

pub use clusterer::{Assignment, ClustererConfig, OnlineClusterer};
pub use error::DiarizationError;
pub use identifier::{IdentifierConfig, SpeakerIdentifier, SpeakerKind, SpeakerTag};
