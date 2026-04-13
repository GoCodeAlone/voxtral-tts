//! # voxtral-tts
//!
//! Streaming TTS lib crate for the Core Dump sidecar.
//! Three-stage pipeline: Q4 Ministral backbone → flow-matching transformer → codec decoder → 24 kHz PCM.

pub mod audio;
pub mod audio_output;
pub mod engine;
pub mod ring_buffer;
#[cfg(feature = "wgpu")]
pub mod gguf;
pub mod models;
pub mod profiling;
pub mod tokenizer;
pub mod tts;

#[cfg(test)]
mod test_utils;

// Re-exports
pub use audio::AudioBuffer;
pub use audio_output::AudioOutputStream;
pub use engine::{SpeakConfig, SpeakResult, TtsEngine};
pub use ring_buffer::AudioRingBuffer;
pub use tts::backbone::{GeneratedFrame, TtsBackbone};
pub use tts::pipeline::TtsPipeline;
pub use tts::voice::VoiceRegistry;
