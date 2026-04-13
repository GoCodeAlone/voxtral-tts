//! High-level TTS engine API for the Core Dump sidecar.
//!
//! `TtsEngine` wraps the Q4 backbone, FM transformer, codec decoder, tokenizer,
//! voice registry, and cpal audio output stream into a single `speak()` call.
//! Audio streams to the speakers frame-by-frame via a ring buffer — no temp files.

use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use burn::backend::wgpu::WgpuDevice;
use burn::backend::Wgpu;
use burn::tensor::{Tensor, TensorData};

use crate::audio_output::AudioOutputStream;
use crate::tts::codec::CodecDecoder;
use crate::tts::config::{AudioCodebookLayout, TtsSpecialTokens, VoiceEmbeddingConfig};
use crate::tts::embeddings::AudioCodebookEmbeddings;
use crate::tts::voice::VoiceRegistry;
use crate::tokenizer::TekkenEncoder;

use crate::gguf::tts_loader::Q4TtsModelLoader;
use crate::gguf::tts_model::{Q4FmTransformer, Q4TtsBackbone};

/// Configuration for a single `speak()` call.
pub struct SpeakConfig {
    /// Flow-matching Euler ODE steps: 3=fast, 4=balanced, 8=quality.
    pub euler_steps: usize,
    /// Whether to use GPU acceleration.
    pub use_gpu: bool,
    /// Maximum number of audio frames to generate (~20s at 12.5 fps).
    pub max_frames: usize,
}

impl Default for SpeakConfig {
    fn default() -> Self {
        Self {
            euler_steps: 4,
            use_gpu: true,
            max_frames: 2000,
        }
    }
}

/// Timing and quality metrics returned by `speak()`.
pub struct SpeakResult {
    /// Total audio duration in milliseconds.
    pub duration_ms: u64,
    /// Wall-clock time for the full generation + drain in milliseconds.
    pub generation_ms: u64,
    /// generation_ms / duration_ms. < 1.0 means faster than real-time.
    pub rtf: f64,
    /// Time from `speak()` call to the first frame reaching cpal (ms).
    pub ttfa_ms: u64,
    /// Number of audio frames generated.
    pub frames_generated: u32,
}

/// Streaming TTS engine backed by Q4-quantized Voxtral.
///
/// Holds all inference state (backbone, FM, codec, tokenizer, voices, audio device).
/// The cpal output stream is kept open across calls to avoid per-utterance overhead.
/// `fm` is wrapped in a Mutex so `speak()` can configure euler_steps per call.
pub struct TtsEngine {
    backbone: Q4TtsBackbone,
    fm: Mutex<Q4FmTransformer>,
    codec: CodecDecoder<Wgpu>,
    tokenizer: TekkenEncoder,
    voices: VoiceRegistry,
    audio_out: AudioOutputStream,
    device: WgpuDevice,
}

impl TtsEngine {
    /// Load the TTS engine from a model directory.
    ///
    /// Expected layout:
    /// ```text
    /// <model_dir>/
    ///   voxtral-tts-q4.gguf
    ///   tekken.json
    ///   voice_embedding/
    ///     casual_female.safetensors
    ///     ...
    /// ```
    pub fn load(model_dir: &Path) -> Result<Self, String> {
        let gguf_path = model_dir.join("voxtral-tts-q4.gguf");
        let tekken_path = model_dir.join("tekken.json");
        let voice_dir = model_dir.join("voice_embedding");

        let device = WgpuDevice::default();

        // Load Q4 backbone, FM transformer, codec decoder
        let mut loader = Q4TtsModelLoader::from_file(&gguf_path)
            .map_err(|e| format!("failed to open GGUF {}: {e}", gguf_path.display()))?;
        let (backbone, fm, codec) = loader
            .load(&device)
            .map_err(|e| format!("failed to load Q4 model: {e}"))?;

        // Load tokenizer
        let tekken_json = std::fs::read_to_string(&tekken_path)
            .map_err(|e| format!("failed to read {}: {e}", tekken_path.display()))?;
        let tokenizer = TekkenEncoder::from_json(&tekken_json)
            .map_err(|e| format!("failed to load tokenizer: {e}"))?;

        // Load voice registry
        let voice_config = VoiceEmbeddingConfig::default();
        let voices = VoiceRegistry::from_directory(&voice_dir, &voice_config)
            .map_err(|e| format!("failed to load voices from {}: {e}", voice_dir.display()))?;

        // Open audio output stream (stays open across calls)
        let audio_out =
            AudioOutputStream::open(24000).map_err(|e| format!("failed to open audio: {e}"))?;

        Ok(Self {
            backbone,
            fm: Mutex::new(fm),
            codec,
            tokenizer,
            voices,
            audio_out,
            device,
        })
    }

    /// Synthesize speech from text, streaming audio to the default output device.
    ///
    /// Audio begins playing approximately 80ms after the first frame is generated.
    /// Returns timing metrics when synthesis and playback are complete.
    pub fn speak(
        &self,
        text: &str,
        voice_id: &str,
        config: &SpeakConfig,
    ) -> Result<SpeakResult, String> {
        let t0 = Instant::now();

        // Lock FM for the duration of this call; configure euler steps
        let mut fm = self
            .fm
            .lock()
            .map_err(|e| format!("fm lock poisoned: {e}"))?;
        fm.set_euler_steps(config.euler_steps);

        // Tokenize
        let token_ids = self.tokenizer.encode(text);
        if token_ids.is_empty() {
            return Err(format!("tokenizer produced no tokens for: {text:?}"));
        }

        // Load voice embedding
        let voice_embed = self
            .voices
            .load_voice::<Wgpu>(voice_id, &self.device)
            .map_err(|e| format!("failed to load voice '{voice_id}': {e}"))?;

        // Build input sequence
        let special = TtsSpecialTokens::default();
        let text_ids_i32: Vec<i32> = token_ids.iter().map(|&id| id as i32).collect();
        let input_sequence = self.build_input_sequence(voice_embed, &text_ids_i32, &special)?;

        // Build codebook
        let codebook_layout = AudioCodebookLayout::default();
        let codebook = AudioCodebookEmbeddings::new(
            self.backbone.audio_codebook_embeddings().clone(),
            codebook_layout,
        );

        // Stream frames through codec → ring buffer → cpal
        let ring = Arc::clone(self.audio_out.ring());
        let codec = &self.codec;
        let device = &self.device;

        let mut first_audio_time: Option<Instant> = None;
        let mut frame_count = 0u32;

        let result = pollster::block_on(self.backbone.generate_with_callback(
            input_sequence,
            &*fm,
            &codebook,
            config.max_frames,
            |frame| {
                // Codec decode this single frame → 1920 PCM samples
                let acoustic_data: Vec<f32> =
                    frame.acoustic_levels.iter().map(|&v| v as f32).collect();
                let acoustic_tensor = Tensor::<Wgpu, 2>::from_data(
                    TensorData::new(acoustic_data, [1, 36]),
                    device,
                );
                let waveform = codec.decode(&[frame.semantic_idx], acoustic_tensor);
                let [_batch, n_samples] = waveform.dims();
                let wav_data = waveform.to_data();
                let samples = match wav_data.as_slice::<f32>() {
                    Ok(s) => s[..n_samples].to_vec(),
                    Err(e) => {
                        tracing::error!("[tts] codec decode slice error: {e}");
                        return false;
                    }
                };

                ring.push(&samples);

                if first_audio_time.is_none() {
                    first_audio_time = Some(Instant::now());
                }
                frame_count += 1;
                true
            },
        ));

        let frames_generated = result.map_err(|e| format!("[tts] generate: {e}"))?;

        // Wait for cpal to drain the ring buffer (playback complete)
        self.audio_out.wait_drain();

        let generation_ms = t0.elapsed().as_millis() as u64;
        let duration_ms = (frames_generated as f64 / 12.5 * 1000.0) as u64;
        let rtf = if duration_ms > 0 {
            generation_ms as f64 / duration_ms as f64
        } else {
            0.0
        };
        let ttfa_ms = first_audio_time
            .map(|t| (t - t0).as_millis() as u64)
            .unwrap_or(0);

        let short_text = &text[..text.len().min(40)];
        eprintln!(
            "[tts] \"{short_text}\" → {frames_generated} frames, {:.2}s audio in {generation_ms}ms (RTF={rtf:.2}x) ttfa={ttfa_ms}ms",
            duration_ms as f64 / 1000.0,
        );

        Ok(SpeakResult {
            duration_ms,
            generation_ms,
            rtf,
            ttfa_ms,
            frames_generated,
        })
    }

    /// List available voice presets.
    pub fn list_voices(&self) -> Vec<&str> {
        self.voices.list_voices()
    }

    /// Build [1, seq_len, 3072] input embedding tensor for backbone prefill.
    fn build_input_sequence(
        &self,
        voice_embed: Tensor<Wgpu, 2>,
        text_ids: &[i32],
        special: &TtsSpecialTokens,
    ) -> Result<Tensor<Wgpu, 3>, String> {
        let bos = self
            .backbone
            .embed_tokens_from_ids(&[special.bos_token_id as i32], 1, 1);
        let begin_audio = self
            .backbone
            .embed_tokens_from_ids(&[special.begin_audio_token_id as i32], 1, 1);
        let next_audio_text = self
            .backbone
            .embed_tokens_from_ids(&[special.next_audio_text_token_id as i32], 1, 1);
        let repeat_audio_text = self
            .backbone
            .embed_tokens_from_ids(&[special.repeat_audio_text_token_id as i32], 1, 1);
        let text_embeds = self
            .backbone
            .embed_tokens_from_ids(text_ids, 1, text_ids.len());

        let voice_3d = voice_embed.unsqueeze_dim::<3>(0);

        Ok(Tensor::cat(
            vec![
                bos,
                begin_audio.clone(),
                voice_3d,
                next_audio_text,
                text_embeds,
                repeat_audio_text,
                begin_audio,
            ],
            1,
        ))
    }
}
