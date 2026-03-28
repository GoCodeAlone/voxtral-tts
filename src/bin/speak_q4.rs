//! CLI for Q4 GGUF Voxtral TTS (text-to-speech synthesis).

use anyhow::{bail, Context, Result};
use burn::backend::Wgpu;
use burn::tensor::Tensor;
use clap::Parser;
use std::path::PathBuf;
use std::time::Instant;
use tracing::info;

use voxtral_mini_realtime::audio::AudioBuffer;
use voxtral_mini_realtime::gguf::Q4TtsModelLoader;
use voxtral_mini_realtime::tokenizer::TekkenEncoder;
use voxtral_mini_realtime::tts::config::{AudioCodebookLayout, TtsSpecialTokens};
use voxtral_mini_realtime::tts::embeddings::AudioCodebookEmbeddings;
use voxtral_mini_realtime::tts::voice::load_voice_from_bytes;

type Backend = Wgpu;

#[derive(Parser)]
#[command(name = "voxtral-speak-q4")]
#[command(about = "Synthesize speech using Q4 GGUF Voxtral TTS")]
struct Cli {
    /// Text to synthesize.
    #[arg(short, long)]
    text: String,

    /// Voice preset name (e.g., "casual_female").
    #[arg(short, long, default_value = "casual_female")]
    voice: String,

    /// Path to Q4 GGUF model file.
    #[arg(short, long, default_value = "models/voxtral-tts-q4.gguf")]
    model: String,

    /// Directory containing voice_embedding/*.safetensors files.
    #[arg(long, default_value = "models/voxtral-tts/voice_embedding")]
    voices_dir: String,

    /// Path to Tekken tokenizer JSON.
    #[arg(long, default_value = "models/voxtral-tts/tekken.json")]
    tokenizer: String,

    /// Output WAV file path.
    #[arg(short, long, default_value = "output.wav")]
    output: String,

    /// Maximum number of audio frames to generate.
    #[arg(long, default_value_t = 2000)]
    max_frames: usize,

    /// Number of Euler ODE steps (3 = real-time, 4 = safe default, 8 = highest quality).
    #[arg(long, default_value_t = 4)]
    euler_steps: usize,
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_target(false)
        .with_writer(std::io::stderr)
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    let cli = Cli::parse();
    let device = burn::backend::wgpu::WgpuDevice::default();

    // Load model
    let start = Instant::now();
    info!("Loading Q4 TTS model from {}", cli.model);
    let mut loader =
        Q4TtsModelLoader::from_file(&PathBuf::from(&cli.model)).context("Failed to open GGUF")?;
    let (backbone, mut fm, codec) = loader.load(&device).context("Failed to load Q4 model")?;
    info!(
        elapsed_ms = start.elapsed().as_millis() as u64,
        "Q4 TTS model loaded"
    );

    fm.set_euler_steps(cli.euler_steps);
    info!(euler_steps = cli.euler_steps, "Euler steps configured");

    // Load tokenizer
    let tokenizer_json = std::fs::read_to_string(&cli.tokenizer)
        .with_context(|| format!("Failed to read tokenizer: {}", cli.tokenizer))?;
    let tokenizer =
        TekkenEncoder::from_json(&tokenizer_json).context("Failed to load tokenizer")?;

    // Tokenize
    let token_ids = tokenizer.encode(&cli.text);
    info!(text = %cli.text, n_tokens = token_ids.len(), "Text tokenized");

    // Load voice
    let voice_path = PathBuf::from(&cli.voices_dir).join(format!("{}.safetensors", cli.voice));
    if !voice_path.exists() {
        bail!(
            "Voice '{}' not found at {}",
            cli.voice,
            voice_path.display()
        );
    }
    let voice_bytes = std::fs::read(&voice_path)?;
    let voice_embed: Tensor<Backend, 2> =
        load_voice_from_bytes(&voice_bytes, 3072, &device).context("Failed to load voice")?;
    info!(
        voice = %cli.voice,
        frames = voice_embed.dims()[0],
        "Voice loaded"
    );

    // Build input sequence
    let special = TtsSpecialTokens::default();
    let bos = backbone.embed_tokens_from_ids(&[special.bos_token_id as i32], 1, 1);
    let begin_audio = backbone.embed_tokens_from_ids(&[special.begin_audio_token_id as i32], 1, 1);
    let next_audio_text =
        backbone.embed_tokens_from_ids(&[special.next_audio_text_token_id as i32], 1, 1);
    let repeat_audio_text =
        backbone.embed_tokens_from_ids(&[special.repeat_audio_text_token_id as i32], 1, 1);
    let text_ids_i32: Vec<i32> = token_ids.iter().map(|&id| id as i32).collect();
    let text_embeds = backbone.embed_tokens_from_ids(&text_ids_i32, 1, text_ids_i32.len());

    let input_sequence = Tensor::cat(
        vec![
            bos,
            begin_audio.clone(),
            voice_embed.unsqueeze_dim::<3>(0),
            next_audio_text,
            text_embeds,
            repeat_audio_text,
            begin_audio,
        ],
        1,
    );
    let [_, seq_len, _] = input_sequence.dims();
    info!(seq_len, "Input sequence built");

    // Generate
    let codebook = AudioCodebookEmbeddings::new(
        backbone.audio_codebook_embeddings().clone(),
        AudioCodebookLayout::default(),
    );

    let gen_start = Instant::now();
    let frames =
        pollster::block_on(backbone.generate_async(input_sequence, &fm, &codebook, cli.max_frames))
            .map_err(|e| anyhow::anyhow!("Generation failed: {e}"))?;
    info!(
        elapsed_ms = gen_start.elapsed().as_millis() as u64,
        frames = frames.len(),
        "Audio frames generated"
    );

    if frames.is_empty() {
        bail!("No audio frames generated");
    }

    // Codec decode
    let n_frames = frames.len();
    let semantic_indices: Vec<usize> = frames.iter().map(|f| f.semantic_idx).collect();
    let mut acoustic_data = Vec::with_capacity(n_frames * 36);
    for frame in &frames {
        for &level in &frame.acoustic_levels {
            acoustic_data.push(level as f32);
        }
    }
    let acoustic_tensor: Tensor<Backend, 2> = Tensor::from_data(
        burn::tensor::TensorData::new(acoustic_data, [n_frames, 36]),
        &device,
    );

    let waveform = codec.decode(&semantic_indices, acoustic_tensor);
    let [_batch, total_samples] = waveform.dims();

    let wav_data = waveform.to_data();
    let mut samples: Vec<f32> = wav_data.as_slice::<f32>().unwrap()[..total_samples].to_vec();

    // Peak normalize
    let peak = samples.iter().map(|s| s.abs()).fold(0.0f32, f32::max);
    if peak > 1e-6 {
        let gain = 0.95 / peak;
        for s in &mut samples {
            *s *= gain;
        }
    }

    let audio = AudioBuffer::new(samples, 24000);
    let output_path = PathBuf::from(&cli.output);
    audio
        .save(&output_path)
        .with_context(|| format!("Failed to save {}", output_path.display()))?;

    let duration = audio.len() as f64 / audio.sample_rate as f64;
    let total_ms = start.elapsed().as_millis();
    println!(
        "Saved {:.2}s of audio to {} (total {:.1}s, gen RTF {:.2}x)",
        duration,
        output_path.display(),
        total_ms as f64 / 1000.0,
        gen_start.elapsed().as_millis() as f64 / 1000.0 / duration,
    );

    Ok(())
}
