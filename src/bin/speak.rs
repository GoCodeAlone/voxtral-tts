//! CLI for Voxtral TTS (text-to-speech synthesis).

use anyhow::{bail, Context, Result};
use burn::backend::Wgpu;
use clap::Parser;
use std::path::PathBuf;
use std::time::Instant;
use tracing::info;

use voxtral_mini_realtime::tokenizer::TekkenEncoder;
use voxtral_mini_realtime::tts::pipeline::TtsPipeline;

type Backend = Wgpu;

#[derive(Parser)]
#[command(name = "voxtral-speak")]
#[command(about = "Synthesize speech using Voxtral TTS 4B")]
struct Cli {
    /// Text to synthesize.
    #[arg(short, long, required_unless_present_any = ["list_voices", "token_ids"])]
    text: Option<String>,

    /// Voice preset name (e.g., "alloy", "breeze", "casual_female").
    #[arg(short, long, default_value = "alloy")]
    voice: String,

    /// Path to TTS model directory (containing consolidated.safetensors and voice_embedding/).
    #[arg(short, long, default_value = "models/voxtral-tts")]
    model: String,

    /// Output WAV file path.
    #[arg(short, long, default_value = "output.wav")]
    output: String,

    /// Path to Tekken tokenizer JSON (defaults to <model>/tekken.json).
    #[arg(long)]
    tokenizer: Option<String>,

    /// List available voice presets and exit.
    #[arg(long)]
    list_voices: bool,

    /// Maximum number of audio frames to generate.
    #[arg(long, default_value_t = 2000)]
    max_frames: usize,

    /// Pre-tokenized token IDs (comma-separated). Bypasses Tekken text tokenization.
    #[arg(long, value_delimiter = ',')]
    token_ids: Option<Vec<u32>>,
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_target(false)
        .with_writer(std::io::stderr)
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    let cli = Cli::parse();
    let device = Default::default();

    let model_dir = PathBuf::from(&cli.model);
    if !model_dir.join("consolidated.safetensors").exists() {
        bail!(
            "TTS model not found at {}\nDownload the TTS model weights to {}",
            model_dir.join("consolidated.safetensors").display(),
            model_dir.display()
        );
    }

    // Load pipeline
    let start = Instant::now();
    info!("Loading TTS pipeline from {}", model_dir.display());
    let pipeline = TtsPipeline::<Backend>::from_model_dir(&model_dir, &device)
        .context("Failed to load TTS pipeline")?;
    info!(
        elapsed_ms = start.elapsed().as_millis() as u64,
        "TTS pipeline loaded"
    );

    // Handle --list-voices
    if cli.list_voices {
        let voices = pipeline.list_voices();
        println!("Available voices ({}):", voices.len());
        for name in &voices {
            println!("  {name}");
        }
        return Ok(());
    }

    let text = cli.text.as_deref().unwrap_or_default();
    if text.is_empty() && cli.token_ids.is_none() {
        bail!("--text or --token-ids is required for synthesis");
    }

    // Validate voice
    if !pipeline.has_voice(&cli.voice) {
        bail!(
            "Voice '{}' not found. Available: {}",
            cli.voice,
            pipeline.list_voices().join(", ")
        );
    }

    // Get token IDs — either pre-tokenized or via Tekken BPE encoding
    let token_ids: Vec<u32> = if let Some(ids) = &cli.token_ids {
        info!(n_tokens = ids.len(), "Using pre-tokenized IDs");
        ids.clone()
    } else {
        let tokenizer_path = match &cli.tokenizer {
            Some(p) => PathBuf::from(p),
            None => model_dir.join("tekken.json"),
        };
        if !tokenizer_path.exists() {
            bail!("Tokenizer not found at {}", tokenizer_path.display());
        }

        info!("Loading Tekken tokenizer from {}", tokenizer_path.display());
        let encoder =
            TekkenEncoder::from_file(&tokenizer_path).context("Failed to load Tekken tokenizer")?;

        info!(text = %text, voice = %cli.voice, "Synthesizing speech");
        encoder.encode(text)
    };
    info!(n_tokens = token_ids.len(), "Text tokenized");

    // Generate audio
    let gen_start = Instant::now();
    let audio =
        pipeline.generate_with_max_frames(&token_ids, &cli.voice, cli.max_frames, &device)?;
    info!(
        elapsed_ms = gen_start.elapsed().as_millis() as u64,
        samples = audio.len(),
        duration_sec = format!("{:.2}", audio.len() as f64 / audio.sample_rate as f64),
        "Audio generated"
    );

    // Save output
    let output_path = PathBuf::from(&cli.output);
    audio
        .save(&output_path)
        .with_context(|| format!("Failed to save audio to {}", output_path.display()))?;
    info!(path = %output_path.display(), "Audio saved");

    println!(
        "Saved {:.2}s of audio to {}",
        audio.len() as f64 / audio.sample_rate as f64,
        output_path.display()
    );

    Ok(())
}
