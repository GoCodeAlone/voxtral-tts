//! Q4 TTS benchmark — measures load time, generation time, and codec decode time.
//! Run with: cargo test --release --features "wgpu,cli,hub" --test bench_q4_tts -- --nocapture

use burn::backend::Wgpu;
use burn::tensor::Tensor;
use std::path::Path;
use std::time::Instant;

type Backend = Wgpu;

fn models_available() -> bool {
    Path::new("models/voxtral-tts-q4.gguf").exists()
        && Path::new("models/voxtral-tts/voice_embedding/casual_female.safetensors").exists()
        && Path::new("models/voxtral-tts/tekken.json").exists()
}

#[test]
fn bench_q4_tts_mary() {
    run_bench("Mary had a little lamb", 300, "mary", Some(3));
}

#[test]
fn bench_q4_tts_fox() {
    run_bench(
        "The quick brown fox jumps over the lazy dog",
        300,
        "fox_e4",
        Some(4),
    );
}

#[test]
fn bench_q4_tts_fox_e3() {
    run_bench(
        "The quick brown fox jumps over the lazy dog",
        300,
        "fox_e3",
        Some(3),
    );
}

/// Test reduced Euler steps for speed/quality tradeoff
#[test]
fn bench_q4_tts_euler_steps() {
    if !models_available() {
        println!("Skipping: model files not found");
        return;
    }

    // Run with different step counts to find the quality/speed sweet spot
    for steps in [8, 4, 3, 2] {
        run_bench(
            "Mary had a little lamb",
            300,
            &format!("mary_euler{steps}"),
            Some(steps),
        );
    }
}

fn run_bench(text: &str, max_frames: usize, tag: &str, euler_steps_override: Option<usize>) {
    if !models_available() {
        println!("Skipping {tag}: model files not found");
        return;
    }

    let device = burn::backend::wgpu::WgpuDevice::default();

    // Load model
    let t0 = Instant::now();
    let mut loader = voxtral_mini_realtime::gguf::Q4TtsModelLoader::from_file(Path::new(
        "models/voxtral-tts-q4.gguf",
    ))
    .unwrap();
    let (backbone, mut fm, codec) = loader.load(&device).unwrap();
    let load_ms = t0.elapsed().as_millis();

    // Override Euler steps if requested
    if let Some(steps) = euler_steps_override {
        fm.set_euler_steps(steps);
    }

    // Load voice
    let voice_bytes =
        std::fs::read("models/voxtral-tts/voice_embedding/casual_female.safetensors").unwrap();
    let voice_embed: Tensor<Backend, 2> =
        voxtral_mini_realtime::tts::voice::load_voice_from_bytes(&voice_bytes, 3072, &device)
            .unwrap();

    // Tokenize
    let tokenizer_json = std::fs::read_to_string("models/voxtral-tts/tekken.json").unwrap();
    let tokenizer =
        voxtral_mini_realtime::tokenizer::TekkenEncoder::from_json(&tokenizer_json).unwrap();
    let token_ids = tokenizer.encode(text);

    // Build input sequence
    let special = voxtral_mini_realtime::tts::config::TtsSpecialTokens::default();
    let bos = backbone.embed_tokens_from_ids(&[special.bos_token_id as i32], 1, 1);
    let begin_audio = backbone.embed_tokens_from_ids(&[special.begin_audio_token_id as i32], 1, 1);
    let next_audio_text =
        backbone.embed_tokens_from_ids(&[special.next_audio_text_token_id as i32], 1, 1);
    let repeat_audio_text =
        backbone.embed_tokens_from_ids(&[special.repeat_audio_text_token_id as i32], 1, 1);
    let text_ids_i32: Vec<i32> = token_ids.iter().map(|&id| id as i32).collect();
    let text_embeds = backbone.embed_tokens_from_ids(&text_ids_i32, 1, text_ids_i32.len());
    let voice_3d = voice_embed.unsqueeze_dim::<3>(0);

    let input_sequence = Tensor::cat(
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
    );
    let [_, seq_len, _] = input_sequence.dims();

    // Build codebook
    let codebook_layout = voxtral_mini_realtime::tts::config::AudioCodebookLayout::default();
    let codebook = voxtral_mini_realtime::tts::embeddings::AudioCodebookEmbeddings::new(
        backbone.audio_codebook_embeddings().clone(),
        codebook_layout,
    );

    // Generate
    let t1 = Instant::now();
    let frames =
        pollster::block_on(backbone.generate_async(input_sequence, &fm, &codebook, max_frames))
            .unwrap();
    let gen_ms = t1.elapsed().as_millis();

    // Codec decode
    let t2 = Instant::now();
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
    let codec_ms = t2.elapsed().as_millis();

    let audio_duration = total_samples as f64 / 24000.0;
    let rtf = gen_ms as f64 / 1000.0 / audio_duration;

    // Save WAV
    let wav_data = waveform.to_data();
    let samples = wav_data.as_slice::<f32>().unwrap();
    let mut norm_samples = samples[..total_samples].to_vec();
    let peak = norm_samples.iter().map(|s| s.abs()).fold(0.0f32, f32::max);
    if peak > 1e-6 {
        let gain = 0.95 / peak;
        for s in &mut norm_samples {
            *s *= gain;
        }
    }
    let audio_buf = voxtral_mini_realtime::audio::AudioBuffer::new(norm_samples, 24000);
    let wav_path = format!("/tmp/bench_q4_{tag}.wav");
    audio_buf.save(Path::new(&wav_path)).unwrap();

    let steps = euler_steps_override.unwrap_or(8);
    println!();
    println!("=== Q4 TTS Benchmark: \"{text}\" (euler_steps={steps}) ===");
    println!("  Model load:    {load_ms:>6} ms");
    println!("  Input seq:     {seq_len:>6} tokens");
    println!("  Generation:    {gen_ms:>6} ms  ({n_frames} frames)");
    println!("  Codec decode:  {codec_ms:>6} ms");
    println!("  Audio:         {audio_duration:>6.2} s  ({total_samples} samples @ 24kHz)");
    println!("  RTF:           {rtf:>6.2}x  (gen_time / audio_duration)");
    println!("  Saved:         {wav_path}");
}
