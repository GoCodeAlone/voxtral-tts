//! Integration test: Q4 GGUF TTS end-to-end inference.
//!
//! Loads the quantized TTS model, runs a short synthesis with a voice preset,
//! and verifies the output is non-silent audio.

use burn::backend::Wgpu;
use burn::tensor::Tensor;
use std::path::Path;

type Backend = Wgpu;

/// Skip helper: returns true if model files are missing.
fn models_available() -> bool {
    Path::new("models/voxtral-tts-q4.gguf").exists()
        && Path::new("models/voxtral-tts/voice_embedding/casual_female.safetensors").exists()
        && Path::new("models/voxtral-tts/tekken.json").exists()
}

#[test]
fn test_q4_tts_load_and_generate() {
    if !models_available() {
        println!("Skipping: TTS Q4 GGUF or voice/tokenizer files not found");
        return;
    }

    let device = burn::backend::wgpu::WgpuDevice::default();

    // Phase 1: Load Q4 GGUF
    println!("Loading Q4 TTS GGUF...");
    let start = std::time::Instant::now();
    let mut loader = voxtral_mini_realtime::gguf::Q4TtsModelLoader::from_file(Path::new(
        "models/voxtral-tts-q4.gguf",
    ))
    .expect("Failed to open GGUF");

    let (backbone, fm, codec) = loader.load(&device).expect("Failed to load Q4 TTS model");
    println!("Loaded in {:.1}s", start.elapsed().as_secs_f32());

    assert_eq!(backbone.n_layers(), 26);
    assert_eq!(backbone.d_model(), 3072);
    assert_eq!(fm.config().n_layers, 3);

    // Phase 2: Load voice embedding
    let voice_bytes =
        std::fs::read("models/voxtral-tts/voice_embedding/casual_female.safetensors").unwrap();
    let voice_embed: Tensor<Backend, 2> =
        voxtral_mini_realtime::tts::voice::load_voice_from_bytes(&voice_bytes, 3072, &device)
            .expect("Failed to load voice");

    let [voice_frames, dim] = voice_embed.dims();
    println!("Voice: casual_female, {voice_frames} frames, dim={dim}");
    assert_eq!(dim, 3072);
    assert!(voice_frames > 0);

    // Phase 3: Tokenize text
    let tokenizer_json = std::fs::read_to_string("models/voxtral-tts/tekken.json").unwrap();
    let tokenizer = voxtral_mini_realtime::tokenizer::TekkenEncoder::from_json(&tokenizer_json)
        .expect("Failed to load tokenizer");

    let text = "Mary had a little lamb";
    let token_ids = tokenizer.encode(text);
    println!("Text: \"{text}\" → {} tokens", token_ids.len());
    assert!(!token_ids.is_empty());

    // Phase 4: Build input sequence using Q4 backbone's embed_tokens_from_ids
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
    println!("Input sequence: {seq_len} tokens");

    // Phase 5: Run async generate (short: 50 frames max for test speed)
    let codebook_layout = voxtral_mini_realtime::tts::config::AudioCodebookLayout::default();
    let codebook = voxtral_mini_realtime::tts::embeddings::AudioCodebookEmbeddings::new(
        backbone.audio_codebook_embeddings().clone(),
        codebook_layout,
    );

    println!("Running Q4 generate_async (max 200 frames)...");
    let gen_start = std::time::Instant::now();
    let frames = pollster::block_on(backbone.generate_async(input_sequence, &fm, &codebook, 200))
        .expect("generate_async failed");

    println!(
        "Generated {} frames in {:.1}s",
        frames.len(),
        gen_start.elapsed().as_secs_f32()
    );
    assert!(!frames.is_empty(), "Should generate at least one frame");

    // Verify frame contents are plausible
    for (i, frame) in frames.iter().enumerate() {
        assert!(
            frame.semantic_idx < 8192,
            "Frame {i}: semantic_idx {} out of range",
            frame.semantic_idx
        );
        for (j, &level) in frame.acoustic_levels.iter().enumerate() {
            assert!(
                level <= 20,
                "Frame {i}, codebook {j}: acoustic level {level} > 20"
            );
        }
    }

    // Phase 6: Codec decode → waveform
    println!("Decoding through codec...");
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
    println!(
        "Waveform: {total_samples} samples ({:.2}s at 24kHz)",
        total_samples as f64 / 24000.0
    );

    assert!(total_samples > 0, "Waveform should have samples");

    // Verify audio is non-silent
    let wav_data = waveform.to_data();
    let samples = wav_data.as_slice::<f32>().unwrap();
    let peak = samples.iter().map(|s| s.abs()).fold(0.0f32, f32::max);
    let rms: f32 = (samples.iter().map(|s| s * s).sum::<f32>() / samples.len() as f32).sqrt();

    println!("Audio stats: peak={peak:.4}, rms={rms:.6}");
    assert!(
        peak > 0.001,
        "Audio peak {peak} is too low — likely silent output"
    );
    assert!(
        rms > 0.0001,
        "Audio RMS {rms} is too low — likely near-silent output"
    );

    // Save WAV for manual inspection / Whisper verification
    let mut norm_samples = samples.to_vec();
    if peak > 1e-6 {
        let gain = 0.95 / peak;
        for s in &mut norm_samples {
            *s *= gain;
        }
    }
    let audio_buf = voxtral_mini_realtime::audio::AudioBuffer::new(norm_samples, 24000);
    audio_buf
        .save(std::path::Path::new("/tmp/q4_tts_mary.wav"))
        .expect("Failed to save WAV");
    println!("Saved /tmp/q4_tts_mary.wav");

    println!("Q4 TTS E2E test PASSED");
}
