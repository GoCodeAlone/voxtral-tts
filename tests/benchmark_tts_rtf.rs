//! RTF (Real-Time Factor) benchmark for the Q4 TTS pipeline.
//!
//! Runs 10 phrases through the full TTS pipeline (backbone → FM → codec)
//! and reports RTF min/avg/max/p99 in a grep-friendly format.
//!
//! Requires model files at `$CORE_DUMP_MODEL_DIR`:
//!   $CORE_DUMP_MODEL_DIR/voxtral-tts-q4.gguf
//!   $CORE_DUMP_MODEL_DIR/tekken.json
//!   $CORE_DUMP_MODEL_DIR/voice_embedding/
//!
//! Run with:
//!   CORE_DUMP_MODEL_DIR=/path/to/models \
//!     cargo test --release --features "wgpu,native-tokenizer" -- benchmark_tts_rtf --nocapture

use burn::tensor::Tensor;
use std::path::PathBuf;
use std::time::Instant;

use voxtral_tts::backend::ActiveBackend;

const PHRASES: &[&str] = &[
    "Hello world.",
    "The quick brown fox jumps over the lazy dog.",
    "Mary had a little lamb, its fleece was white as snow.",
    "To be or not to be, that is the question.",
    "All that glitters is not gold.",
    "The only way to do great work is to love what you do.",
    "In the beginning was the word.",
    "It was the best of times, it was the worst of times.",
    "Ask not what your country can do for you.",
    "We shall overcome.",
];

fn model_dir() -> Option<PathBuf> {
    std::env::var("CORE_DUMP_MODEL_DIR").ok().map(PathBuf::from)
}

#[test]
fn benchmark_tts_rtf() {
    let model_dir = match model_dir() {
        Some(d) if d.exists() => d,
        Some(d) => {
            println!("SKIP: CORE_DUMP_MODEL_DIR={} not found", d.display());
            return;
        }
        None => {
            println!("SKIP: CORE_DUMP_MODEL_DIR not set");
            return;
        }
    };

    let gguf_path = model_dir.join("voxtral-tts-q4.gguf");
    let tekken_path = model_dir.join("tekken.json");
    let voice_dir = model_dir.join("voice_embedding");

    if !gguf_path.exists() || !tekken_path.exists() || !voice_dir.exists() {
        println!("SKIP: model files missing in {}", model_dir.display());
        return;
    }

    // Load model once, run all phrases
    let device = voxtral_tts::backend::default_device();

    let t_load = Instant::now();
    let mut loader = voxtral_tts::gguf::Q4TtsModelLoader::from_file(&gguf_path)
        .expect("open GGUF");
    let (mut backbone, mut fm, codec) = loader.load(&device).expect("load Q4 model");
    backbone.fuse_projections();
    fm.set_euler_steps(4);
    let load_ms = t_load.elapsed().as_millis();
    println!("Model load: {load_ms} ms");

    let tokenizer_json = std::fs::read_to_string(&tekken_path).expect("read tekken.json");
    let tokenizer = voxtral_tts::tokenizer::TekkenEncoder::from_json(&tokenizer_json)
        .expect("init tokenizer");

    let voice_bytes_path = voice_dir.join("casual_female.safetensors");
    let voice_bytes = std::fs::read(&voice_bytes_path).expect("read voice embedding");
    let voice_embed: Tensor<ActiveBackend, 2> =
        voxtral_tts::tts::voice::load_voice_from_bytes(&voice_bytes, 3072, &device)
            .expect("load voice");

    let codebook_layout = voxtral_tts::tts::config::AudioCodebookLayout::default();
    let codebook = voxtral_tts::tts::embeddings::AudioCodebookEmbeddings::new(
        backbone.audio_codebook_embeddings().clone(),
        codebook_layout,
    );
    let special = voxtral_tts::tts::config::TtsSpecialTokens::default();

    let mut rtf_values: Vec<f64> = Vec::with_capacity(PHRASES.len());

    for (i, phrase) in PHRASES.iter().enumerate() {
        let token_ids = tokenizer.encode(phrase);
        let text_ids_i32: Vec<i32> = token_ids.iter().map(|&id| id as i32).collect();

        let bos = backbone.embed_tokens_from_ids(&[special.bos_token_id as i32], 1, 1);
        let begin_audio =
            backbone.embed_tokens_from_ids(&[special.begin_audio_token_id as i32], 1, 1);
        let next_audio_text =
            backbone.embed_tokens_from_ids(&[special.next_audio_text_token_id as i32], 1, 1);
        let repeat_audio_text =
            backbone.embed_tokens_from_ids(&[special.repeat_audio_text_token_id as i32], 1, 1);
        let text_embeds = backbone.embed_tokens_from_ids(&text_ids_i32, 1, text_ids_i32.len());
        let voice_3d = voice_embed.clone().unsqueeze_dim::<3>(0);

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

        let t_gen = Instant::now();
        let frames =
            pollster::block_on(backbone.generate_async(input_sequence, &fm, &codebook, 2000))
                .expect("generate");
        let gen_ms = t_gen.elapsed().as_millis();

        let n_frames = frames.len();
        let semantic_indices: Vec<usize> = frames.iter().map(|f| f.semantic_idx).collect();
        let mut acoustic_data = Vec::with_capacity(n_frames * 36);
        for frame in &frames {
            for &level in &frame.acoustic_levels {
                acoustic_data.push(level as f32);
            }
        }
        let acoustic_tensor: Tensor<ActiveBackend, 2> = Tensor::from_data(
            burn::tensor::TensorData::new(acoustic_data, [n_frames, 36]),
            &device,
        );
        let waveform = codec.decode(&semantic_indices, acoustic_tensor);
        let [_batch, total_samples] = waveform.dims();
        let audio_duration = total_samples as f64 / 24000.0;
        let rtf = gen_ms as f64 / 1000.0 / audio_duration;
        rtf_values.push(rtf);

        println!(
            "Phrase {:>2}: {:>6} ms gen, {:>5.2}s audio, RTF={:.3}x — {:?}",
            i + 1,
            gen_ms,
            audio_duration,
            rtf,
            &phrase[..phrase.len().min(40)],
        );
    }

    // Compute statistics
    let n = rtf_values.len() as f64;
    let rtf_avg = rtf_values.iter().sum::<f64>() / n;
    let rtf_min = rtf_values.iter().cloned().fold(f64::MAX, f64::min);
    let rtf_max = rtf_values.iter().cloned().fold(f64::MIN, f64::max);
    let mut sorted = rtf_values.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let p99_idx = ((n * 0.99) as usize).min(sorted.len() - 1);
    let rtf_p99 = sorted[p99_idx];

    // Grep-friendly summary line for CI parsing
    println!(
        "Summary: min={:.3}x avg={:.3}x max={:.3}x p99={:.3}x (n={})",
        rtf_min, rtf_avg, rtf_max, rtf_p99, PHRASES.len()
    );
}
