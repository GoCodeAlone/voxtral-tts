//! END_AUDIO investigation: why does speak() hit max_frames while generate_async stops early?
//!
//! Hypothesis: per-frame codec decode inside generate_with_callback's callback runs GPU compute
//! between backbone forward passes, possibly interfering with wgpu queue state or KV caches.
//!
//! Run with:
//!   VOXTRAL_MODEL_DIR=/Users/jon/.core-dump/models/voice \
//!   cargo test --release --test end_audio_investigation -- --nocapture --test-threads=1

use burn::backend::Wgpu;
use burn::tensor::{Tensor, TensorData};
use std::path::Path;
use std::time::Instant;

type Backend = Wgpu;

const MODEL_DIR: &str = "/Users/jon/.core-dump/models/voice";
/// Cap for investigation — we only need enough frames to know if END_AUDIO fires.
/// generate_async stops at ~55 frames for "Mary had a little lamb", so 300 is safe.
const MAX_FRAMES: usize = 300;

fn model_dir() -> String {
    std::env::var("VOXTRAL_MODEL_DIR").unwrap_or_else(|_| MODEL_DIR.to_string())
}

fn models_available() -> bool {
    let dir = model_dir();
    Path::new(&format!("{dir}/voxtral-tts-q4.gguf")).exists()
        && Path::new(&format!("{dir}/voice_embedding/casual_female.safetensors")).exists()
        && Path::new(&format!("{dir}/tekken.json")).exists()
}

fn load_model(
    dir: &str,
) -> (
    voxtral_tts::gguf::tts_model::Q4TtsBackbone,
    voxtral_tts::gguf::tts_model::Q4FmTransformer,
    voxtral_tts::tts::codec::CodecDecoder<Backend>,
    burn::backend::wgpu::WgpuDevice,
) {
    let device = burn::backend::wgpu::WgpuDevice::default();
    let t0 = Instant::now();
    let mut loader =
        voxtral_tts::gguf::Q4TtsModelLoader::from_file(Path::new(&format!(
            "{dir}/voxtral-tts-q4.gguf"
        )))
        .expect("open GGUF");
    let (backbone, fm, codec) = loader.load(&device).expect("load model");
    println!("  [model loaded in {}ms]", t0.elapsed().as_millis());
    (backbone, fm, codec, device)
}

fn build_input(
    backbone: &voxtral_tts::gguf::tts_model::Q4TtsBackbone,
    text: &str,
    dir: &str,
    device: &burn::backend::wgpu::WgpuDevice,
) -> Tensor<Backend, 3> {
    let voice_bytes =
        std::fs::read(format!("{dir}/voice_embedding/casual_female.safetensors")).unwrap();
    let voice_embed: Tensor<Backend, 2> =
        voxtral_tts::tts::voice::load_voice_from_bytes(&voice_bytes, 3072, device).unwrap();
    let tokenizer_json = std::fs::read_to_string(format!("{dir}/tekken.json")).unwrap();
    let tokenizer =
        voxtral_tts::tokenizer::TekkenEncoder::from_json(&tokenizer_json).unwrap();
    let token_ids = tokenizer.encode(text);
    let special = voxtral_tts::tts::config::TtsSpecialTokens::default();
    let bos = backbone.embed_tokens_from_ids(&[special.bos_token_id as i32], 1, 1);
    let begin_audio =
        backbone.embed_tokens_from_ids(&[special.begin_audio_token_id as i32], 1, 1);
    let next_audio_text =
        backbone.embed_tokens_from_ids(&[special.next_audio_text_token_id as i32], 1, 1);
    let repeat_audio_text =
        backbone.embed_tokens_from_ids(&[special.repeat_audio_text_token_id as i32], 1, 1);
    let text_ids_i32: Vec<i32> = token_ids.iter().map(|&id| id as i32).collect();
    let text_embeds = backbone.embed_tokens_from_ids(&text_ids_i32, 1, text_ids_i32.len());
    let voice_3d = voice_embed.unsqueeze_dim::<3>(0);
    Tensor::cat(
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
    )
}

fn make_codebook(
    backbone: &voxtral_tts::gguf::tts_model::Q4TtsBackbone,
) -> voxtral_tts::tts::embeddings::AudioCodebookEmbeddings<Backend> {
    let layout = voxtral_tts::tts::config::AudioCodebookLayout::default();
    voxtral_tts::tts::embeddings::AudioCodebookEmbeddings::new(
        backbone.audio_codebook_embeddings().clone(),
        layout,
    )
}

fn run_generate_async(
    backbone: &voxtral_tts::gguf::tts_model::Q4TtsBackbone,
    fm: &voxtral_tts::gguf::tts_model::Q4FmTransformer,
    input: Tensor<Backend, 3>,
    max_frames: usize,
) -> (usize, bool, u128) {
    let codebook = make_codebook(backbone);
    let t = Instant::now();
    let frames =
        pollster::block_on(backbone.generate_async(input, fm, &codebook, max_frames)).unwrap();
    let ms = t.elapsed().as_millis();
    let n = frames.len();
    (n, n >= max_frames, ms)
}

fn run_callback_noop(
    backbone: &voxtral_tts::gguf::tts_model::Q4TtsBackbone,
    fm: &voxtral_tts::gguf::tts_model::Q4FmTransformer,
    input: Tensor<Backend, 3>,
    max_frames: usize,
) -> (u32, bool, u128) {
    let codebook = make_codebook(backbone);
    let t = Instant::now();
    let n = pollster::block_on(
        backbone.generate_with_callback(input, fm, &codebook, max_frames, |_| true),
    )
    .unwrap();
    let ms = t.elapsed().as_millis();
    (n, n >= max_frames as u32, ms)
}

fn run_callback_with_codec_sync(
    backbone: &voxtral_tts::gguf::tts_model::Q4TtsBackbone,
    fm: &voxtral_tts::gguf::tts_model::Q4FmTransformer,
    codec: &voxtral_tts::tts::codec::CodecDecoder<Backend>,
    input: Tensor<Backend, 3>,
    max_frames: usize,
    device: &burn::backend::wgpu::WgpuDevice,
) -> (u32, bool, u128) {
    let codebook = make_codebook(backbone);
    let t = Instant::now();
    let n = pollster::block_on(backbone.generate_with_callback(
        input,
        fm,
        &codebook,
        max_frames,
        |frame| {
            let acoustic_data: Vec<f32> =
                frame.acoustic_levels.iter().map(|&v| v as f32).collect();
            let acoustic_tensor = Tensor::<Backend, 2>::from_data(
                TensorData::new(acoustic_data, [1, 36]),
                device,
            );
            let waveform = codec.decode(&[frame.semantic_idx], acoustic_tensor);
            // GPU sync readback (mirrors speak()'s ring buffer push)
            let _ = waveform.to_data().as_slice::<f32>().unwrap().to_vec();
            true
        },
    ))
    .unwrap();
    let ms = t.elapsed().as_millis();
    (n, n >= max_frames as u32, ms)
}

fn end_audio_str(hit_max: bool) -> &'static str {
    if hit_max { "⚠ NO_END_AUDIO (hit max)" } else { "✓ END_AUDIO detected" }
}

#[test]
fn end_audio_all_experiments() {
    if !models_available() {
        println!("SKIP: models not found at {}", model_dir());
        return;
    }
    let dir = model_dir();

    println!();
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  END_AUDIO Investigation — max_frames={MAX_FRAMES}                  ║");
    println!("╚══════════════════════════════════════════════════════════════╝");

    // ── Load model once ──────────────────────────────────────────────────────
    println!("\n[Loading model (shared across all tests)...]");
    let (backbone, mut fm_loaded, codec, device) = load_model(&dir);

    // ── TEST 1: euler_steps sweep with generate_async ─────────────────────
    println!();
    println!("━━ TEST 1: euler_steps sweep — generate_async ━━━━━━━━━━━━━━━━━━");
    println!("  Text: \"Mary had a little lamb\", max_frames={MAX_FRAMES}");
    for &steps in &[3usize, 4, 8] {
        fm_loaded.set_euler_steps(steps);
        let input = build_input(&backbone, "Mary had a little lamb", &dir, &device);
        let (n, hit, ms) = run_generate_async(&backbone, &fm_loaded, input, MAX_FRAMES);
        println!("  euler_steps={steps}: {n:>4} frames / {ms:>6}ms  {}", end_audio_str(hit));
    }

    // ── TEST 2: generate_async vs callback (noop) ──────────────────────────
    println!();
    println!("━━ TEST 2: generate_async vs generate_with_callback (noop) ━━━━━");
    println!("  Noop callback = no codec, no GPU work — isolates the mechanism");
    for &steps in &[3usize, 4, 8] {
        fm_loaded.set_euler_steps(steps);

        let input_a = build_input(&backbone, "Mary had a little lamb", &dir, &device);
        let (n_async, hit_async, ms_async) =
            run_generate_async(&backbone, &fm_loaded, input_a, MAX_FRAMES);

        let input_b = build_input(&backbone, "Mary had a little lamb", &dir, &device);
        let (n_cb, hit_cb, ms_cb) = run_callback_noop(&backbone, &fm_loaded, input_b, MAX_FRAMES);

        println!(
            "  steps={steps}: async={n_async}f/{ms_async}ms {} | noop_cb={n_cb}f/{ms_cb}ms {}",
            end_audio_str(hit_async),
            end_audio_str(hit_cb),
        );
    }

    // ── TEST 3: callback WITH per-frame codec decode + GPU sync ──────────────
    println!();
    println!("━━ TEST 3: generate_with_callback WITH codec decode + GPU sync ━━");
    println!("  Mirrors speak() exactly — does per-frame GPU work break END_AUDIO?");
    for &steps in &[3usize, 4, 8] {
        fm_loaded.set_euler_steps(steps);
        let input = build_input(&backbone, "Mary had a little lamb", &dir, &device);
        let (n, hit, ms) =
            run_callback_with_codec_sync(&backbone, &fm_loaded, &codec, input, MAX_FRAMES, &device);
        println!("  steps={steps}: {n:>4} frames / {ms:>6}ms  {}", end_audio_str(hit));
    }

    // ── TEST 4: speak() end-to-end ────────────────────────────────────────
    println!();
    println!("━━ TEST 4: TtsEngine::speak() end-to-end ━━━━━━━━━━━━━━━━━━━━━━");
    // Load engine separately (it opens cpal stream)
    match voxtral_tts::TtsEngine::load(Path::new(&dir)) {
        Err(e) => println!("  SKIP: TtsEngine::load failed: {e}"),
        Ok(engine) => {
            for &steps in &[3usize, 4, 8] {
                let cfg = voxtral_tts::SpeakConfig {
                    euler_steps: steps,
                    use_gpu: true,
                    max_frames: MAX_FRAMES,
                };
                match engine.speak("Mary had a little lamb", "casual_female", &cfg) {
                    Ok(r) => println!(
                        "  steps={steps}: {:>4} frames / {:>6}ms  RTF={:.2}x  ttfa={}ms  {}",
                        r.frames_generated, r.generation_ms, r.rtf, r.ttfa_ms,
                        end_audio_str(r.frames_generated >= MAX_FRAMES as u32),
                    ),
                    Err(e) => println!("  steps={steps}: ERROR: {e}"),
                }
            }
        }
    }

    // ── TEST 5: Performance comparison (euler_steps=3, 3 phrases) ────────
    println!();
    println!("━━ TEST 5: Performance comparison — euler_steps=3, 3 phrases ━━━");
    fm_loaded.set_euler_steps(3);
    let phrases = [
        "Mary had a little lamb",
        "Hello, I am VERA.",
        "The network scan found three vulnerable hosts on subnet ten.",
    ];

    println!("  --- generate_async ---");
    for phrase in &phrases {
        let input = build_input(&backbone, phrase, &dir, &device);
        let (n, hit, ms) = run_generate_async(&backbone, &fm_loaded, input, MAX_FRAMES);
        let audio_s = n as f64 / 12.5;
        let rtf = if audio_s > 0.0 { ms as f64 / 1000.0 / audio_s } else { 0.0 };
        println!(
            "  \"{phrase}\": {n}f / {ms}ms / {audio_s:.2}s / RTF={rtf:.2}x  {}",
            end_audio_str(hit)
        );
    }

    match voxtral_tts::TtsEngine::load(Path::new(&dir)) {
        Err(e) => println!("  speak() SKIP: {e}"),
        Ok(engine) => {
            let cfg = voxtral_tts::SpeakConfig {
                euler_steps: 3,
                use_gpu: true,
                max_frames: MAX_FRAMES,
            };
            println!("  --- speak() ---");
            for phrase in &phrases {
                match engine.speak(phrase, "casual_female", &cfg) {
                    Ok(r) => println!(
                        "  \"{phrase}\": {}f / {}ms / {:.2}s / RTF={:.2}x  {}",
                        r.frames_generated, r.generation_ms,
                        r.duration_ms as f64 / 1000.0, r.rtf,
                        end_audio_str(r.frames_generated >= MAX_FRAMES as u32),
                    ),
                    Err(e) => println!("  \"{phrase}\": ERROR: {e}"),
                }
            }
        }
    }

    println!();
    println!("Investigation complete.");
}
