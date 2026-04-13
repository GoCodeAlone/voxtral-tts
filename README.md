# voxtral-tts

**Streaming TTS lib crate** — optimized for real-time applications that need low-latency speech synthesis.

Forked from [TrevorS/voxtral-mini-realtime-rs](https://github.com/TrevorS/voxtral-mini-realtime-rs) — a pure-Rust implementation of Mistral's [Voxtral 4B TTS](https://huggingface.co/mistralai/Voxtral-4B-TTS-2603) using the [Burn](https://burn.dev) ML framework with wgpu GPU acceleration.

## Why This Fork Exists

The upstream repo is a full-featured CLI application with ASR, TTS, WASM browser support, HuggingFace Hub downloads, and sharded model loading. This fork strips it down to a focused **lib crate** optimized for real-time audio streaming:

1. **Streaming audio output** — the upstream collects all frames into a Vec, then writes a WAV file. This fork streams audio frame-by-frame through a ring buffer to cpal, so playback begins ~80ms after the first frame generates (on hardware with RTF < 1.0x).

2. **No CLI, no browser, no downloads** — stripped the binary targets, clap/indicatif, WASM/wasm-bindgen, HuggingFace Hub fetching, and sharded model loading. Models are managed by the consuming application.

3. **Ring buffer + cpal integration** — `AudioRingBuffer` with Mutex/Condvar backpressure feeds a persistent cpal output stream. The stream stays open across utterances (no per-call open/close overhead).

4. **Per-frame codec decode** — instead of batch-decoding all frames after generation, each frame is codec-decoded and pushed to the ring buffer immediately. This overlaps codec GPU work with backbone generation, resulting in ~1.3-1.5x faster total generation time.

5. **RTF/TTFA instrumentation** — every `speak()` call measures Real-Time Factor (generation_ms / audio_duration_ms) and Time To First Audio. Used for hardware tier calibration — auto-disables TTS if RTF > 5.0x.

6. **High-level `TtsEngine` API** — single `speak(text, voice_id, config)` call that handles tokenization, voice embedding, backbone decode, codec synthesis, and audio output internally.

## What Changed From Upstream

| Component | Upstream | This Fork |
|-----------|----------|-----------|
| Crate type | Binary + lib | **Lib only** |
| Audio output | Collect all → write WAV | **Frame-by-frame → ring buffer → cpal** |
| Dependencies | clap, indicatif, wasm-bindgen, hub | **None of these** |
| API | CLI flags | **`TtsEngine::speak()`** |
| Codec decode | Batch after generation | **Per-frame, overlapped with generation** |
| Timing | None | **RTF, TTFA, generation_ms** |

## What's Preserved

- All 3 pipeline stages: Q4 Ministral backbone → flow-matching transformer → codec decoder
- Q4 GGUF model loading (`Q4TtsModelLoader`)
- Tekken tokenizer (`TekkenEncoder`)
- Voice preset loading from SafeTensors
- Burn ML wgpu backend for GPU compute
- Custom WGSL compute shaders for Q4 inference
- 235 unit/integration tests (all passing)

## Quick Start

```rust
use voxtral_tts::{TtsEngine, SpeakConfig};
use std::path::Path;

// Load from model directory containing:
//   voxtral-tts-q4.gguf, tekken.json, voice_embedding/*.safetensors
let engine = TtsEngine::load(Path::new("/path/to/models/voice"))?;

// Speak — audio streams to default output device immediately
let result = engine.speak("Hello, I am VERA.", "casual_female", &SpeakConfig::default())?;

println!("{}ms audio in {}ms (RTF={:.2}x, TTFA={}ms)",
    result.duration_ms, result.generation_ms, result.rtf, result.ttfa_ms);
```

## Configuration

```rust
let config = SpeakConfig {
    euler_steps: 3,     // 3=fast, 4=balanced, 8=quality
    use_gpu: true,      // wgpu Metal/Vulkan/WebGPU
    max_frames: 2000,   // safety cap (~160s of audio)
};
```

## Benchmarks

NVIDIA DGX Spark (GB10, LPDDR5x), from upstream:

| Euler Steps | Gen Time | Audio | RTF | Model Size |
|-------------|----------|-------|-----|------------|
| 3 | 3.7s | 3.84s | **0.97x** | 2.67 GB |
| 4 | 5.0s | 4.96s | 1.01x | 2.67 GB |
| 8 | 20.6s | 2.96s | 6.97x | ~8 GB (BF16) |

RTF < 1.0x = faster than real-time. At 3 Euler steps on capable hardware, audio plays in real-time with streaming.

## Building

```bash
# Default: wgpu + native-tokenizer
cargo build --release

# Run tests (GPU required for full suite)
cargo test --release
```

## Model Files

Download from [TrevorJS/voxtral-tts-q4-gguf](https://huggingface.co/TrevorJS/voxtral-tts-q4-gguf):
- `voxtral-tts-q4.gguf` — Q4-quantized weights (2.67 GB)
- `tekken.json` — Tekken BPE tokenizer
- `voice_embedding/*.safetensors` — 20 voice presets across 9 languages

## License

Apache-2.0 (same as upstream)
