//! Q4_0 quantized model structs for TTS.
//!
//! Mirrors the f32 TTS model in `src/tts/` but uses [`Q4Linear`] for all
//! weight-heavy layers (attention projections, FFN, FM projections).
//! Non-linear ops (RMSNorm, RoPE, softmax, SiLU, convolutions) stay as
//! regular Burn f32 tensors/ops.

use burn::backend::wgpu::WgpuDevice;
use burn::backend::Wgpu;
use burn::nn::Linear;
use burn::tensor::{Int, Tensor, TensorData};

use crate::models::layers::{KVCache, LayerCaches, RmsNorm, RoPE};
use crate::models::time_embedding::TimeEmbedding;
use crate::tts::config::{FmTransformerConfig, TtsBackboneConfig};

use super::linear::Q4Linear;
use super::model::TokEmbedStore;

// ---------------------------------------------------------------------------
// Q4TtsDecoderLayer
// ---------------------------------------------------------------------------

/// TTS backbone decoder layer with Q4-quantized weights.
///
/// Plain pre-norm transformer layer (NO ADA RMSNorm — unlike ASR Q4DecoderLayer).
/// Uses CAUSAL attention.
pub struct Q4TtsDecoderLayer {
    attention_norm: RmsNorm<Wgpu>,
    attention: super::model::Q4Attention,
    ffn_norm: RmsNorm<Wgpu>,
    ffn: super::model::Q4FeedForward,
}

impl Q4TtsDecoderLayer {
    /// Create a new Q4 TTS decoder layer.
    pub fn new(
        attention_norm: RmsNorm<Wgpu>,
        attention: super::model::Q4Attention,
        ffn_norm: RmsNorm<Wgpu>,
        ffn: super::model::Q4FeedForward,
    ) -> Self {
        Self {
            attention_norm,
            attention,
            ffn_norm,
            ffn,
        }
    }

    /// Forward pass with causal attention.
    pub fn forward(&self, x: Tensor<Wgpu, 3>, rope: &RoPE<Wgpu>, offset: usize) -> Tensor<Wgpu, 3> {
        let residual = x.clone();
        let x = self.attention_norm.forward(x);
        let x = self.attention.forward(x, rope, offset, true);
        let x = x + residual;

        let residual = x.clone();
        let x = self.ffn_norm.forward(x);
        let x = self.ffn.forward(x);
        x + residual
    }

    /// Forward pass with KV cache.
    pub fn forward_with_cache(
        &self,
        x: Tensor<Wgpu, 3>,
        rope: &RoPE<Wgpu>,
        cache: &mut KVCache<Wgpu>,
    ) -> Tensor<Wgpu, 3> {
        let residual = x.clone();
        let x = self.attention_norm.forward(x);
        let x = self.attention.forward_with_cache(x, rope, cache, true);
        let x = x + residual;

        let residual = x.clone();
        let x = self.ffn_norm.forward(x);
        let x = self.ffn.forward(x);
        x + residual
    }
}

// ---------------------------------------------------------------------------
// Q4TtsBackbone
// ---------------------------------------------------------------------------

/// TTS decoder backbone with Q4-quantized transformer layers.
///
/// Ministral 3B architecture with plain pre-norm (no ADA RMSNorm).
/// Holds 26 transformer layers, token embeddings (tied with LM head),
/// audio codebook embeddings, and final RMSNorm.
pub struct Q4TtsBackbone {
    layers: Vec<Q4TtsDecoderLayer>,
    norm: RmsNorm<Wgpu>,
    tok_embeddings: TokEmbedStore,
    audio_codebook_embeddings: Tensor<Wgpu, 2>,
    rope: RoPE<Wgpu>,
    config: TtsBackboneConfig,
    d_model: usize,
    device: WgpuDevice,
}

impl Q4TtsBackbone {
    /// Create a new Q4 TTS backbone.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        layers: Vec<Q4TtsDecoderLayer>,
        norm: RmsNorm<Wgpu>,
        tok_embeddings: TokEmbedStore,
        audio_codebook_embeddings: Tensor<Wgpu, 2>,
        rope: RoPE<Wgpu>,
        config: TtsBackboneConfig,
        device: WgpuDevice,
    ) -> Self {
        let d_model = config.dim;
        Self {
            layers,
            norm,
            tok_embeddings,
            audio_codebook_embeddings,
            rope,
            config,
            d_model,
            device,
        }
    }

    /// Embed token IDs from a CPU slice — avoids GPU readback (safe on WASM).
    pub fn embed_tokens_from_ids(&self, ids: &[i32], batch: usize, seq: usize) -> Tensor<Wgpu, 3> {
        match &self.tok_embeddings {
            TokEmbedStore::F32(embed) => {
                let id_tensor = Tensor::<Wgpu, 2, Int>::from_data(
                    TensorData::new(ids.to_vec(), [batch, seq]),
                    &self.device,
                );
                let flat_ids = id_tensor.reshape([batch * seq]);
                let selected = embed.clone().select(0, flat_ids);
                selected.reshape([batch, seq, self.d_model])
            }
            TokEmbedStore::Q4 { cpu_bytes, .. } => {
                self.embed_from_q4_bytes(cpu_bytes, ids, batch, seq)
            }
        }
    }

    /// Dequantize specific rows from CPU Q4 bytes.
    fn embed_from_q4_bytes(
        &self,
        cpu_bytes: &[u8],
        ids: &[i32],
        batch: usize,
        seq: usize,
    ) -> Tensor<Wgpu, 3> {
        let blocks_per_row = self.d_model / 32;
        let bytes_per_row = blocks_per_row * 18;
        let mut output = vec![0.0f32; ids.len() * self.d_model];

        for (i, &id) in ids.iter().enumerate() {
            let row_offset = (id as usize) * bytes_per_row;
            let row_bytes = &cpu_bytes[row_offset..row_offset + bytes_per_row];
            let out_slice = &mut output[i * self.d_model..(i + 1) * self.d_model];

            for block in 0..blocks_per_row {
                let bo = block * 18;
                let d =
                    half::f16::from_bits(u16::from_le_bytes([row_bytes[bo], row_bytes[bo + 1]]))
                        .to_f32();
                let base = block * 32;
                for j in 0..16 {
                    let byte = row_bytes[bo + 2 + j];
                    out_slice[base + j] = ((byte & 0x0F) as f32 - 8.0) * d;
                    out_slice[base + j + 16] = (((byte >> 4) & 0x0F) as f32 - 8.0) * d;
                }
            }
        }

        Tensor::from_data(
            TensorData::new(output, [batch, seq, self.d_model]),
            &self.device,
        )
    }

    /// Compute logits from hidden states (LM head with tied embeddings).
    pub fn lm_head(&self, hidden_states: Tensor<Wgpu, 3>) -> Tensor<Wgpu, 3> {
        match &self.tok_embeddings {
            TokEmbedStore::F32(embed) => {
                let [batch, seq, _] = hidden_states.dims();
                let vocab_size = embed.dims()[0];
                let embed_t = embed.clone().transpose().unsqueeze::<3>();
                let logits = hidden_states.matmul(embed_t);
                logits.reshape([batch, seq, vocab_size])
            }
            TokEmbedStore::Q4 { lm_head, .. } => lm_head.forward(hidden_states),
        }
    }

    /// Forward pass with KV cache.
    pub fn forward_with_cache(
        &self,
        x: Tensor<Wgpu, 3>,
        caches: &mut LayerCaches<Wgpu>,
    ) -> Tensor<Wgpu, 3> {
        let mut x = x;
        for (i, layer) in self.layers.iter().enumerate() {
            if let Some(cache) = caches.get_mut(i) {
                x = layer.forward_with_cache(x, &self.rope, cache);
            }
        }
        self.norm.forward(x)
    }

    /// Create a new set of KV caches for autoregressive decoding.
    pub fn create_cache(&self) -> LayerCaches<Wgpu> {
        LayerCaches::new(self.layers.len())
    }

    /// Number of layers.
    pub fn n_layers(&self) -> usize {
        self.layers.len()
    }

    /// Model dimension.
    pub fn d_model(&self) -> usize {
        self.d_model
    }

    /// Access audio codebook embeddings.
    pub fn audio_codebook_embeddings(&self) -> &Tensor<Wgpu, 2> {
        &self.audio_codebook_embeddings
    }

    /// Access config.
    pub fn config(&self) -> &TtsBackboneConfig {
        &self.config
    }

    /// Access device.
    pub fn device(&self) -> &WgpuDevice {
        &self.device
    }

    /// Run autoregressive generation to produce audio frames (async, WASM-safe).
    ///
    /// Uses `into_data_async().await` for all GPU readbacks instead of
    /// `into_scalar().elem()` which panics in WASM.
    ///
    /// # Arguments
    /// * `input_sequence` - Pre-built input embeddings [1, seq_len, dim]
    /// * `fm` - Q4 flow-matching transformer
    /// * `codebook` - Audio codebook embeddings for frame embedding lookup
    /// * `max_frames` - Maximum number of audio frames to generate
    ///
    /// # Returns
    /// Vector of generated frames (semantic index + 36 acoustic levels per frame).
    pub async fn generate_async(
        &self,
        input_sequence: Tensor<Wgpu, 3>,
        fm: &Q4FmTransformer,
        codebook: &crate::tts::embeddings::AudioCodebookEmbeddings<Wgpu>,
        max_frames: usize,
    ) -> Result<Vec<crate::tts::backbone::GeneratedFrame>, String> {
        use crate::tts::backbone::GeneratedFrame;
        use crate::tts::codec::quantizer::Fsq;

        let acoustic_dim = fm.config().acoustic_dim;
        let mut caches = self.create_cache();
        let mut frames = Vec::with_capacity(max_frames);

        // Phase 1: Prefill — process the entire input sequence, populate KV caches.
        let prefill_out = self.forward_with_cache(input_sequence, &mut caches);

        // Extract the last hidden state from prefill as the first decode input
        let [batch, seq_len, dim] = prefill_out.dims();
        let mut h = prefill_out.slice([0..batch, (seq_len - 1)..seq_len, 0..dim]); // [1, 1, dim]

        // Phase 2: Decode loop — one frame per iteration.
        for frame_idx in 0..max_frames {
            // Semantic token: projection → logits → argmax
            let semantic_logits = fm.semantic_logits(h.clone()); // [1, semantic_output_size]
            let semantic_idx = semantic_logits.argmax(1); // [1, 1]

            // Async GPU readback (WASM-safe)
            let semantic_idx_val: i32 = Tensor::<Wgpu, 2, Int>::into_data_async(semantic_idx)
                .await
                .map_err(|e| format!("Failed to read semantic index: {e}"))?
                .to_vec::<i32>()
                .map_err(|e| format!("Failed to extract semantic data: {e}"))?[0];
            let semantic_idx_val = semantic_idx_val as usize;

            tracing::debug!(
                frame = frame_idx,
                semantic_idx = semantic_idx_val,
                "Generated semantic token"
            );

            // Check for END_AUDIO (index 1)
            if semantic_idx_val == 1 {
                tracing::info!(frame = frame_idx, "END_AUDIO token detected, stopping");
                break;
            }

            // Acoustic tokens: Euler ODE from noise → FSQ quantize
            let noise: Tensor<Wgpu, 2> = Tensor::random(
                [1, acoustic_dim],
                burn::tensor::Distribution::Normal(0.0, 1.0),
                &self.device,
            );
            let acoustic_raw = fm.euler_ode_solve(h.clone(), noise); // [1, 36]

            // FSQ quantize: continuous R^36 → discrete indices [0, 20]
            let acoustic_indices = Fsq::quantize(acoustic_raw); // [1, 36]

            // Async GPU readback for acoustic data
            let acoustic_data = Tensor::<Wgpu, 2>::into_data_async(acoustic_indices)
                .await
                .map_err(|e| format!("Failed to read acoustic indices: {e}"))?;
            let acoustic_slice = acoustic_data
                .as_slice::<f32>()
                .map_err(|e| format!("Failed to extract acoustic data: {e}"))?;

            let mut acoustic_levels = [0usize; 36];
            for (i, &v) in acoustic_slice.iter().enumerate().take(36) {
                acoustic_levels[i] = v as usize;
            }

            // Map semantic index: subtract 2 to get raw VQ index (0..8191)
            let raw_semantic = semantic_idx_val.saturating_sub(2);

            frames.push(GeneratedFrame {
                semantic_idx: raw_semantic,
                acoustic_levels,
            });

            // Embed frame → next input for backbone
            let frame_embed = codebook.embed_frame(raw_semantic, &acoustic_levels); // [1, dim]
            let next_input = frame_embed.unsqueeze_dim::<3>(0); // [1, 1, dim]

            // Forward through backbone with cache → next hidden state
            h = self.forward_with_cache(next_input, &mut caches);
        }

        Ok(frames)
    }
}

// ---------------------------------------------------------------------------
// Q4FmLayer
// ---------------------------------------------------------------------------

/// Bidirectional (NON-CAUSAL) transformer layer for the FM transformer.
///
/// Same structure as Q4EncoderLayer but uses non-causal attention.
pub struct Q4FmLayer {
    attention_norm: RmsNorm<Wgpu>,
    attention: super::model::Q4Attention,
    ffn_norm: RmsNorm<Wgpu>,
    ffn: super::model::Q4FeedForward,
}

impl Q4FmLayer {
    /// Create a new Q4 FM layer.
    pub fn new(
        attention_norm: RmsNorm<Wgpu>,
        attention: super::model::Q4Attention,
        ffn_norm: RmsNorm<Wgpu>,
        ffn: super::model::Q4FeedForward,
    ) -> Self {
        Self {
            attention_norm,
            attention,
            ffn_norm,
            ffn,
        }
    }

    /// Forward pass with non-causal (bidirectional) attention.
    pub fn forward(&self, x: Tensor<Wgpu, 3>, rope: &RoPE<Wgpu>, offset: usize) -> Tensor<Wgpu, 3> {
        let residual = x.clone();
        let x = self.attention_norm.forward(x);
        let x = self.attention.forward(x, rope, offset, false);
        let x = x + residual;

        let residual = x.clone();
        let x = self.ffn_norm.forward(x);
        let x = self.ffn.forward(x);
        x + residual
    }
}

// ---------------------------------------------------------------------------
// Q4FmTransformer
// ---------------------------------------------------------------------------

/// Flow-matching transformer with Q4-quantized weights.
///
/// Predicts acoustic velocity vectors from backbone hidden states using
/// Euler ODE steps with classifier-free guidance.
pub struct Q4FmTransformer {
    time_embedding: TimeEmbedding,
    rope: RoPE<Wgpu>,
    layers: Vec<Q4FmLayer>,
    llm_projection: Q4Linear,
    time_projection: Q4Linear,
    input_projection: Linear<Wgpu>,
    semantic_codebook_output: Q4Linear,
    acoustic_codebook_output: Linear<Wgpu>,
    norm: RmsNorm<Wgpu>,
    config: FmTransformerConfig,
}

impl Q4FmTransformer {
    /// Create a new Q4 FM transformer.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        time_embedding: TimeEmbedding,
        rope: RoPE<Wgpu>,
        layers: Vec<Q4FmLayer>,
        llm_projection: Q4Linear,
        time_projection: Q4Linear,
        input_projection: Linear<Wgpu>,
        semantic_codebook_output: Q4Linear,
        acoustic_codebook_output: Linear<Wgpu>,
        norm: RmsNorm<Wgpu>,
        config: FmTransformerConfig,
    ) -> Self {
        Self {
            time_embedding,
            rope,
            layers,
            llm_projection,
            time_projection,
            input_projection,
            semantic_codebook_output,
            acoustic_codebook_output,
            norm,
            config,
        }
    }

    /// Compute semantic logits via direct projection of backbone hidden state.
    ///
    /// Returns logits of shape [batch, semantic_output_size] with masking:
    /// - Index 0 (EMPTY_AUDIO) masked to -inf
    /// - Indices >= (2 + 8192) masked to -inf (beyond valid semantic range)
    pub fn semantic_logits(&self, h: Tensor<Wgpu, 3>) -> Tensor<Wgpu, 2> {
        let [batch, _seq, _dim] = h.dims();
        let device = h.device();

        let logits = self.semantic_codebook_output.forward(h);
        let mut logits = logits.reshape([batch, self.config.semantic_output_size]);

        let mut mask_data = vec![0.0f32; self.config.semantic_output_size];
        mask_data[0] = f32::NEG_INFINITY;
        let valid_end = 2 + 8192;
        for v in mask_data.iter_mut().skip(valid_end) {
            *v = f32::NEG_INFINITY;
        }
        let mask: Tensor<Wgpu, 1> = Tensor::from_floats(mask_data.as_slice(), &device);
        logits = logits + mask.unsqueeze_dim::<2>(0);

        logits
    }

    /// Predict acoustic velocity from backbone hidden state via FM layers.
    ///
    /// Builds a 3-token sequence:
    /// - pos 0: `input_projection(x_t)`
    /// - pos 1: `time_projection(time_embed(t))`
    /// - pos 2: `llm_projection(h)`
    ///
    /// Runs through bidirectional layers, then extracts position 0's hidden
    /// and projects via `acoustic_codebook_output` → velocity in R^36.
    pub fn predict_velocity(
        &self,
        h: Tensor<Wgpu, 3>,
        x_t: Tensor<Wgpu, 3>,
        t: f32,
    ) -> Tensor<Wgpu, 2> {
        let device = h.device();

        // Project inputs to FM dim
        let x_proj = self.input_projection.forward(x_t); // [1, 1, 3072]
        let t_embed = self.time_embedding.embed(t, &device); // [1, 1, 3072]
        let t_proj = self.time_projection.forward(t_embed); // [1, 1, 3072]
        let h_proj = self.llm_projection.forward(h); // [1, 1, 3072]

        // Build 3-token sequence: [x_t, t, h]
        let seq = Tensor::cat(vec![x_proj, t_proj, h_proj], 1);

        // Run through bidirectional layers
        let mut hidden = seq;
        for layer in &self.layers {
            hidden = layer.forward(hidden, &self.rope, 0);
        }

        // Apply final norm
        hidden = self.norm.forward(hidden);

        // Extract position 0 (acoustic state position) and project to velocity
        let [batch, _seq, dim] = hidden.dims();
        let h_acoustic = hidden.slice([0..batch, 0..1, 0..dim]);
        let velocity = self.acoustic_codebook_output.forward(h_acoustic);
        velocity.reshape([batch, self.config.acoustic_dim])
    }

    /// Run the Euler ODE solver with classifier-free guidance.
    ///
    /// Starting from Gaussian noise in R^36, performs Euler steps from
    /// t=0 (noise) to t=1 (signal) with CFG (alpha=1.2).
    pub fn euler_ode_solve(&self, h: Tensor<Wgpu, 3>, noise: Tensor<Wgpu, 2>) -> Tensor<Wgpu, 2> {
        let device = h.device();
        let n_points = self.config.euler_steps; // 8 points → 7 intervals
        let alpha = self.config.cfg_alpha;

        // Zero hidden state for unconditional pass
        let [batch, seq, dim] = h.dims();
        let h_uncond: Tensor<Wgpu, 3> = Tensor::zeros([batch, seq, dim], &device);

        let mut x_t = noise;

        // linspace(0, 1, n_points): iterate over N-1 intervals
        for step in 0..(n_points - 1) {
            let t = step as f32 / (n_points - 1) as f32;
            let dt = 1.0 / (n_points - 1) as f32;

            let [b, acoustic_dim] = x_t.dims();
            let x_t_3d = x_t.clone().reshape([b, 1, acoustic_dim]);

            // Conditional velocity
            let v_cond = self.predict_velocity(h.clone(), x_t_3d.clone(), t);

            // Unconditional velocity
            let v_uncond = self.predict_velocity(h_uncond.clone(), x_t_3d, t);

            // CFG: v = alpha * v_cond + (1 - alpha) * v_uncond
            let v = v_cond * alpha + v_uncond * (1.0 - alpha);

            // Euler step: x_{t+dt} = x_t + v * dt
            x_t = x_t + v * dt;
        }

        x_t
    }

    /// Access the config.
    pub fn config(&self) -> &FmTransformerConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::module::{Param, ParamId};

    use crate::models::layers::RoPEConfig;

    /// Helper: build a small Q4TtsDecoderLayer for testing using f32 Q4Linear stand-ins.
    fn make_test_q4_attention(
        dim: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        device: &WgpuDevice,
    ) -> super::super::model::Q4Attention {
        use super::super::model::Q4Attention;
        let wq = make_test_q4_linear(dim, dim, device);
        let wk = make_test_q4_linear(n_kv_heads * head_dim, dim, device);
        let wv = make_test_q4_linear(n_kv_heads * head_dim, dim, device);
        let wo = make_test_q4_linear(dim, dim, device);
        Q4Attention::new(wq, wk, wv, wo, n_heads, n_kv_heads, head_dim, None)
    }

    fn make_test_q4_ffn(
        dim: usize,
        ffn_dim: usize,
        device: &WgpuDevice,
    ) -> super::super::model::Q4FeedForward {
        use super::super::model::Q4FeedForward;
        let w1 = make_test_q4_linear(ffn_dim, dim, device);
        let w2 = make_test_q4_linear(dim, ffn_dim, device);
        let w3 = make_test_q4_linear(ffn_dim, dim, device);
        Q4FeedForward::new(w1, w2, w3)
    }

    /// Create a Q4Linear from random f32 data (for testing only).
    fn make_test_q4_linear(
        out_features: usize,
        in_features: usize,
        device: &WgpuDevice,
    ) -> Q4Linear {
        // Round up to nearest multiple of 32 for Q4 block alignment
        let out_aligned = ((out_features + 31) / 32) * 32;
        let in_aligned = ((in_features + 31) / 32) * 32;
        let num_elements = out_aligned * in_aligned;
        let num_blocks = num_elements / 32;
        let bytes = vec![0u8; num_blocks * 18];
        let q4 = super::super::tensor::Q4Tensor::from_q4_bytes(
            &bytes,
            [out_aligned, in_aligned],
            device,
        )
        .expect("Q4Tensor creation");
        Q4Linear::new(q4, None)
    }

    fn make_test_rms_norm(dim: usize, device: &WgpuDevice) -> RmsNorm<Wgpu> {
        RmsNorm {
            weight: burn::nn::RmsNorm {
                gamma: Param::initialized(ParamId::new(), Tensor::<Wgpu, 1>::ones([dim], device)),
                epsilon: 1e-5,
            },
        }
    }

    #[test]
    fn test_q4_tts_decoder_layer_shape() {
        let device = WgpuDevice::default();
        let dim = 64;
        let n_heads = 4;
        let n_kv_heads = 2;
        let head_dim = dim / n_heads;
        let ffn_dim = 256;

        let attention_norm = make_test_rms_norm(dim, &device);
        let attention = make_test_q4_attention(dim, n_heads, n_kv_heads, head_dim, &device);
        let ffn_norm = make_test_rms_norm(dim, &device);
        let ffn = make_test_q4_ffn(dim, ffn_dim, &device);

        let layer = Q4TtsDecoderLayer::new(attention_norm, attention, ffn_norm, ffn);

        let rope = RoPEConfig::new(head_dim, 1024)
            .with_theta(1_000_000.0)
            .init(&device);

        let x = Tensor::<Wgpu, 3>::zeros([1, 10, dim], &device);
        let out = layer.forward(x, &rope, 0);
        assert_eq!(out.dims(), [1, 10, dim]);
    }

    #[test]
    fn test_q4_fm_layer_shape() {
        let device = WgpuDevice::default();
        let dim = 64;
        let n_heads = 4;
        let n_kv_heads = 2;
        let head_dim = dim / n_heads;
        let ffn_dim = 256;

        let attention_norm = make_test_rms_norm(dim, &device);
        let attention = make_test_q4_attention(dim, n_heads, n_kv_heads, head_dim, &device);
        let ffn_norm = make_test_rms_norm(dim, &device);
        let ffn = make_test_q4_ffn(dim, ffn_dim, &device);

        let layer = Q4FmLayer::new(attention_norm, attention, ffn_norm, ffn);

        let rope = RoPEConfig::new(head_dim, 16)
            .with_theta(10_000.0)
            .init(&device);

        // FM input is always 3 tokens
        let x = Tensor::<Wgpu, 3>::zeros([1, 3, dim], &device);
        let out = layer.forward(x, &rope, 0);
        assert_eq!(out.dims(), [1, 3, dim]);
    }
}
