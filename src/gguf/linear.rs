//! Q4_0 quantized linear layer.
//!
//! [`Q4Linear`] wraps a [`Q4Tensor`] weight matrix and optional f32 bias,
//! providing a `forward` method that delegates to [`q4_matmul`].
//!
//! On wgpu builds the WGSL fused shader path is used.  On non-wgpu builds
//! (cuda/metal/rocm) weights are dequantized at construction time and stored as
//! a full-precision `Tensor<ActiveBackend, 2>` so inference uses the native
//! cuBLAS / MPSGraph matmul without repeated CPU dequantization.

use burn::tensor::Tensor;

use crate::backend::ActiveBackend;

#[cfg(feature = "wgpu")]
use super::op::q4_matmul;
#[cfg(feature = "wgpu")]
use super::tensor::Q4Tensor;

#[cfg(not(feature = "wgpu"))]
use super::op::{q4_matmul, DequantPrecision};
#[cfg(not(feature = "wgpu"))]
use super::tensor::Q4Tensor;
#[cfg(not(feature = "wgpu"))]
use crate::backend::ActiveDevice;

// ---------------------------------------------------------------------------
// Q4Linear
// ---------------------------------------------------------------------------

/// A linear layer with Q4_0 quantized weights.
///
/// On wgpu: stores weights as `Q4Tensor` (GPU buffer); the WGSL shader
/// dequantizes on-the-fly.
/// On non-wgpu: pre-dequantizes at construction and stores a full-precision
/// `Tensor<ActiveBackend, 2>` for fast native matmul.
pub struct Q4Linear {
    #[cfg(feature = "wgpu")]
    weights: Q4Tensor,
    #[cfg(not(feature = "wgpu"))]
    weights: Tensor<ActiveBackend, 2>,
    bias: Option<Tensor<ActiveBackend, 1>>,
}

impl Q4Linear {
    /// Create a new Q4 linear layer (wgpu path — keeps weights on GPU).
    #[cfg(feature = "wgpu")]
    pub fn new(weights: Q4Tensor, bias: Option<Tensor<ActiveBackend, 1>>) -> Self {
        Self { weights, bias }
    }

    /// Create a new Q4 linear layer (non-wgpu path — pre-dequantizes weights).
    #[cfg(not(feature = "wgpu"))]
    pub fn new(
        weights: Q4Tensor,
        bias: Option<Tensor<ActiveBackend, 1>>,
        device: &ActiveDevice,
        precision: DequantPrecision,
    ) -> Self {
        let [n, k] = weights.shape();
        let w_f32: Vec<f32> = match precision {
            DequantPrecision::F32 => weights.dequantize_f32(),
            DequantPrecision::F16 => weights
                .dequantize_f16()
                .into_iter()
                .map(|v| v.to_f32())
                .collect(),
        };
        let weight_tensor = Tensor::<ActiveBackend, 2>::from_data(
            burn::tensor::TensorData::new(w_f32, [n, k]),
            device,
        );
        Self {
            weights: weight_tensor,
            bias,
        }
    }

    /// Access the underlying weight tensor (wgpu — Q4Tensor).
    #[cfg(feature = "wgpu")]
    pub fn weights(&self) -> &Q4Tensor {
        &self.weights
    }

    /// Forward pass: `x @ weights^T + bias`.
    ///
    /// `x` shape: `[B, M, K]` where `K = in_features`.
    /// Returns shape: `[B, M, N]` where `N = out_features`.
    #[cfg(feature = "wgpu")]
    pub fn forward(&self, x: Tensor<ActiveBackend, 3>) -> Result<Tensor<ActiveBackend, 3>, String> {
        let out = q4_matmul(x, &self.weights)?;
        Ok(match &self.bias {
            Some(bias) => out + bias.clone().unsqueeze::<3>(),
            None => out,
        })
    }

    /// Forward pass: `x @ weights^T + bias` (non-wgpu — weights pre-dequantized).
    #[cfg(not(feature = "wgpu"))]
    pub fn forward(&self, x: Tensor<ActiveBackend, 3>) -> Result<Tensor<ActiveBackend, 3>, String> {
        let dims = x.dims();
        let (b, m, k) = (dims[0], dims[1], dims[2]);
        let [n, _] = self.weights.dims().try_into().map_err(|_| "weight dims")?;
        let input_2d = x.reshape([b * m, k]);
        let output_2d = input_2d.matmul(self.weights.clone().transpose());
        let out = output_2d.reshape([b, m, n]);
        Ok(match &self.bias {
            Some(bias) => out + bias.clone().unsqueeze::<3>(),
            None => out,
        })
    }
}

// ---------------------------------------------------------------------------
// Q4FusedQKV
// ---------------------------------------------------------------------------

/// Fused Q/K/V projection: stores concatenated Q4 weights and splits output.
///
/// Instead of 3 separate Q4 matmul launches for wq, wk, wv, uses a single
/// concatenated weight matrix `[q_out + k_out + v_out, in_features]`.
/// Reduces kernel launches from 3 to 1 per layer.
pub struct Q4FusedQKV {
    #[cfg(feature = "wgpu")]
    weights: Q4Tensor,
    #[cfg(not(feature = "wgpu"))]
    weights: Tensor<ActiveBackend, 2>,
    q_out: usize,
    k_out: usize,
    v_out: usize,
}

impl Q4FusedQKV {
    /// Create from a pre-built concatenated Q4 tensor (wgpu path).
    #[cfg(feature = "wgpu")]
    pub fn new(weights: Q4Tensor, q_out: usize, k_out: usize, v_out: usize) -> Self {
        Self {
            weights,
            q_out,
            k_out,
            v_out,
        }
    }

    /// Create from a pre-built concatenated Q4 tensor (non-wgpu — pre-dequantizes).
    #[cfg(not(feature = "wgpu"))]
    pub fn new(
        weights: Q4Tensor,
        q_out: usize,
        k_out: usize,
        v_out: usize,
        device: &ActiveDevice,
        precision: DequantPrecision,
    ) -> Self {
        let [n, k] = weights.shape();
        let w_f32: Vec<f32> = match precision {
            DequantPrecision::F32 => weights.dequantize_f32(),
            DequantPrecision::F16 => weights
                .dequantize_f16()
                .into_iter()
                .map(|v| v.to_f32())
                .collect(),
        };
        let weight_tensor = Tensor::<ActiveBackend, 2>::from_data(
            burn::tensor::TensorData::new(w_f32, [n, k]),
            device,
        );
        Self {
            weights: weight_tensor,
            q_out,
            k_out,
            v_out,
        }
    }

    /// Forward: single Q4 matmul → split into (q, k, v).
    #[cfg(feature = "wgpu")]
    pub fn forward(
        &self,
        x: Tensor<ActiveBackend, 3>,
    ) -> Result<(Tensor<ActiveBackend, 3>, Tensor<ActiveBackend, 3>, Tensor<ActiveBackend, 3>), String>
    {
        let fused = q4_matmul(x, &self.weights)?;
        let q = fused.clone().narrow(2, 0, self.q_out);
        let k = fused.clone().narrow(2, self.q_out, self.k_out);
        let v = fused.narrow(2, self.q_out + self.k_out, self.v_out);
        Ok((q, k, v))
    }

    /// Forward: pre-dequantized matmul → split into (q, k, v).
    #[cfg(not(feature = "wgpu"))]
    pub fn forward(
        &self,
        x: Tensor<ActiveBackend, 3>,
    ) -> Result<(Tensor<ActiveBackend, 3>, Tensor<ActiveBackend, 3>, Tensor<ActiveBackend, 3>), String>
    {
        let dims = x.dims();
        let (b, m, k) = (dims[0], dims[1], dims[2]);
        let input_2d = x.reshape([b * m, k]);
        let total_out = self.q_out + self.k_out + self.v_out;
        let output_2d = input_2d.matmul(self.weights.clone().transpose());
        let fused = output_2d.reshape([b, m, total_out]);
        let q = fused.clone().narrow(2, 0, self.q_out);
        let k_t = fused.clone().narrow(2, self.q_out, self.k_out);
        let v = fused.narrow(2, self.q_out + self.k_out, self.v_out);
        Ok((q, k_t, v))
    }
}

// ---------------------------------------------------------------------------
// Q4FusedGateUp
// ---------------------------------------------------------------------------

/// Fused gate+up projection for SwiGLU: stores concatenated w1||w3 Q4 weights.
///
/// Reduces 2 Q4 matmul launches to 1 per FFN layer.
pub struct Q4FusedGateUp {
    #[cfg(feature = "wgpu")]
    weights: Q4Tensor,
    #[cfg(not(feature = "wgpu"))]
    weights: Tensor<ActiveBackend, 2>,
    gate_out: usize,
    up_out: usize,
}

impl Q4FusedGateUp {
    /// Create from a pre-built concatenated Q4 tensor (wgpu path).
    #[cfg(feature = "wgpu")]
    pub fn new(weights: Q4Tensor, gate_out: usize, up_out: usize) -> Self {
        Self {
            weights,
            gate_out,
            up_out,
        }
    }

    /// Create from a pre-built concatenated Q4 tensor (non-wgpu — pre-dequantizes).
    #[cfg(not(feature = "wgpu"))]
    pub fn new(
        weights: Q4Tensor,
        gate_out: usize,
        up_out: usize,
        device: &ActiveDevice,
        precision: DequantPrecision,
    ) -> Self {
        let [n, k] = weights.shape();
        let w_f32: Vec<f32> = match precision {
            DequantPrecision::F32 => weights.dequantize_f32(),
            DequantPrecision::F16 => weights
                .dequantize_f16()
                .into_iter()
                .map(|v| v.to_f32())
                .collect(),
        };
        let weight_tensor = Tensor::<ActiveBackend, 2>::from_data(
            burn::tensor::TensorData::new(w_f32, [n, k]),
            device,
        );
        Self {
            weights: weight_tensor,
            gate_out,
            up_out,
        }
    }

    /// Forward: single Q4 matmul → split into (gate, up) (wgpu path).
    #[cfg(feature = "wgpu")]
    pub fn forward(
        &self,
        x: Tensor<ActiveBackend, 3>,
    ) -> Result<(Tensor<ActiveBackend, 3>, Tensor<ActiveBackend, 3>), String> {
        let fused = q4_matmul(x, &self.weights)?;
        let gate = fused.clone().narrow(2, 0, self.gate_out);
        let up = fused.narrow(2, self.gate_out, self.up_out);
        Ok((gate, up))
    }

    /// Forward: pre-dequantized matmul → split into (gate, up).
    #[cfg(not(feature = "wgpu"))]
    pub fn forward(
        &self,
        x: Tensor<ActiveBackend, 3>,
    ) -> Result<(Tensor<ActiveBackend, 3>, Tensor<ActiveBackend, 3>), String> {
        let dims = x.dims();
        let (b, m, k) = (dims[0], dims[1], dims[2]);
        let total_out = self.gate_out + self.up_out;
        let input_2d = x.reshape([b * m, k]);
        let output_2d = input_2d.matmul(self.weights.clone().transpose());
        let fused = output_2d.reshape([b, m, total_out]);
        let gate = fused.clone().narrow(2, 0, self.gate_out);
        let up = fused.narrow(2, self.gate_out, self.up_out);
        Ok((gate, up))
    }
}
