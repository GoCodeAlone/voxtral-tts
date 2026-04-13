//! Q4_0 quantized weight tensor.
//!
//! [`Q4Tensor`] stores raw Q4_0 block bytes in CPU memory (`raw_data`) so it
//! compiles on every backend.  On wgpu builds the bytes are also uploaded to a
//! GPU storage buffer so the fused WGSL shader can dequantize on-the-fly.
//!
//! For non-wgpu backends (cuda/metal/rocm) call [`Q4Tensor::dequantize_f32`] or
//! [`Q4Tensor::dequantize_f16`] to produce a flat weight vector that can be
//! uploaded to the target device.

use anyhow::{ensure, Result};

#[cfg(feature = "wgpu")]
use burn::backend::wgpu::{WgpuDevice, WgpuRuntime};
#[cfg(feature = "wgpu")]
use burn::backend::Wgpu;
#[cfg(feature = "wgpu")]
use burn::tensor::{Tensor, TensorData};
#[cfg(feature = "wgpu")]
use cubecl::client::ComputeClient;
#[cfg(feature = "wgpu")]
use cubecl::server::Handle;
#[cfg(feature = "wgpu")]
use cubecl::Runtime;

/// A Q4_0 quantized weight tensor.
///
/// `raw_data` always holds the CPU copy of the Q4_0 bytes so the struct
/// compiles on every backend.  On wgpu builds the bytes are additionally
/// uploaded to a GPU buffer for the fused WGSL shader path.
///
/// Buffer layout: 18 bytes per block of 32 elements
/// - bytes [0..1]: f16 scale (little-endian)
/// - bytes [2..17]: 16 packed bytes, lower nibble → element i, upper nibble → element i+16
pub struct Q4Tensor {
    /// CPU copy — always present regardless of backend.
    raw_data: Vec<u8>,
    shape: [usize; 2],
    num_blocks: usize,
    /// GPU handle — wgpu builds only.
    #[cfg(feature = "wgpu")]
    pub(crate) handle: Handle,
    #[cfg(feature = "wgpu")]
    client: ComputeClient<WgpuRuntime>,
    #[cfg(feature = "wgpu")]
    device: WgpuDevice,
}

impl Q4Tensor {
    /// Upload raw Q4_0 bytes and (on wgpu) also copy to GPU storage buffer.
    ///
    /// Shape is `[N, K]` = `[out_features, in_features]`, matching PyTorch/GGUF
    /// convention. `raw_bytes` must contain exactly `(N * K / 32) * 18` bytes.
    #[cfg(feature = "wgpu")]
    pub fn from_q4_bytes(raw_bytes: &[u8], shape: [usize; 2], device: &WgpuDevice) -> Result<Self> {
        let [n, k] = shape;
        let num_elements = k * n;
        ensure!(
            num_elements % 32 == 0,
            "Q4_0 requires element count divisible by 32, got {num_elements}"
        );
        let num_blocks = num_elements / 32;
        let expected_bytes = num_blocks * 18;
        ensure!(
            raw_bytes.len() == expected_bytes,
            "Q4_0 byte count mismatch: expected {expected_bytes} for {num_blocks} blocks, got {}",
            raw_bytes.len()
        );

        let client = WgpuRuntime::client(device);

        // Pad to 4-byte alignment for array<u32> access in the WGSL shader.
        let padded = if !raw_bytes.len().is_multiple_of(4) {
            let pad = 4 - (raw_bytes.len() % 4);
            let mut buf = raw_bytes.to_vec();
            buf.resize(raw_bytes.len() + pad, 0);
            buf
        } else {
            raw_bytes.to_vec()
        };
        let handle = client.create_from_slice(&padded);

        Ok(Self {
            raw_data: raw_bytes.to_vec(),
            shape,
            num_blocks,
            handle,
            client,
            device: device.clone(),
        })
    }

    /// Store raw Q4_0 bytes for non-wgpu backends.
    ///
    /// Shape is `[N, K]` = `[out_features, in_features]`.
    #[cfg(not(feature = "wgpu"))]
    pub fn from_q4_bytes(raw_bytes: &[u8], shape: [usize; 2]) -> Result<Self> {
        let [n, k] = shape;
        let num_elements = k * n;
        ensure!(
            num_elements % 32 == 0,
            "Q4_0 requires element count divisible by 32, got {num_elements}"
        );
        let num_blocks = num_elements / 32;
        let expected_bytes = num_blocks * 18;
        ensure!(
            raw_bytes.len() == expected_bytes,
            "Q4_0 byte count mismatch: expected {expected_bytes} for {num_blocks} blocks, got {}",
            raw_bytes.len()
        );
        Ok(Self {
            raw_data: raw_bytes.to_vec(),
            shape,
            num_blocks,
        })
    }

    /// Logical weight dimensions `[N, K]` = `[out_features, in_features]`.
    pub fn shape(&self) -> [usize; 2] {
        self.shape
    }

    /// Number of Q4_0 blocks in the tensor.
    pub fn num_blocks(&self) -> usize {
        self.num_blocks
    }

    /// Dequantize all Q4_0 blocks to f32 (CPU).
    ///
    /// Returns a flat `Vec<f32>` of shape `[N, K]` in row-major order.
    /// Nibble order matches the WGSL shader exactly:
    /// - Lower nibble of data byte `i` → element `i` within the block
    /// - Upper nibble of data byte `i` → element `i + 16` within the block
    pub fn dequantize_f32(&self) -> Vec<f32> {
        let [n, k] = self.shape;
        let mut out = vec![0.0f32; n * k];
        for block_idx in 0..self.num_blocks {
            let block_offset = block_idx * 18;
            let scale =
                half::f16::from_le_bytes([self.raw_data[block_offset], self.raw_data[block_offset + 1]])
                    .to_f32();
            let base = block_idx * 32;
            for i in 0..16 {
                let byte = self.raw_data[block_offset + 2 + i];
                let lo = (byte & 0x0F) as f32 - 8.0;
                let hi = ((byte >> 4) & 0x0F) as f32 - 8.0;
                out[base + i] = lo * scale;
                out[base + i + 16] = hi * scale;
            }
        }
        out
    }

    /// Dequantize all Q4_0 blocks to f16 (CPU).
    ///
    /// Returns a flat `Vec<half::f16>` of shape `[N, K]` in row-major order.
    /// Use when uploading to a backend that performs f16 matmul natively
    /// (e.g. Metal with `DequantPrecision::F16`).
    pub fn dequantize_f16(&self) -> Vec<half::f16> {
        let [n, k] = self.shape;
        let mut out = vec![half::f16::ZERO; n * k];
        for block_idx in 0..self.num_blocks {
            let block_offset = block_idx * 18;
            let scale =
                half::f16::from_le_bytes([self.raw_data[block_offset], self.raw_data[block_offset + 1]]);
            let base = block_idx * 32;
            for i in 0..16 {
                let byte = self.raw_data[block_offset + 2 + i];
                let lo = half::f16::from_f32((byte & 0x0F) as f32 - 8.0);
                let hi = half::f16::from_f32(((byte >> 4) & 0x0F) as f32 - 8.0);
                out[base + i] = lo * scale;
                out[base + i + 16] = hi * scale;
            }
        }
        out
    }

    /// Dequantize the Q4_0 data to a full-precision `Tensor<Wgpu, 2>`.
    ///
    /// Reads raw bytes back from GPU and dequantizes on CPU.
    /// Intended for diagnostics and testing — the hot path uses
    /// [`q4_matmul`](super::op::q4_matmul) which dequantizes on GPU.
    #[cfg(feature = "wgpu")]
    pub fn dequantize(&self) -> Tensor<Wgpu, 2> {
        let [n, k] = self.shape;
        let output = self.dequantize_f32();
        let tensor_data = TensorData::new(output, [n, k]);
        Tensor::from_data(tensor_data, &self.device)
    }

    /// Read raw Q4_0 bytes from GPU (wgpu only).
    #[cfg(feature = "wgpu")]
    pub fn read_bytes(&self) -> Vec<u8> {
        let raw = self.client.read_one(self.handle.clone());
        let expected = self.num_blocks * 18;
        assert!(
            raw.len() >= expected,
            "Q4Tensor::read_bytes: GPU buffer shorter than expected: got {}, expected {expected}",
            raw.len()
        );
        raw[..expected].to_vec()
    }
}
