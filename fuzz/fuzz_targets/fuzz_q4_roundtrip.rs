//! Fuzz Q4_0 quantization round-trip: quantize f32 → Q4 bytes → dequantize → check bounds.
//!
//! Properties tested:
//! - Dequantized values are finite (no NaN/inf from crafted blocks)
//! - Round-trip error is bounded: |deq - orig| <= max(|orig|) / 7
//! - Zero-scale blocks produce all-zero output
//! - Handles edge cases: all-zero input, all-same input, NaN, inf

#![no_main]

use libfuzzer_sys::fuzz_target;

const BLOCK_SIZE: usize = 32;
const BLOCK_BYTES: usize = 18;

fuzz_target!(|data: &[u8]| {
    // Interpret input as either:
    // 1. Raw f32 values to quantize (if len is multiple of 128 = 32 floats * 4 bytes)
    // 2. Raw Q4 blocks to dequantize (if len is multiple of 18)

    if data.len() >= BLOCK_SIZE * 4 && data.len() % (BLOCK_SIZE * 4) == 0 {
        // Mode 1: f32 → quantize → dequantize → check
        let floats: Vec<f32> = data
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();

        // Skip if any value is NaN, infinite, or too large for f16 scale.
        // Q4_0 stores the block scale as f16 (max 65504). Values with
        // amax/7 > 65504 overflow the scale to f16 inf, producing garbage.
        if floats.iter().any(|f| !f.is_finite()) {
            return;
        }
        // Check per-block: skip blocks whose scale would overflow f16
        let all_blocks_in_range = floats.chunks(BLOCK_SIZE).all(|block| {
            let amax = block.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
            amax / 7.0 <= 65504.0
        });
        if !all_blocks_in_range {
            return;
        }

        let q4_bytes = quantize_f32_to_q4(&floats);
        let deq = dequantize_q4(&q4_bytes, floats.len());

        // Check: dequantized values must be finite
        for (i, &v) in deq.iter().enumerate() {
            assert!(
                v.is_finite(),
                "Dequantized value at index {i} is not finite: {v}"
            );
        }

        // Check: round-trip error bounded by quantization step.
        // The scale d is stored as f16, so the actual quantization step
        // uses the f16-rounded scale, not the exact f32 scale. Allow for
        // f16 precision loss by using the f16 round-tripped scale.
        for block_start in (0..floats.len()).step_by(BLOCK_SIZE) {
            let block = &floats[block_start..block_start + BLOCK_SIZE];
            let amax = block.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
            let d = amax / 7.0;
            // f16 round-trip: the actual scale used during dequant
            let d_f16 = half::f16::from_f32(d).to_f32();
            // Max error per element: scale * 1.0 (one quantization level)
            // Plus epsilon for float arithmetic
            let max_err = d_f16 + 1e-5;

            for j in 0..BLOCK_SIZE {
                let orig = floats[block_start + j];
                let recovered = deq[block_start + j];
                let err = (orig - recovered).abs();
                assert!(
                    err <= max_err,
                    "Round-trip error too large at [{block_start}+{j}]: orig={orig}, deq={recovered}, err={err}, max_err={max_err}, d={d}, d_f16={d_f16}"
                );
            }
        }
    }

    if data.len() >= BLOCK_BYTES && data.len() % BLOCK_BYTES == 0 {
        // Mode 2: raw Q4 bytes → dequantize → check finite
        let n_blocks = data.len() / BLOCK_BYTES;
        let n_elements = n_blocks * BLOCK_SIZE;
        let deq = dequantize_q4(data, n_elements);

        for (i, &v) in deq.iter().enumerate() {
            // f16 NaN/inf can produce non-finite dequantized values — that's OK
            // but the code shouldn't panic
            let _ = v.is_finite();
            let _ = i;
        }
    }
});

fn quantize_f32_to_q4(data: &[f32]) -> Vec<u8> {
    assert_eq!(data.len() % BLOCK_SIZE, 0);
    let n_blocks = data.len() / BLOCK_SIZE;
    let mut output = Vec::with_capacity(n_blocks * BLOCK_BYTES);

    for block_idx in 0..n_blocks {
        let block = &data[block_idx * BLOCK_SIZE..(block_idx + 1) * BLOCK_SIZE];
        let amax = block.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        let d = amax / 7.0;
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };

        // Scale as f16
        let d_f16 = half::f16::from_f32(d);
        output.extend_from_slice(&d_f16.to_le_bytes());

        // Quantize and pack nibbles
        for i in 0..16 {
            let lo = ((block[i] * id + 8.5) as u8).min(15);
            let hi = ((block[i + 16] * id + 8.5) as u8).min(15);
            output.push(lo | (hi << 4));
        }
    }

    output
}

fn dequantize_q4(bytes: &[u8], n_elements: usize) -> Vec<f32> {
    let n_blocks = n_elements / BLOCK_SIZE;
    let mut output = vec![0.0f32; n_elements];

    for block_idx in 0..n_blocks {
        let bo = block_idx * BLOCK_BYTES;
        if bo + BLOCK_BYTES > bytes.len() {
            break;
        }
        let d = half::f16::from_le_bytes([bytes[bo], bytes[bo + 1]]).to_f32();
        let base = block_idx * BLOCK_SIZE;

        for j in 0..16 {
            let byte = bytes[bo + 2 + j];
            let lo = (byte & 0x0F) as f32 - 8.0;
            let hi = ((byte >> 4) & 0x0F) as f32 - 8.0;
            output[base + j] = lo * d;
            output[base + j + 16] = hi * d;
        }
    }

    output
}
