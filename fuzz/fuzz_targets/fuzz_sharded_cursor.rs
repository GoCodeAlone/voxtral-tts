//! Fuzz the ShardedCursor with arbitrary shard layouts and seek patterns.
//!
//! Targets: cross-shard reads, seek arithmetic, edge cases at shard boundaries.

#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;
use std::io::{Read, Seek, SeekFrom};
use voxtral_mini_realtime::gguf::ShardedCursor;

#[derive(Arbitrary, Debug)]
struct FuzzInput {
    /// Shard sizes (capped to small values to avoid OOM).
    shard_sizes: Vec<u8>,
    /// Sequence of operations to perform.
    ops: Vec<Op>,
}

#[derive(Arbitrary, Debug)]
enum Op {
    /// Read N bytes.
    Read(u8),
    /// Seek from start.
    SeekStart(u16),
    /// Seek from current (signed).
    SeekCurrent(i16),
    /// Seek from end (signed).
    SeekEnd(i16),
}

fuzz_target!(|input: FuzzInput| {
    // Create shards with deterministic content (byte = position % 256)
    let shards: Vec<Vec<u8>> = input
        .shard_sizes
        .iter()
        .take(8) // max 8 shards
        .enumerate()
        .map(|(i, &size)| {
            let size = size as usize;
            (0..size).map(|j| ((i * 256 + j) & 0xFF) as u8).collect()
        })
        .collect();

    if shards.is_empty() {
        return;
    }

    let mut cursor = ShardedCursor::new(shards);

    for op in &input.ops {
        match op {
            Op::Read(n) => {
                let mut buf = vec![0u8; *n as usize];
                let _ = cursor.read(&mut buf);
            }
            Op::SeekStart(pos) => {
                let _ = cursor.seek(SeekFrom::Start(*pos as u64));
            }
            Op::SeekCurrent(offset) => {
                let _ = cursor.seek(SeekFrom::Current(*offset as i64));
            }
            Op::SeekEnd(offset) => {
                let _ = cursor.seek(SeekFrom::End(*offset as i64));
            }
        }
    }
});
