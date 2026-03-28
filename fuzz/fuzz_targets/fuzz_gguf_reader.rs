//! Fuzz the GGUF reader with arbitrary bytes.
//!
//! Targets: header parsing, metadata skipping, tensor index parsing,
//! string allocation, dimension overflow, value type dispatch.

#![no_main]

use libfuzzer_sys::fuzz_target;
use voxtral_mini_realtime::gguf::GgufReader;

fuzz_target!(|data: &[u8]| {
    // Attempt to parse arbitrary bytes as GGUF — must not panic or OOM.
    // Errors (invalid magic, truncated data, bad types) are expected and fine.
    let result = GgufReader::from_bytes(data);

    // If parsing succeeded, try accessing tensor data for every tensor.
    if let Ok(mut reader) = result {
        let names: Vec<String> = reader.tensor_names().iter().map(|s| s.to_string()).collect();
        for name in &names {
            // tensor_data does seek + read_exact — should not panic on valid parse
            let _ = reader.tensor_data(name);
        }
    }
});
