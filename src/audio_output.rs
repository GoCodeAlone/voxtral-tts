use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use std::sync::Arc;

use crate::ring_buffer::AudioRingBuffer;

pub struct AudioOutputStream {
    _stream: cpal::Stream,
    ring: Arc<AudioRingBuffer>,
    sample_rate: u32,
}

impl AudioOutputStream {
    /// Open the default audio output device at the given sample rate (mono f32).
    pub fn open(sample_rate: u32) -> Result<Self, String> {
        let host = cpal::default_host();
        let device = host
            .default_output_device()
            .ok_or_else(|| "no audio output device".to_string())?;

        let ring = Arc::new(AudioRingBuffer::new(23040)); // 12 frames × 1920
        let ring_clone = Arc::clone(&ring);

        let config = cpal::StreamConfig {
            channels: 1,
            sample_rate: cpal::SampleRate(sample_rate),
            buffer_size: cpal::BufferSize::Default,
        };

        let stream = device
            .build_output_stream(
                &config,
                move |data: &mut [f32], _| {
                    let n = ring_clone.pop_into(data);
                    for s in &mut data[n..] {
                        *s = 0.0;
                    }
                },
                |err| eprintln!("[audio] output error: {err}"),
                None,
            )
            .map_err(|e| format!("build output stream: {e}"))?;

        stream.play().map_err(|e| format!("play: {e}"))?;

        Ok(Self {
            _stream: stream,
            ring,
            sample_rate,
        })
    }

    pub fn ring(&self) -> &Arc<AudioRingBuffer> {
        &self.ring
    }

    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    /// Block until the ring buffer is empty (playback complete).
    pub fn wait_drain(&self) {
        while !self.ring.is_empty() {
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        // Extra 50ms for cpal to flush its internal buffer
        std::thread::sleep(std::time::Duration::from_millis(50));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_audio_output_opens_default_device() {
        let result = AudioOutputStream::open(24000);
        match result {
            Ok(stream) => assert_eq!(stream.sample_rate(), 24000),
            Err(e) => {
                eprintln!("SKIP: no audio device — {e}");
            }
        }
    }
}
