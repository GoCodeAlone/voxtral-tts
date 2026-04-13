use std::sync::{Condvar, Mutex};

pub struct AudioRingBuffer {
    inner: Mutex<RingInner>,
    not_full: Condvar,
    not_empty: Condvar,
}

struct RingInner {
    buffer: Vec<f32>,
    capacity: usize,
    write_pos: usize,
    read_pos: usize,
    count: usize,
}

impl AudioRingBuffer {
    pub fn new(capacity: usize) -> Self {
        Self {
            inner: Mutex::new(RingInner {
                buffer: vec![0.0; capacity],
                capacity,
                write_pos: 0,
                read_pos: 0,
                count: 0,
            }),
            not_full: Condvar::new(),
            not_empty: Condvar::new(),
        }
    }

    /// Push samples into the ring buffer. Blocks if there is insufficient space.
    pub fn push(&self, samples: &[f32]) {
        let mut inner = self.inner.lock().unwrap();
        // Wait until there's enough room
        while inner.capacity - inner.count < samples.len() {
            inner = self.not_full.wait(inner).unwrap();
        }
        for &s in samples {
            let wp = inner.write_pos;
            inner.buffer[wp] = s;
            inner.write_pos = (wp + 1) % inner.capacity;
            inner.count += 1;
        }
        self.not_empty.notify_all();
    }

    /// Try to push samples. Returns false if there is not enough space (non-blocking).
    pub fn try_push(&self, samples: &[f32]) -> bool {
        let mut inner = self.inner.lock().unwrap();
        if inner.capacity - inner.count < samples.len() {
            return false;
        }
        for &s in samples {
            let wp = inner.write_pos;
            inner.buffer[wp] = s;
            inner.write_pos = (wp + 1) % inner.capacity;
            inner.count += 1;
        }
        self.not_empty.notify_all();
        true
    }

    /// Pop up to `out.len()` samples into `out`. Returns number of samples copied.
    /// Does not block — returns 0 if empty.
    pub fn pop_into(&self, out: &mut [f32]) -> usize {
        let mut inner = self.inner.lock().unwrap();
        let n = out.len().min(inner.count);
        for i in 0..n {
            let rp = inner.read_pos;
            out[i] = inner.buffer[rp];
            inner.read_pos = (rp + 1) % inner.capacity;
        }
        inner.count -= n;
        if n > 0 {
            self.not_full.notify_all();
        }
        n
    }

    pub fn len(&self) -> usize {
        self.inner.lock().unwrap().count
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ring_buffer_push_pop() {
        let ring = AudioRingBuffer::new(4800);
        let samples = vec![0.5f32; 1920];
        ring.push(&samples);
        let mut out = vec![0.0f32; 1920];
        let n = ring.pop_into(&mut out);
        assert_eq!(n, 1920);
        assert_eq!(out[0], 0.5);
    }

    #[test]
    fn test_ring_buffer_underrun_returns_zero() {
        let ring = AudioRingBuffer::new(4800);
        let mut out = vec![0.0f32; 1920];
        let n = ring.pop_into(&mut out);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_ring_buffer_backpressure() {
        let ring = AudioRingBuffer::new(1920);
        let samples = vec![0.5f32; 1920];
        ring.push(&samples);
        assert!(!ring.try_push(&samples));
    }
}
