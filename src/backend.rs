//! Compile-time backend selection.
//!
//! The active Burn backend is selected by Cargo feature flags:
//! - `metal` → burn::backend::Metal (macOS, native Metal compute via CubeCL MSL)
//! - `cuda` → burn::backend::Cuda (NVIDIA, tensor cores via CubeCL CUDA)
//! - `wgpu` (default) → burn::backend::Wgpu (cross-platform, wgpu abstraction)
//!
//! Only one backend should be active at a time.

#[cfg(feature = "cuda")]
pub type ActiveBackend = burn::backend::Cuda;

#[cfg(all(feature = "metal", not(feature = "cuda")))]
pub type ActiveBackend = burn::backend::Metal;

#[cfg(all(feature = "wgpu", not(feature = "cuda"), not(feature = "metal")))]
pub type ActiveBackend = burn::backend::Wgpu;

#[cfg(not(any(feature = "wgpu", feature = "cuda", feature = "metal")))]
compile_error!("At least one backend feature must be enabled: wgpu, cuda, or metal");

/// The device type for the active backend.
pub type ActiveDevice = <ActiveBackend as burn::tensor::backend::Backend>::Device;

/// Create a default device for the active backend.
pub fn default_device() -> ActiveDevice {
    ActiveDevice::default()
}
