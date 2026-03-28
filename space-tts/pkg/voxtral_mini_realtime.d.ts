/* tslint:disable */
/* eslint-disable */

/**
 * Tekken BPE tokenizer for browser use.
 *
 * Loads from a tekken.json string and encodes text to token IDs.
 */
export class TekkenTokenizerWasm {
    free(): void;
    [Symbol.dispose](): void;
    /**
     * Encode text to token IDs (Uint32Array).
     */
    encode(text: string): Uint32Array;
    /**
     * Load tokenizer from a JSON string (fetched from HuggingFace).
     */
    constructor(json: string);
}

/**
 * Q4 GGUF Voxtral transcription model for browser use.
 *
 * Loads a Q4-quantized GGUF model (split into ≤512 MB shards to stay
 * under the browser's 2 GB `ArrayBuffer` limit) and provides a simple
 * API for transcribing audio via WebGPU.
 */
export class VoxtralQ4 {
    free(): void;
    [Symbol.dispose](): void;
    /**
     * Append a GGUF shard to the internal buffer.
     *
     * Call this once per shard (in order), then call `loadModelFromShards`
     * to parse and load the assembled GGUF.  Each shard should be ≤512 MB
     * so it fits in a single browser `ArrayBuffer`.
     */
    appendModelShard(shard: Uint8Array): void;
    /**
     * Get the expected sample rate for input audio.
     */
    getSampleRate(): number;
    /**
     * Check if the model is loaded and ready for transcription.
     */
    isReady(): boolean;
    /**
     * Load model weights from a GGUF byte array and tokenizer JSON.
     *
     * # Arguments
     * * `gguf_bytes` - The Q4 GGUF model as a Uint8Array (~2GB)
     * * `tokenizer_json` - The tokenizer configuration as a string (from tekken.json)
     */
    loadModel(gguf_bytes: Uint8Array, tokenizer_json: string): void;
    /**
     * Parse the accumulated shards as a GGUF file and load the model.
     *
     * Must be called after all shards have been appended via `appendModelShard`.
     * Uses two-phase loading: all Q4 tensors are loaded first, then the GGUF
     * reader is dropped (freeing ~2.5 GB of shard data), and finally the
     * token embeddings are dequantized to f32 (~1.5 GiB).
     */
    loadModelFromShards(tokenizer_json: string): void;
    /**
     * Create a new VoxtralQ4 instance.
     *
     * Call `initWgpuDevice()` first, then create this, then load GGUF weights.
     */
    constructor();
    /**
     * Transcribe audio to text.
     *
     * Long audio is automatically chunked to stay within WebGPU's shared
     * memory limits (max 1200 mel frames per chunk, matching the CLI).
     *
     * # Arguments
     * * `audio` - Audio samples as a Float32Array (must be 16kHz mono)
     *
     * # Returns
     * The transcribed text.
     */
    transcribe(audio: Float32Array): Promise<string>;
}

/**
 * Q4 GGUF Voxtral TTS model for browser use.
 *
 * Loads a Q4-quantized GGUF model (split into ≤512 MB shards) and provides
 * an async API for synthesizing speech via WebGPU.
 */
export class VoxtralTts {
    free(): void;
    [Symbol.dispose](): void;
    /**
     * Append a GGUF shard to the internal buffer.
     */
    appendModelShard(shard: Uint8Array): void;
    /**
     * Check if the model and a voice are loaded.
     */
    isReady(): boolean;
    /**
     * Parse the accumulated shards as a GGUF file and load the TTS model.
     *
     * Uses two-phase loading: Q4 tensors loaded first, then GGUF reader
     * dropped (freeing shard memory), then tok_embeddings finalized.
     */
    loadModelFromShards(): void;
    /**
     * Load a voice embedding from SafeTensors bytes.
     *
     * Call this before `synthesize()`. Voices are ~200 KB each.
     */
    loadVoice(voice_bytes: Uint8Array): void;
    /**
     * Create a new VoxtralTts instance.
     *
     * Call `initWgpuDevice()` first, then create this, then load GGUF weights.
     */
    constructor();
    /**
     * Synthesize speech from token IDs.
     *
     * Returns a Float32Array of 24 kHz audio samples.
     */
    synthesize(token_ids: Uint32Array, max_frames: number): Promise<Float32Array>;
}

/**
 * Initialize the WebGPU device asynchronously.
 *
 * **Must** be called (and awaited) before creating `VoxtralQ4`.
 *
 * Manually creates the wgpu device requesting the adapter's full limits
 * (especially `max_compute_invocations_per_workgroup`) instead of relying
 * on `init_setup_async` which may end up with WebGPU spec-defaults (256).
 */
export function initWgpuDevice(): Promise<void>;

/**
 * Initialize panic hook for better error messages in browser console.
 */
export function start(): void;

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
    readonly memory: WebAssembly.Memory;
    readonly __wbg_tekkentokenizerwasm_free: (a: number, b: number) => void;
    readonly __wbg_voxtralq4_free: (a: number, b: number) => void;
    readonly __wbg_voxtraltts_free: (a: number, b: number) => void;
    readonly initWgpuDevice: () => any;
    readonly tekkentokenizerwasm_encode: (a: number, b: number, c: number) => [number, number];
    readonly tekkentokenizerwasm_new: (a: number, b: number) => [number, number, number];
    readonly voxtralq4_appendModelShard: (a: number, b: number, c: number) => void;
    readonly voxtralq4_getSampleRate: (a: number) => number;
    readonly voxtralq4_isReady: (a: number) => number;
    readonly voxtralq4_loadModel: (a: number, b: number, c: number, d: number, e: number) => [number, number];
    readonly voxtralq4_loadModelFromShards: (a: number, b: number, c: number) => [number, number];
    readonly voxtralq4_new: () => number;
    readonly voxtralq4_transcribe: (a: number, b: number, c: number) => any;
    readonly voxtraltts_appendModelShard: (a: number, b: number, c: number) => void;
    readonly voxtraltts_isReady: (a: number) => number;
    readonly voxtraltts_loadModelFromShards: (a: number) => [number, number];
    readonly voxtraltts_loadVoice: (a: number, b: number, c: number) => [number, number];
    readonly voxtraltts_new: () => number;
    readonly voxtraltts_synthesize: (a: number, b: number, c: number, d: number) => any;
    readonly start: () => void;
    readonly wasm_bindgen__closure__destroy__hd6e701f6b11034a8: (a: number, b: number) => void;
    readonly wasm_bindgen__convert__closures_____invoke__h0bf2673d9a4ed97e: (a: number, b: number, c: any, d: any) => void;
    readonly wasm_bindgen__convert__closures_____invoke__h658b39318c0bce1e: (a: number, b: number, c: any) => void;
    readonly __wbindgen_malloc: (a: number, b: number) => number;
    readonly __wbindgen_realloc: (a: number, b: number, c: number, d: number) => number;
    readonly __wbindgen_exn_store: (a: number) => void;
    readonly __externref_table_alloc: () => number;
    readonly __wbindgen_externrefs: WebAssembly.Table;
    readonly __wbindgen_free: (a: number, b: number, c: number) => void;
    readonly __externref_table_dealloc: (a: number) => void;
    readonly __wbindgen_start: () => void;
}

export type SyncInitInput = BufferSource | WebAssembly.Module;

/**
 * Instantiates the given `module`, which can either be bytes or
 * a precompiled `WebAssembly.Module`.
 *
 * @param {{ module: SyncInitInput }} module - Passing `SyncInitInput` directly is deprecated.
 *
 * @returns {InitOutput}
 */
export function initSync(module: { module: SyncInitInput } | SyncInitInput): InitOutput;

/**
 * If `module_or_path` is {RequestInfo} or {URL}, makes a request and
 * for everything else, calls `WebAssembly.instantiate` directly.
 *
 * @param {{ module_or_path: InitInput | Promise<InitInput> }} module_or_path - Passing `InitInput` directly is deprecated.
 *
 * @returns {Promise<InitOutput>}
 */
export default function __wbg_init (module_or_path?: { module_or_path: InitInput | Promise<InitInput> } | InitInput | Promise<InitInput>): Promise<InitOutput>;
