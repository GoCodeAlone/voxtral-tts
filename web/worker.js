/**
 * Voxtral WebWorker for off-main-thread Q4 GGUF inference (ES Module Worker).
 *
 * ASR messages:
 *   { type: 'init' }
 *   { type: 'loadModel', ggufBytes, tokenizerJson }
 *   { type: 'loadFromServer' }
 *   { type: 'transcribe', audio }
 *
 * TTS messages:
 *   { type: 'initTts' }
 *   { type: 'loadTtsFromServer' }
 *   { type: 'loadVoice', voiceName }
 *   { type: 'synthesize', tokenIds, maxFrames }
 */

import init, { VoxtralQ4, VoxtralTts, initWgpuDevice } from '../pkg/voxtral_mini_realtime.js';

let voxtral = null;
let tts = null;

self.onmessage = async (e) => {
    const { type, ...data } = e.data;

    try {
        switch (type) {
            case 'init':
                await handleInit();
                break;
            case 'loadModel':
                handleLoadModel(data.ggufBytes, data.tokenizerJson);
                break;
            case 'loadFromServer':
                await handleLoadFromServer();
                break;
            case 'transcribe':
                await handleTranscribe(data.audio);
                break;
            case 'initTts':
                await handleInitTts();
                break;
            case 'loadTtsFromServer':
                await handleLoadTtsFromServer();
                break;
            case 'loadVoice':
                await handleLoadVoice(data.voiceName);
                break;
            case 'synthesize':
                await handleSynthesize(data.tokenIds, data.maxFrames ?? 2000);
                break;
            default:
                throw new Error(`Unknown message type: ${type}`);
        }
    } catch (error) {
        self.postMessage({ type: 'error', message: error.message || String(error) });
    }
};

async function handleInit() {
    self.postMessage({ type: 'progress', stage: 'Initializing WASM...' });
    await init();
    self.postMessage({ type: 'progress', stage: 'Initializing WebGPU device...' });
    await initWgpuDevice();
    voxtral = new VoxtralQ4();
    self.postMessage({ type: 'ready' });
}

function handleLoadModel(ggufBytes, tokenizerJson) {
    if (!voxtral) throw new Error('Worker not initialized.');
    self.postMessage({ type: 'progress', stage: 'Loading Q4 GGUF model...' });

    const bytes = ggufBytes instanceof Uint8Array
        ? ggufBytes
        : new Uint8Array(ggufBytes);

    voxtral.loadModel(bytes, tokenizerJson);
    self.postMessage({ type: 'modelLoaded' });
}

async function handleLoadFromServer() {
    if (!voxtral) throw new Error('Worker not initialized.');

    // Discover shards from server
    self.postMessage({ type: 'progress', stage: 'Discovering model shards...' });
    const shardsResp = await fetch('/api/shards');
    const { shards } = await shardsResp.json();

    if (!shards || shards.length === 0) {
        throw new Error('No model shards found on server.');
    }

    // Download shards sequentially (each ≤512 MB)
    let totalBytes = 0;
    for (let i = 0; i < shards.length; i++) {
        const name = shards[i];
        self.postMessage({
            type: 'progress',
            stage: `Downloading shard ${i + 1}/${shards.length} (${name})...`,
            percent: Math.round((i / shards.length) * 60),
        });

        const resp = await fetch(`/models/voxtral-q4-shards/${name}`);
        const buf = await resp.arrayBuffer();
        voxtral.appendModelShard(new Uint8Array(buf));
        totalBytes += buf.byteLength;
    }

    // Fetch tokenizer
    self.postMessage({ type: 'progress', stage: 'Loading tokenizer...', percent: 65 });
    const tokResp = await fetch('/models/voxtral/tekken.json');
    const tokenizerJson = await tokResp.text();

    // Parse GGUF and load into WebGPU
    self.postMessage({ type: 'progress', stage: 'Loading model into WebGPU...', percent: 70 });
    voxtral.loadModelFromShards(tokenizerJson);

    self.postMessage({ type: 'modelLoaded' });
}

async function handleTranscribe(audio) {
    if (!voxtral || !voxtral.isReady()) {
        throw new Error('Model not loaded.');
    }
    self.postMessage({ type: 'progress', stage: 'Transcribing...' });

    const audioData = audio instanceof Float32Array
        ? audio
        : new Float32Array(audio);

    const text = await voxtral.transcribe(audioData);
    self.postMessage({ type: 'transcription', text });
}

// ---------------------------------------------------------------------------
// TTS handlers
// ---------------------------------------------------------------------------

async function handleInitTts() {
    self.postMessage({ type: 'progress', stage: 'Initializing WASM...' });
    await init();
    self.postMessage({ type: 'progress', stage: 'Initializing WebGPU device...' });
    await initWgpuDevice();
    tts = new VoxtralTts();
    self.postMessage({ type: 'ready', mode: 'tts' });
}

async function handleLoadTtsFromServer() {
    if (!tts) throw new Error('TTS not initialized. Call initTts first.');

    // Discover TTS shards
    self.postMessage({ type: 'progress', stage: 'Discovering TTS model shards...' });
    const shardsResp = await fetch('/api/tts-shards');
    const { shards } = await shardsResp.json();

    if (!shards || shards.length === 0) {
        throw new Error('No TTS model shards found on server.');
    }

    // Download shards sequentially
    for (let i = 0; i < shards.length; i++) {
        const name = shards[i];
        self.postMessage({
            type: 'progress',
            stage: `Downloading TTS shard ${i + 1}/${shards.length} (${name})...`,
            percent: Math.round((i / shards.length) * 70),
        });

        const resp = await fetch(`/models/voxtral-tts-q4-shards/${name}`);
        const buf = await resp.arrayBuffer();
        tts.appendModelShard(new Uint8Array(buf));
    }

    // Load model into WebGPU
    self.postMessage({ type: 'progress', stage: 'Loading TTS model into WebGPU...', percent: 80 });
    tts.loadModelFromShards();

    self.postMessage({ type: 'modelLoaded', mode: 'tts' });
}

/** @type {Map<string, boolean>} */
const voiceCache = new Map();

async function handleLoadVoice(voiceName) {
    if (!tts) throw new Error('TTS not initialized.');

    if (voiceCache.has(voiceName)) {
        self.postMessage({ type: 'voiceLoaded', voiceName });
        return;
    }

    self.postMessage({ type: 'progress', stage: `Loading voice "${voiceName}"...` });
    const resp = await fetch(`/models/voxtral-tts/voice_embedding/${voiceName}.safetensors`);
    if (!resp.ok) {
        throw new Error(`Voice "${voiceName}" not found (HTTP ${resp.status})`);
    }
    const buf = await resp.arrayBuffer();
    tts.loadVoice(new Uint8Array(buf));
    voiceCache.set(voiceName, true);
    self.postMessage({ type: 'voiceLoaded', voiceName });
}

async function handleSynthesize(tokenIds, maxFrames) {
    if (!tts || !tts.isReady()) {
        throw new Error('TTS model or voice not loaded.');
    }

    self.postMessage({ type: 'progress', stage: 'Synthesizing speech...' });

    const ids = tokenIds instanceof Uint32Array
        ? tokenIds
        : new Uint32Array(tokenIds);

    const samples = await tts.synthesize(ids, maxFrames);
    self.postMessage({ type: 'audio', samples, sampleRate: 24000 }, [samples.buffer]);
}
