/**
 * Voxtral TTS WebWorker — Q4 GGUF text-to-speech off the main thread.
 *
 * Messages:
 *   { type: 'init' }
 *   { type: 'loadFromServer' }
 *   { type: 'loadVoice', voiceName }
 *   { type: 'synthesize', tokenIds, maxFrames }
 *   { type: 'clearCache' }
 *   { type: 'checkCache' }
 */

import init, { VoxtralTts, TekkenTokenizerWasm, initWgpuDevice } from './pkg/voxtral_mini_realtime.js';

const HF_MODEL = "https://huggingface.co/TrevorJS/voxtral-tts-q4-gguf/resolve/main";
const HF_VOICES = "https://huggingface.co/mistralai/Voxtral-4B-TTS-2603/resolve/main/voice_embedding";
const HF_TOKENIZER = "https://huggingface.co/mistralai/Voxtral-4B-TTS-2603/resolve/main/tekken.json";
const SHARD_NAMES = ["shard-aa", "shard-ab", "shard-ac", "shard-ad", "shard-ae", "shard-af"];
const CACHE_NAME = "voxtral-tts-weights-v1";

let tts = null;
let tokenizer = null;

self.onmessage = async (e) => {
    const { type, ...data } = e.data;

    try {
        switch (type) {
            case 'init':
                await handleInit();
                break;
            case 'loadFromServer':
                await handleLoadFromServer();
                break;
            case 'loadVoice':
                await handleLoadVoice(data.voiceName);
                break;
            case 'tokenize':
                handleTokenize(data.text);
                break;
            case 'synthesize':
                await handleSynthesize(data.tokenIds, data.maxFrames ?? 200);
                break;
            case 'clearCache':
                await handleClearCache();
                break;
            case 'checkCache':
                await handleCheckCache();
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
    tts = new VoxtralTts();
    self.postMessage({ type: 'ready' });
}

async function cachedFetch(cache, url) {
    const cached = await cache.match(url);
    if (cached) return { response: cached, fromCache: true };
    const resp = await fetch(url);
    if (!resp.ok) {
        throw new Error(`Failed to download ${url}: ${resp.status} ${resp.statusText}`);
    }
    await cache.put(url, resp.clone());
    return { response: resp, fromCache: false };
}

async function handleLoadFromServer() {
    if (!tts) throw new Error('Worker not initialized.');

    const cache = await caches.open(CACHE_NAME);

    for (let i = 0; i < SHARD_NAMES.length; i++) {
        const name = SHARD_NAMES[i];
        const url = `${HF_MODEL}/${name}`;

        self.postMessage({
            type: 'progress',
            stage: `Loading ${name} (${i + 1}/${SHARD_NAMES.length})...`,
            percent: Math.round((i / SHARD_NAMES.length) * 60),
        });

        const { response, fromCache } = await cachedFetch(cache, url);

        if (fromCache) {
            self.postMessage({
                type: 'progress',
                stage: `Loaded ${name} from cache (${i + 1}/${SHARD_NAMES.length})`,
                percent: Math.round(((i + 1) / SHARD_NAMES.length) * 60),
            });
        }

        const buf = await response.arrayBuffer();
        tts.appendModelShard(new Uint8Array(buf));
    }

    // Load tokenizer
    self.postMessage({ type: 'progress', stage: 'Loading tokenizer...', percent: 65 });
    const { response: tokResp } = await cachedFetch(cache, HF_TOKENIZER);
    const tokenizerJson = await tokResp.text();
    tokenizer = new TekkenTokenizerWasm(tokenizerJson);

    // Finalize model
    self.postMessage({ type: 'progress', stage: 'Loading into WebGPU...', percent: 70 });
    tts.loadModelFromShards();

    self.postMessage({ type: 'modelLoaded' });
}

function handleTokenize(text) {
    if (!tokenizer) throw new Error('Tokenizer not loaded.');
    const ids = tokenizer.encode(text);
    self.postMessage({ type: 'tokenized', tokenIds: Array.from(ids) });
}

const voiceCache = new Map();

async function handleLoadVoice(voiceName) {
    if (!tts) throw new Error('Worker not initialized.');

    if (voiceCache.has(voiceName)) {
        self.postMessage({ type: 'voiceLoaded', voiceName });
        return;
    }

    self.postMessage({ type: 'progress', stage: `Loading voice "${voiceName}"...` });

    const cache = await caches.open(CACHE_NAME);
    const url = `${HF_VOICES}/${voiceName}.safetensors`;
    const { response } = await cachedFetch(cache, url);
    const buf = await response.arrayBuffer();
    tts.loadVoice(new Uint8Array(buf));
    voiceCache.set(voiceName, true);

    self.postMessage({ type: 'voiceLoaded', voiceName });
}

async function handleSynthesize(tokenIds, maxFrames) {
    if (!tts || !tts.isReady()) {
        throw new Error('Model or voice not loaded.');
    }

    self.postMessage({ type: 'progress', stage: 'Synthesizing speech...' });

    const ids = tokenIds instanceof Uint32Array
        ? tokenIds
        : new Uint32Array(tokenIds);

    const samples = await tts.synthesize(ids, maxFrames);
    self.postMessage({ type: 'audio', samples: Array.from(samples), sampleRate: 24000 });
}

async function handleClearCache() {
    const deleted = await caches.delete(CACHE_NAME);
    voiceCache.clear();
    self.postMessage({ type: 'cacheCleared', deleted });
}

async function handleCheckCache() {
    try {
        const cache = await caches.open(CACHE_NAME);
        const keys = await cache.keys();
        const shardsCached = SHARD_NAMES.filter(name =>
            keys.some(k => k.url.endsWith(name))
        );
        self.postMessage({
            type: 'cacheStatus',
            cached: shardsCached.length === SHARD_NAMES.length,
            shardsCached: shardsCached.length,
            shardsTotal: SHARD_NAMES.length,
        });
    } catch {
        self.postMessage({ type: 'cacheStatus', cached: false, shardsCached: 0, shardsTotal: SHARD_NAMES.length });
    }
}
