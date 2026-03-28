/**
 * Voxtral TTS client — wraps worker communication and audio playback.
 */

// Pre-tokenized via tiktoken-rs Tekken encoder (offset 1000).
// Avoids needing tiktoken in the browser.
const TEKKEN_OFFSET = 1000;

export class VoxtralTtsClient {
    constructor() {
        this.worker = null;
        this.pending = null;
        this.onProgress = null;
        this.onError = null;
    }

    async init() {
        return new Promise((resolve, reject) => {
            this.worker = new Worker('./worker.js', { type: 'module' });
            this.worker.onmessage = (e) => this._handleMessage(e.data);
            this.worker.onerror = (e) => {
                if (this.onError) this.onError(e.message);
                reject(e);
            };
            this.pending = { resolve, reject };
            this.worker.postMessage({ type: 'init' });
        });
    }

    async loadFromServer() {
        return new Promise((resolve, reject) => {
            this.pending = { resolve, reject };
            this.worker.postMessage({ type: 'loadFromServer' });
        });
    }

    async loadVoice(voiceName) {
        return new Promise((resolve, reject) => {
            this.pending = { resolve, reject };
            this.worker.postMessage({ type: 'loadVoice', voiceName });
        });
    }

    /**
     * Synthesize speech from token IDs.
     * @param {Uint32Array} tokenIds - Tekken token IDs (with 1000 offset)
     * @param {number} maxFrames - Max audio frames to generate
     * @returns {Promise<{samples: Float32Array, sampleRate: number}>}
     */
    async synthesize(tokenIds, maxFrames = 200) {
        return new Promise((resolve, reject) => {
            this.pending = { resolve, reject };
            this.worker.postMessage({
                type: 'synthesize',
                tokenIds: Array.from(tokenIds),
                maxFrames,
            });
        });
    }

    async checkCache() {
        return new Promise((resolve) => {
            this.pending = { resolve };
            this.worker.postMessage({ type: 'checkCache' });
        });
    }

    /**
     * Tokenize text via the WASM Tekken BPE encoder.
     * @param {string} text
     * @returns {Promise<Uint32Array>}
     */
    async tokenize(text) {
        return new Promise((resolve, reject) => {
            this.pending = { resolve, reject };
            this.worker.postMessage({ type: 'tokenize', text });
        });
    }

    async clearCache() {
        return new Promise((resolve) => {
            this.pending = { resolve };
            this.worker.postMessage({ type: 'clearCache' });
        });
    }

    _handleMessage(data) {
        switch (data.type) {
            case 'ready':
            case 'modelLoaded':
            case 'voiceLoaded':
                if (this.pending) { this.pending.resolve(data); this.pending = null; }
                break;
            case 'audio':
                if (this.pending) {
                    this.pending.resolve({
                        samples: new Float32Array(data.samples),
                        sampleRate: data.sampleRate,
                    });
                    this.pending = null;
                }
                break;
            case 'tokenized':
                if (this.pending) {
                    this.pending.resolve(new Uint32Array(data.tokenIds));
                    this.pending = null;
                }
                break;
            case 'cacheStatus':
                if (this.pending) { this.pending.resolve(data); this.pending = null; }
                break;
            case 'cacheCleared':
                if (this.pending) { this.pending.resolve(data); this.pending = null; }
                break;
            case 'progress':
                if (this.onProgress) this.onProgress(data.stage, data.percent);
                break;
            case 'error':
                if (this.pending) {
                    this.pending.reject(new Error(data.message));
                    this.pending = null;
                } else if (this.onError) {
                    this.onError(data.message);
                }
                break;
        }
    }
}

/**
 * Create a WAV blob from Float32Array samples.
 */
export function samplesToWavBlob(samples, sampleRate) {
    const numChannels = 1;
    const bitsPerSample = 16;
    const byteRate = sampleRate * numChannels * (bitsPerSample / 8);
    const blockAlign = numChannels * (bitsPerSample / 8);
    const dataSize = samples.length * (bitsPerSample / 8);
    const buffer = new ArrayBuffer(44 + dataSize);
    const view = new DataView(buffer);

    // RIFF header
    writeString(view, 0, 'RIFF');
    view.setUint32(4, 36 + dataSize, true);
    writeString(view, 8, 'WAVE');

    // fmt chunk
    writeString(view, 12, 'fmt ');
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true); // PCM
    view.setUint16(22, numChannels, true);
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, byteRate, true);
    view.setUint16(32, blockAlign, true);
    view.setUint16(34, bitsPerSample, true);

    // data chunk
    writeString(view, 36, 'data');
    view.setUint32(40, dataSize, true);

    let offset = 44;
    for (let i = 0; i < samples.length; i++) {
        const s = Math.max(-1, Math.min(1, samples[i]));
        view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
        offset += 2;
    }

    return new Blob([buffer], { type: 'audio/wav' });
}

function writeString(view, offset, str) {
    for (let i = 0; i < str.length; i++) {
        view.setUint8(offset + i, str.charCodeAt(i));
    }
}
