/**
 * SonicSentinel AI - Audio Engine & Canvas Visualizer
 * Real Web Audio API capture, procedural sound synthesizer, and gold/amber acoustic canvas rendering
 */

class SonicAudioEngine {
    constructor() {
        this.audioCtx = null;
        this.masterGain = null;
        this.analyser = null;
        this.micStream = null;
        this.micSource = null;
        this.isPlaying = false;
        this.isMicActive = false;
        this.micStatus = "Available";
        this.activeNodes = [];
        this.bufferLength = 256;
        this.freqArray = null;
        this.timeArray = null;
    }

    init() {
        if (!this.audioCtx) {
            const AudioContext = window.AudioContext || window.webkitAudioContext;
            this.audioCtx = new AudioContext();
            this.masterGain = this.audioCtx.createGain();
            this.masterGain.gain.setValueAtTime(0.7, this.audioCtx.currentTime);
            this.masterGain.connect(this.audioCtx.destination);

            this.analyser = this.audioCtx.createAnalyser();
            this.analyser.fftSize = 512;
            this.analyser.smoothingTimeConstant = 0.85;
            this.bufferLength = this.analyser.frequencyBinCount;
            this.freqArray = new Uint8Array(this.bufferLength);
            this.timeArray = new Uint8Array(this.bufferLength);
        }
        if (this.audioCtx.state === 'suspended') {
            this.audioCtx.resume();
        }
    }

    setVolume(val) {
        if (this.masterGain && this.audioCtx) {
            const vol = Math.max(0, Math.min(1, val));
            this.masterGain.gain.setValueAtTime(vol, this.audioCtx.currentTime);
        }
    }

    stopAll() {
        this.activeNodes.forEach(node => {
            try {
                if (node.stop) node.stop();
                node.disconnect();
            } catch (e) {}
        });
        this.activeNodes = [];
        this.isPlaying = false;
    }

    playSynth(type, duration = 3.0, onComplete = null) {
        this.init();
        this.stopAll();
        this.isPlaying = true;

        const now = this.audioCtx.currentTime;
        const sr = this.audioCtx.sampleRate;
        const gain = this.audioCtx.createGain();
        gain.connect(this.analyser);
        this.analyser.connect(this.masterGain);

        switch (type) {
            case "gunshot": {
                const len = sr * duration;
                const buf = this.audioCtx.createBuffer(1, len, sr);
                const d = buf.getChannelData(0);
                for (let i = 0; i < len; i++) {
                    const t = i / sr;
                    const blast = (Math.random() * 2 - 1) * Math.exp(-t * 22);
                    const sub = Math.sin(2 * Math.PI * (80 - t * 40) * t) * Math.exp(-t * 8);
                    d[i] = blast * 0.8 + sub * 0.4;
                }
                const src = this.audioCtx.createBufferSource();
                src.buffer = buf;
                src.connect(gain);
                src.start(now);
                this.activeNodes.push(src);
                break;
            }
            case "glass": {
                const len = sr * duration;
                const buf = this.audioCtx.createBuffer(1, len, sr);
                const d = buf.getChannelData(0);
                for (let i = 0; i < len; i++) {
                    const t = i / sr;
                    const crack = (Math.random() * 2 - 1) * Math.exp(-t * 18);
                    const shard1 = Math.sin(2 * Math.PI * 5200 * t) * Math.exp(-((t - 0.05) % 0.15) * 20);
                    const shard2 = Math.sin(2 * Math.PI * 7200 * t) * Math.exp(-((t - 0.12) % 0.22) * 25);
                    d[i] = (crack * 0.6 + shard1 * 0.2 + shard2 * 0.2) * Math.exp(-t * 2.0);
                }
                const src = this.audioCtx.createBufferSource();
                src.buffer = buf;
                src.connect(gain);
                src.start(now);
                this.activeNodes.push(src);
                break;
            }
            case "siren": {
                const osc = this.audioCtx.createOscillator();
                const lfo = this.audioCtx.createOscillator();
                const lfoG = this.audioCtx.createGain();
                osc.type = "sawtooth";
                lfo.frequency.setValueAtTime(1.5, now);
                lfoG.gain.setValueAtTime(600, now);
                osc.frequency.setValueAtTime(1200, now);
                lfo.connect(osc.frequency);
                osc.connect(gain);
                osc.start(now);
                lfo.start(now);
                osc.stop(now + duration);
                lfo.stop(now + duration);
                this.activeNodes.push(osc, lfo);
                break;
            }
            case "machinery": {
                const osc1 = this.audioCtx.createOscillator();
                const osc2 = this.audioCtx.createOscillator();
                osc1.type = "sawtooth";
                osc1.frequency.setValueAtTime(70, now);
                osc2.type = "triangle";
                osc2.frequency.setValueAtTime(210, now);
                osc1.connect(gain);
                osc2.connect(gain);
                osc1.start(now);
                osc2.start(now);
                osc1.stop(now + duration);
                osc2.stop(now + duration);
                this.activeNodes.push(osc1, osc2);
                break;
            }
            case "scream": {
                const osc = this.audioCtx.createOscillator();
                osc.type = "sawtooth";
                osc.frequency.setValueAtTime(900, now);
                osc.frequency.exponentialRampToValueAtTime(1400, now + 0.5);
                osc.frequency.exponentialRampToValueAtTime(800, now + duration);
                const filter = this.audioCtx.createBiquadFilter();
                filter.type = "bandpass";
                filter.frequency.setValueAtTime(2400, now);
                osc.connect(filter);
                filter.connect(gain);
                osc.start(now);
                osc.stop(now + duration);
                this.activeNodes.push(osc, filter);
                break;
            }
            case "help": {
                if ('speechSynthesis' in window) {
                    const msg = new SpeechSynthesisUtterance("Emergency! Somebody please help me!");
                    msg.rate = 1.1;
                    window.speechSynthesis.speak(msg);
                }
                const osc = this.audioCtx.createOscillator();
                osc.type = "sine";
                osc.frequency.setValueAtTime(350, now);
                osc.frequency.linearRampToValueAtTime(550, now + 0.5);
                osc.connect(gain);
                osc.start(now);
                osc.stop(now + duration);
                this.activeNodes.push(osc);
                break;
            }
            default: {
                const len = sr * duration;
                const buf = this.audioCtx.createBuffer(1, len, sr);
                const d = buf.getChannelData(0);
                for (let i = 0; i < len; i++) {
                    d[i] = (Math.random() * 2 - 1) * 0.15;
                }
                const src = this.audioCtx.createBufferSource();
                src.buffer = buf;
                src.connect(gain);
                src.start(now);
                this.activeNodes.push(src);
                break;
            }
        }

        setTimeout(() => {
            this.stopAll();
            if (onComplete) onComplete();
        }, duration * 1000);
    }

    async startMic(onStatus) {
        this.init();
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true, video: false });
            this.micStream = stream;
            this.micSource = this.audioCtx.createMediaStreamSource(stream);
            this.micSource.connect(this.analyser);
            this.isMicActive = true;
            this.micStatus = "Active";
            if (onStatus) onStatus("Active");
            return true;
        } catch (err) {
            this.isMicActive = false;
            this.micStatus = (err.name === "NotAllowedError") ? "Permission denied" : "Disconnected";
            if (onStatus) onStatus(this.micStatus);
            return false;
        }
    }

    stopMic(onStatus) {
        if (this.micStream) {
            this.micStream.getTracks().forEach(t => t.stop());
            this.micStream = null;
        }
        if (this.micSource) {
            this.micSource.disconnect();
            this.micSource = null;
        }
        this.isMicActive = false;
        this.micStatus = "Available";
        if (onStatus) onStatus("Available");
    }

    drawWaveform(canvas, color = "#2DD4BF") {
        if (!canvas) return;
        const ctx = canvas.getContext('2d');
        const w = canvas.width;
        const h = canvas.height;

        ctx.fillStyle = "#042422";
        ctx.fillRect(0, 0, w, h);

        if (!this.analyser) {
            ctx.strokeStyle = "rgba(45, 212, 191, 0.25)";
            ctx.lineWidth = 1;
            ctx.beginPath();
            ctx.moveTo(0, h / 2);
            ctx.lineTo(w, h / 2);
            ctx.stroke();
            return;
        }

        this.analyser.getByteTimeDomainData(this.timeArray);

        // Subtle engineering grid lines
        ctx.strokeStyle = "rgba(45, 212, 191, 0.08)";
        ctx.lineWidth = 1;
        ctx.beginPath();
        for (let x = 0; x < w; x += 40) { ctx.moveTo(x, 0); ctx.lineTo(x, h); }
        for (let y = 0; y < h; y += 20) { ctx.moveTo(0, y); ctx.lineTo(w, y); }
        ctx.stroke();

        ctx.lineWidth = 2;
        ctx.strokeStyle = color;
        ctx.shadowColor = "rgba(45, 212, 191, 0.6)";
        ctx.shadowBlur = 6;
        ctx.beginPath();
        const slice = w * 1.0 / this.bufferLength;
        let x = 0;
        for (let i = 0; i < this.bufferLength; i++) {
            const v = this.timeArray[i] / 128.0;
            const y = (v * h) / 2;
            if (i === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
            x += slice;
        }
        ctx.lineTo(w, h / 2);
        ctx.stroke();
        ctx.shadowBlur = 0; // reset shadow
    }

    drawSpectrogram(canvas) {
        if (!canvas) return;
        const ctx = canvas.getContext('2d');
        const w = canvas.width;
        const h = canvas.height;

        ctx.fillStyle = "#042422";
        ctx.fillRect(0, 0, w, h);

        if (!this.analyser) return;

        this.analyser.getByteFrequencyData(this.freqArray);
        const barW = (w / this.bufferLength) * 2.2;
        let x = 0;

        for (let i = 0; i < this.bufferLength; i++) {
            const barH = (this.freqArray[i] / 255) * h;
            const intensity = this.freqArray[i] / 255;
            
            // High-tech Teal to Mint/Cyan spectrogram heatmap
            const r = Math.min(255, Math.floor(intensity * 45));
            const g = Math.min(255, Math.floor(120 + intensity * 135));
            const b = Math.min(255, Math.floor(140 + intensity * 115));

            ctx.fillStyle = `rgb(${r}, ${g}, ${b})`;
            ctx.fillRect(x, h - barH, barW, barH);
            x += barW + 1;
            if (x >= w) break;
        }
    }
}

const sonicAudio = new SonicAudioEngine();
