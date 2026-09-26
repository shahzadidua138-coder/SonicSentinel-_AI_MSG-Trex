/**
 * SonicSentinel AI - Interactive Homepage Engine
 * Ambient acoustic waves, Web Audio API oscilloscope, dual-model live simulation,
 * interactive sound presets, real-time dataset playback, and trained ML fast APIs.
 */

document.addEventListener('DOMContentLoaded', () => {
    initThemeToggle();
    initScrollAnimations();
    initAmbientWaveCanvas();
    initHeroMicToggle();
    initInteractiveStudio();
    initHeaderScroll();
    initMobileNav();
    initCategoryAudition();
    loadIncidentFeed();
});

/* ==========================================================================
   0. THEME TOGGLE & PERSISTENCE ENGINE (Light / Dark Mode)
   ========================================================================== */
function initThemeToggle() {
    const themeBtn = document.getElementById('themeToggleBtn');
    const currentTheme = localStorage.getItem('sonicsentinel_theme') || 'light';

    function applyTheme(theme) {
        document.documentElement.setAttribute('data-theme', theme);
        localStorage.setItem('sonicsentinel_theme', theme);
        if (themeBtn) {
            const icon = themeBtn.querySelector('i');
            if (icon) {
                icon.className = (theme === 'dark') ? 'fa-solid fa-moon' : 'fa-regular fa-sun';
            }
        }
    }

    applyTheme(currentTheme);

    if (themeBtn) {
        themeBtn.addEventListener('click', (e) => {
            e.preventDefault();
            const activeTheme = document.documentElement.getAttribute('data-theme') || 'light';
            const newTheme = (activeTheme === 'dark') ? 'light' : 'dark';
            applyTheme(newTheme);
        });
    }
}

/* ==========================================================================
   0.1 SMOOTH SCROLL REVEAL & INTERSECTION OBSERVER ANIMATIONS
   ========================================================================== */
function initScrollAnimations() {
    const targets = document.querySelectorAll(
        '.reveal-on-scroll, .why-feature-card, .pipeline-step-node, .category-card-item, .metric-stat-box, .section, .why-cards-container'
    );
    if (!targets.length) return;

    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('is-visible');
                observer.unobserve(entry.target);
            }
        });
    }, {
        threshold: 0.1,
        rootMargin: '0px 0px -40px 0px'
    });

    targets.forEach((el, idx) => {
        el.classList.add('reveal-on-scroll');
        if (!el.style.transitionDelay && (idx % 3 !== 0)) {
            el.style.transitionDelay = `${(idx % 3) * 0.12}s`;
        }
        observer.observe(el);
    });
}

/* ==========================================================================
   1. AMBIENT FLUID WAVE CANVAS (Airy luminous wave ribbons across background)
   ========================================================================== */
function initAmbientWaveCanvas() {
    const canvas = document.getElementById('ambientWaveCanvas');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    let width, height;
    let step = 0;

    function resize() {
        width = canvas.width = window.innerWidth;
        height = canvas.height = window.innerHeight;
    }
    window.addEventListener('resize', resize);
    resize();

    function drawWaves() {
        ctx.clearRect(0, 0, width, height);

        // Wave 1: Gentle teal ribbon
        ctx.beginPath();
        ctx.lineWidth = 1.8;
        ctx.strokeStyle = 'rgba(45, 212, 191, 0.22)';
        for (let x = 0; x < width; x += 10) {
            const y = Math.sin(x * 0.003 + step * 0.015) * 45 + 
                      Math.cos(x * 0.0015 - step * 0.01) * 30 + 
                      (height * 0.35);
            if (x === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
        }
        ctx.stroke();

        // Wave 2: Cyan harmonic ribbon
        ctx.beginPath();
        ctx.lineWidth = 1.2;
        ctx.strokeStyle = 'rgba(20, 184, 166, 0.16)';
        for (let x = 0; x < width; x += 10) {
            const y = Math.sin(x * 0.004 - step * 0.02) * 55 + 
                      Math.sin(x * 0.002 + step * 0.008) * 40 + 
                      (height * 0.42);
            if (x === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
        }
        ctx.stroke();

        // Wave 3: Deep subtle bass ribbon
        ctx.beginPath();
        ctx.lineWidth = 2.2;
        ctx.strokeStyle = 'rgba(13, 148, 136, 0.12)';
        for (let x = 0; x < width; x += 12) {
            const y = Math.cos(x * 0.0025 + step * 0.012) * 65 + 
                      (height * 0.6);
            if (x === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
        }
        ctx.stroke();

        step++;
        requestAnimationFrame(drawWaves);
    }
    drawWaves();
}

/* ==========================================================================
   2. HERO MIC BUTTON INTERACTIVE TOGGLE
   ========================================================================== */
function initHeroMicToggle() {
    const micBtn = document.getElementById('heroMicBtn');
    const statusLabel = document.getElementById('heroAudioStatus');
    const soundBars = document.querySelectorAll('.hero-soundwave-anim-strip .sw-bar');
    if (!micBtn) return;

    let isListening = false;

    micBtn.addEventListener('click', () => {
        isListening = !isListening;
        if (isListening) {
            micBtn.classList.add('listening');
            micBtn.innerHTML = '<i class="fa-solid fa-waveform-lines"></i>';
            if (statusLabel) statusLabel.textContent = 'Listening Live: Acoustic Stream Active';
            soundBars.forEach(bar => {
                bar.style.animationDuration = '0.5s';
                bar.style.filter = 'drop-shadow(0 0 6px #2DD4BF)';
            });

            // Smooth scroll down to interactive studio
            const studio = document.getElementById('interactiveStudio');
            if (studio) {
                studio.scrollIntoView({ behavior: 'smooth' });
            }
            // Trigger studio mic
            const studioToggle = document.getElementById('studioMicToggleBox');
            if (studioToggle && !isStudioLive) {
                studioToggle.click();
            }
        } else {
            micBtn.classList.remove('listening');
            micBtn.innerHTML = '<i class="fa-solid fa-microphone"></i>';
            if (statusLabel) statusLabel.textContent = 'Click mic to test live audio stream';
            soundBars.forEach(bar => {
                bar.style.animationDuration = '1.2s';
                bar.style.filter = 'none';
            });
            if (isStudioLive) {
                const studioToggle = document.getElementById('studioMicToggleBox');
                if (studioToggle) studioToggle.click();
            }
        }
    });
}

/* ==========================================================================
   3. INTERACTIVE ACOUSTIC STUDIO (SRS §1.2 & §1.6 Steps 2, 8, 10, 11, 12, 13)
   ========================================================================== */
let audioCtx = null;
let analyser = null;
let micStream = null;
let isStudioLive = false;
let animWaveId = null;
let activeAudioElement = null;
let liveMicPollTimer = null;

const PRESET_DATA = {
    'gunshot': {
        name: 'Gunshot',
        severity: 'Critical',
        sevClass: 'critical',
        pyConf: 98.4,
        gtmConf: 96.1,
        snr: '34.2 dB',
        quality: 'Good',
        margin: '94.2%',
        freq: 'Impulsive transient: 100 Hz - 8,000 Hz broadband shockwave',
        recommendation: 'Immediate lockdown and notify law enforcement'
    },
    'glass': {
        name: 'Glass Breaking',
        severity: 'High',
        sevClass: 'high',
        pyConf: 95.8,
        gtmConf: 94.2,
        snr: '29.7 dB',
        quality: 'Good',
        margin: '89.5%',
        freq: 'High-frequency scatter: 2,500 Hz - 9,000 Hz',
        recommendation: 'Dispatch perimeter security patrol to Zone 3'
    },
    'machinery': {
        name: 'Machinery Fault',
        severity: 'High',
        sevClass: 'high',
        pyConf: 93.6,
        gtmConf: 91.8,
        snr: '26.4 dB',
        quality: 'Good',
        margin: '82.1%',
        freq: 'Bearing harmonic spikes: 120 Hz - 1,800 Hz',
        recommendation: 'Schedule mechanical preventive inspection on Turbine #4'
    },
    'alarm': {
        name: 'Alarm / Siren',
        severity: 'High',
        sevClass: 'high',
        pyConf: 97.2,
        gtmConf: 98.0,
        snr: '32.1 dB',
        quality: 'Good',
        margin: '95.1%',
        freq: 'Continuous modulated sine: 800 Hz - 1,600 Hz sweep',
        recommendation: 'Verify facility emergency suppression protocol'
    },
    'scream': {
        name: 'Panic Scream',
        severity: 'Critical',
        sevClass: 'critical',
        pyConf: 96.9,
        gtmConf: 95.3,
        snr: '28.8 dB',
        quality: 'Good',
        margin: '91.4%',
        freq: 'High-pitch vocal rough formant: 1,200 Hz - 4,500 Hz',
        recommendation: 'Alert safety floor wardens to Sector 2'
    },
    'aggression': {
        name: 'Aggression',
        severity: 'High',
        sevClass: 'high',
        pyConf: 94.7,
        gtmConf: 92.4,
        snr: '25.3 dB',
        quality: 'Good',
        margin: '84.8%',
        freq: 'Aggressive vocal bursts and physical impact: 250 Hz - 3,200 Hz',
        recommendation: 'Security personnel intervention required at monitor station'
    },
    'help': {
        name: 'Call for Help',
        severity: 'Critical',
        sevClass: 'critical',
        pyConf: 95.5,
        gtmConf: 93.8,
        snr: '27.1 dB',
        quality: 'Good',
        margin: '88.3%',
        freq: 'Distress formant speech recognition ("Help me", "Emergency"): 300 Hz - 3,400 Hz',
        recommendation: 'Safety floor warden immediate dispatch for welfare check'
    },
    'animal': {
        name: 'Animal Sound',
        severity: 'Low',
        sevClass: 'low',
        pyConf: 91.2,
        gtmConf: 89.5,
        snr: '24.0 dB',
        quality: 'Good',
        margin: '79.6%',
        freq: 'Barking or wildlife vocalization: 350 Hz - 2,800 Hz',
        recommendation: 'Perimeter monitoring log recorded. No emergency dispatch.'
    },
    'horn': {
        name: 'Vehicle Horn',
        severity: 'Medium',
        sevClass: 'medium',
        pyConf: 92.8,
        gtmConf: 91.0,
        snr: '29.4 dB',
        quality: 'Good',
        margin: '81.2%',
        freq: 'Dual-tone harmonic klaxon: 400 Hz - 1,200 Hz',
        recommendation: 'Traffic sound logged. Environmental baseline updated.'
    },
    'noise': {
        name: 'Background Noise',
        severity: 'Informational',
        sevClass: 'low',
        pyConf: 99.1,
        gtmConf: 98.8,
        snr: '18.2 dB',
        quality: 'Acceptable',
        margin: '97.6%',
        freq: 'Diffuse ambient ventilation: 40 Hz - 300 Hz pink noise',
        recommendation: 'Normal ambient operation, baseline recalibrated'
    }
};

function initInteractiveStudio() {
    const waveCanvas = document.getElementById('studioWaveformCanvas');
    const specCanvas = document.getElementById('studioSpectrogramCanvas');
    const micToggleBox = document.getElementById('studioMicToggleBox');
    const presetBtns = document.querySelectorAll('.preset-chip-btn');
    const btnModePreset = document.getElementById('btnModePreset');
    const btnModeLive = document.getElementById('btnModeLive');
    const inlineFileInput = document.getElementById('studioInlineFileInput');

    if (!waveCanvas) return;

    // Start synthetic idle oscilloscope
    runIdleOscilloscope(waveCanvas, specCanvas);

    // Preset buttons click
    presetBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            presetBtns.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            if (btnModePreset) {
                btnModePreset.classList.add('active');
                if (btnModeLive) btnModeLive.classList.remove('active');
            }
            if (isStudioLive) stopRealMicrophone(waveCanvas, specCanvas);
            const soundKey = btn.getAttribute('data-sound');
            loadStudioPreset(soundKey, waveCanvas, specCanvas);
        });
    });

    // Mode Buttons
    if (btnModePreset) {
        btnModePreset.addEventListener('click', () => {
            btnModePreset.classList.add('active');
            if (btnModeLive) btnModeLive.classList.remove('active');
            if (isStudioLive) stopRealMicrophone(waveCanvas, specCanvas);
        });
    }

    if (btnModeLive) {
        btnModeLive.addEventListener('click', async () => {
            btnModeLive.classList.add('active');
            if (btnModePreset) btnModePreset.classList.remove('active');
            if (!isStudioLive) {
                await startRealMicrophone(waveCanvas, specCanvas);
            }
        });
    }

    // Real Browser Mic Click
    if (micToggleBox) {
        micToggleBox.addEventListener('click', async () => {
            if (!isStudioLive) {
                await startRealMicrophone(waveCanvas, specCanvas);
                if (btnModeLive) {
                    btnModeLive.classList.add('active');
                    if (btnModePreset) btnModePreset.classList.remove('active');
                }
            } else {
                stopRealMicrophone(waveCanvas, specCanvas);
                if (btnModePreset) {
                    btnModePreset.classList.add('active');
                    if (btnModeLive) btnModeLive.classList.remove('active');
                }
            }
        });
    }

    // Global file input listener for inline uploads
    if (inlineFileInput) {
        window.handleStudioFileUpload = function(files) {
            handleUploadedAudioFile(files, waveCanvas, specCanvas);
        };
    }
}

async function loadStudioPreset(key, waveCanvas, specCanvas) {
    const fallbackData = PRESET_DATA[key] || PRESET_DATA['gunshot'];

    // Play real audio sample from dataset
    playRealAudioSample(key);

    // Immediate UI feedback
    const decisionName = document.getElementById('studioDecisionName');
    const decisionMatch = document.getElementById('studioMatchStatus');
    const pyFill = document.getElementById('pyModelBarFill');
    const pyVal = document.getElementById('pyModelConfidence');
    const gtmFill = document.getElementById('gtmModelBarFill');
    const gtmVal = document.getElementById('gtmModelConfidence');
    const diffVal = document.getElementById('confDeltaFormula');
    const marginVal = document.getElementById('topTwoMarginFormula');
    const snrVal = document.getElementById('studioSnrVal');
    const gradeVal = document.getElementById('studioGradeVal');

    if (decisionName) decisionName.textContent = 'Analyzing with Model...';
    if (decisionMatch) decisionMatch.innerHTML = '<i class="fa-solid fa-circle-notch fa-spin"></i> Extracting 373 Acoustic Features...';
    animateBurst(waveCanvas, specCanvas, key);

    try {
        const response = await fetch(`/api/audio/classify-sample/${key}`);
        if (response.ok) {
            const data = await response.json();
            const pred = data.prediction;
            const quality = data.quality || {};

            const pyConfPct = (pred.python_confidence * 100).toFixed(1);
            const gtmConfPct = (pred.gtm_confidence * 100).toFixed(1);
            const diffPct = (pred.confidence_difference * 100).toFixed(1);
            const marginPct = (pred.top_two_margin * 100).toFixed(1);

            if (decisionName) decisionName.textContent = pred.final_class;
            if (decisionMatch) {
                decisionMatch.innerHTML = `<i class="fa-solid fa-circle-check"></i> ${pred.consistency_status} (&Delta; = ${diffPct}%)`;
            }
            if (pyFill) pyFill.style.width = `${pyConfPct}%`;
            if (pyVal) pyVal.textContent = `${pyConfPct}%`;
            if (gtmFill) gtmFill.style.width = `${gtmConfPct}%`;
            if (gtmVal) gtmVal.textContent = `${gtmConfPct}%`;
            if (diffVal) diffVal.textContent = `${diffPct}%`;
            if (marginVal) marginVal.textContent = `${marginPct}%`;
            if (snrVal) snrVal.textContent = `${quality.snr_db || 28.5} dB`;
            if (gradeVal) gradeVal.textContent = quality.quality_grade || 'Good';

            // Trigger Threat Alert HUD if hazard category detected
            if (['Gunshot', 'Panic Scream', 'Aggression', 'Glass Breaking', 'Alarm / Siren'].includes(pred.final_class)) {
                showThreatAlertHud(
                    pred.final_class,
                    `${pred.final_class.toUpperCase()} DETECTED • ${pyConfPct}% CONFIDENCE`,
                    `Automatic Alert: Sector Alpha Security Unit alerted. Incident Certificate #SS-${Date.now().toString().slice(-4)} generated.`
                );
            } else {
                dismissThreatAlertHud();
            }

            // Prepend detection to incident feed table
            appendIncidentToFeed(pred, quality, data.sample_filename || key);
            return;
        }
    } catch (e) {
        console.warn('Real model API network check, applying fallback:', e);
    }

    // Fallback if network interrupted
    if (decisionName) decisionName.textContent = fallbackData.name;
    if (decisionMatch) {
        const delta = Math.abs(fallbackData.pyConf - fallbackData.gtmConf).toFixed(1);
        decisionMatch.innerHTML = `<i class="fa-solid fa-circle-check"></i> Acceptable Match (&Delta; = ${delta}%)`;
    }
    if (pyFill) pyFill.style.width = `${fallbackData.pyConf}%`;
    if (pyVal) pyVal.textContent = `${fallbackData.pyConf}%`;
    if (gtmFill) gtmFill.style.width = `${fallbackData.gtmConf}%`;
    if (gtmVal) gtmVal.textContent = `${fallbackData.gtmConf}%`;
    if (diffVal) diffVal.textContent = `${Math.abs(fallbackData.pyConf - fallbackData.gtmConf).toFixed(1)}%`;
    if (marginVal) marginVal.textContent = fallbackData.margin;
    if (snrVal) snrVal.textContent = fallbackData.snr;
    if (gradeVal) gradeVal.textContent = fallbackData.quality;

    if (fallbackData.sevClass === 'critical' || fallbackData.sevClass === 'high') {
        showThreatAlertHud(
            fallbackData.name,
            `${fallbackData.name.toUpperCase()} DETECTED • ${fallbackData.pyConf}% CONFIDENCE`,
            `Automatic Dispatch: ${fallbackData.recommendation}. Incident Certificate generated.`
        );
    } else {
        dismissThreatAlertHud();
    }
}

/* ==========================================================================
   THREAT ALERT HUD CONTROLLER
   ========================================================================== */
function showThreatAlertHud(name, headline, detail) {
    const hud = document.getElementById('acousticThreatAlertHud');
    const headEl = document.getElementById('hudThreatHeadline');
    const detailEl = document.getElementById('hudThreatDetail');
    if (!hud) return;

    if (headEl) headEl.textContent = headline;
    if (detailEl) detailEl.textContent = detail;
    hud.classList.add('active');

    // Auto-dismiss after 9 seconds if not interacted
    clearTimeout(window._hudDismissTimer);
    window._hudDismissTimer = setTimeout(() => {
        dismissThreatAlertHud();
    }, 9000);
}

function dismissThreatAlertHud() {
    const hud = document.getElementById('acousticThreatAlertHud');
    if (hud) hud.classList.remove('active');
    clearTimeout(window._hudDismissTimer);
}
window.dismissThreatAlertHud = dismissThreatAlertHud;

/* ==========================================================================
   HERO INTERACTIVE AUDITION TRIGGER
   ========================================================================== */
function triggerHeroAudition(key, name, conf, dispatchAdvice) {
    playRealAudioSample(key);

    const statusLabel = document.getElementById('heroAudioStatus');
    const waveStrip = document.getElementById('heroSoundwaveStrip');
    if (statusLabel) {
        statusLabel.innerHTML = `<span style="color:#EF4444; font-weight:800;">[ACTIVE THREAT]</span> ${name} &bull; ${conf} Confidence`;
    }
    if (waveStrip) {
        waveStrip.classList.add('active-surge');
        setTimeout(() => waveStrip.classList.remove('active-surge'), 3000);
    }

    // Trigger HUD
    showThreatAlertHud(
        name,
        `${name.toUpperCase()} DETECTED &bull; ${conf} CONFIDENCE`,
        `Automatic Dispatch: ${dispatchAdvice}. Real-time DSP Spectrogram synchronized.`
    );

    // Also trigger in studio if canvases are available
    const wCanvas = document.getElementById('studioWaveformCanvas');
    const sCanvas = document.getElementById('studioSpectrogramCanvas');
    if (wCanvas && sCanvas) {
        animateBurst(wCanvas, sCanvas, key);
        loadStudioPreset(key, wCanvas, sCanvas);
    }
}
window.triggerHeroAudition = triggerHeroAudition;

function playRealAudioSample(categoryKey) {
    try {
        if (activeAudioElement) {
            activeAudioElement.pause();
            activeAudioElement.currentTime = 0;
        }
        activeAudioElement = new Audio(`/api/audio/sample/${categoryKey}`);
        activeAudioElement.volume = 0.7;
        activeAudioElement.play().catch(err => {
            synthesizeAcousticSample(categoryKey);
        });
    } catch (e) {
        synthesizeAcousticSample(categoryKey);
    }
}

// Synthesize pleasant realistic signature tones using Web Audio API
function synthesizeAcousticSample(type) {
    try {
        const ctx = getAudioContext();
        if (ctx.state === 'suspended') ctx.resume();

        const osc = ctx.createOscillator();
        const gain = ctx.createGain();
        osc.connect(gain);
        gain.connect(ctx.destination);

        const now = ctx.currentTime;

        if (type === 'gunshot') {
            osc.type = 'sawtooth';
            osc.frequency.setValueAtTime(400, now);
            osc.frequency.exponentialRampToValueAtTime(60, now + 0.25);
            gain.gain.setValueAtTime(0.8, now);
            gain.gain.exponentialRampToValueAtTime(0.001, now + 0.3);
            osc.start(now);
            osc.stop(now + 0.3);
        } else if (type === 'glass') {
            osc.type = 'triangle';
            osc.frequency.setValueAtTime(3200, now);
            osc.frequency.exponentialRampToValueAtTime(6500, now + 0.15);
            gain.gain.setValueAtTime(0.3, now);
            gain.gain.exponentialRampToValueAtTime(0.001, now + 0.35);
            osc.start(now);
            osc.stop(now + 0.35);
        } else if (type === 'alarm') {
            osc.type = 'sine';
            osc.frequency.setValueAtTime(900, now);
            osc.frequency.linearRampToValueAtTime(1400, now + 0.2);
            osc.frequency.linearRampToValueAtTime(900, now + 0.4);
            gain.gain.setValueAtTime(0.35, now);
            gain.gain.linearRampToValueAtTime(0.01, now + 0.5);
            osc.start(now);
            osc.stop(now + 0.5);
        } else if (type === 'machinery') {
            osc.type = 'sawtooth';
            osc.frequency.setValueAtTime(120, now);
            gain.gain.setValueAtTime(0.4, now);
            gain.gain.exponentialRampToValueAtTime(0.01, now + 0.6);
            osc.start(now);
            osc.stop(now + 0.6);
        } else if (type === 'scream') {
            osc.type = 'sine';
            osc.frequency.setValueAtTime(1200, now);
            osc.frequency.linearRampToValueAtTime(2200, now + 0.3);
            gain.gain.setValueAtTime(0.35, now);
            gain.gain.exponentialRampToValueAtTime(0.01, now + 0.45);
            osc.start(now);
            osc.stop(now + 0.45);
        } else {
            osc.type = 'sine';
            osc.frequency.setValueAtTime(80, now);
            gain.gain.setValueAtTime(0.15, now);
            gain.gain.exponentialRampToValueAtTime(0.001, now + 0.4);
            osc.start(now);
            osc.stop(now + 0.4);
        }
    } catch (e) {
        console.warn('Web Audio synthesis not supported or prevented:', e);
    }
}

function playTone(freq, duration, type = 'sine') {
    try {
        const ctx = getAudioContext();
        if (ctx.state === 'suspended') ctx.resume();
        const osc = ctx.createOscillator();
        const gain = ctx.createGain();
        osc.type = type;
        osc.frequency.value = freq;
        gain.gain.setValueAtTime(0.2, ctx.currentTime);
        gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + duration);
        osc.connect(gain);
        gain.connect(ctx.destination);
        osc.start();
        osc.stop(ctx.currentTime + duration);
    } catch (e) {}
}

function getAudioContext() {
    if (!audioCtx) {
        audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    }
    return audioCtx;
}

/* Oscilloscope Renderers */
function runIdleOscilloscope(wCanvas, sCanvas) {
    if (!wCanvas || !sCanvas) return;
    const wCtx = wCanvas.getContext('2d');
    const sCtx = sCanvas.getContext('2d');
    let phase = 0;

    function renderIdle() {
        if (isStudioLive) return; // real mic taking over

        const w = wCanvas.width = wCanvas.clientWidth;
        const h = wCanvas.height = wCanvas.clientHeight;
        wCtx.clearRect(0, 0, w, h);

        // Draw waveform grid
        wCtx.strokeStyle = 'rgba(45, 212, 191, 0.1)';
        wCtx.lineWidth = 1;
        wCtx.beginPath();
        wCtx.moveTo(0, h / 2);
        wCtx.lineTo(w, h / 2);
        wCtx.stroke();

        // Waveform trace
        wCtx.beginPath();
        wCtx.lineWidth = 2;
        wCtx.strokeStyle = '#2DD4BF';
        for (let x = 0; x < w; x++) {
            const y = (h / 2) + Math.sin(x * 0.04 + phase) * 18 * Math.cos(x * 0.01 - phase * 0.5) +
                      Math.sin(x * 0.1 + phase * 2) * 6;
            if (x === 0) wCtx.moveTo(x, y);
            else wCtx.lineTo(x, y);
        }
        wCtx.stroke();

        // Spectrogram heat-map bars
        const sw = sCanvas.width = sCanvas.clientWidth;
        const sh = sCanvas.height = sCanvas.clientHeight;
        sCtx.clearRect(0, 0, sw, sh);

        const barCount = 48;
        const barWidth = sw / barCount;
        for (let i = 0; i < barCount; i++) {
            const mag = Math.abs(Math.sin(i * 0.18 + phase * 0.8)) * 0.75 + 
                        Math.cos(i * 0.08 - phase) * 0.25;
            const barH = mag * (sh * 0.85);
            const r = Math.floor(13 + mag * 32);
            const g = Math.floor(148 + mag * 90);
            const b = Math.floor(136 + mag * 70);
            sCtx.fillStyle = `rgb(${r}, ${g}, ${b})`;
            sCtx.fillRect(i * barWidth, sh - barH, barWidth - 1.5, barH);
        }

        phase += 0.04;
        animWaveId = requestAnimationFrame(renderIdle);
    }
    renderIdle();
}

function animateBurst(wCanvas, sCanvas, key) {
    if (!wCanvas || !sCanvas) return;
    const wCtx = wCanvas.getContext('2d');
    const sCtx = sCanvas.getContext('2d');
    let frame = 0;
    const maxFrames = 30;

    function renderBurst() {
        if (frame > maxFrames || isStudioLive) return;

        const w = wCanvas.width = wCanvas.clientWidth;
        const h = wCanvas.height = wCanvas.clientHeight;
        wCtx.clearRect(0, 0, w, h);

        const intensity = 1 - (frame / maxFrames);
        wCtx.beginPath();
        wCtx.lineWidth = 2.5;
        wCtx.strokeStyle = key === 'gunshot' || key === 'scream' ? '#EF4444' : '#2DD4BF';

        for (let x = 0; x < w; x++) {
            const envelope = Math.exp(-Math.pow((x - w * 0.4) / (w * 0.18), 2));
            const y = (h / 2) + Math.sin(x * 0.15) * 36 * envelope * intensity * (Math.random() * 0.4 + 0.8);
            if (x === 0) wCtx.moveTo(x, y);
            else wCtx.lineTo(x, y);
        }
        wCtx.stroke();

        const sw = sCanvas.width = sCanvas.clientWidth;
        const sh = sCanvas.height = sCanvas.clientHeight;
        sCtx.clearRect(0, 0, sw, sh);

        const barCount = 48;
        const barWidth = sw / barCount;
        for (let i = 0; i < barCount; i++) {
            const barH = (Math.random() * 0.7 + 0.3) * (sh * 0.9) * intensity;
            sCtx.fillStyle = key === 'gunshot' || key === 'scream' ? '#F87171' : '#14B8A6';
            sCtx.fillRect(i * barWidth, sh - barH, barWidth - 1.5, barH);
        }

        frame++;
        requestAnimationFrame(renderBurst);
    }
    renderBurst();
}

/* Real Browser Microphone Recording & Live Model Streaming */
async function startRealMicrophone(wCanvas, sCanvas) {
    try {
        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
            alert('Microphone access is not supported in this browser.');
            return;
        }

        const stream = await navigator.mediaDevices.getUserMedia({ audio: true, video: false });
        micStream = stream;
        isStudioLive = true;

        const ctx = getAudioContext();
        if (ctx.state === 'suspended') await ctx.resume();

        const source = ctx.createMediaStreamSource(stream);
        analyser = ctx.createAnalyser();
        analyser.fftSize = 256;
        source.connect(analyser);

        const toggleBox = document.getElementById('studioMicToggleBox');
        if (toggleBox) {
            toggleBox.style.borderColor = '#10B981';
            const titleEl = document.getElementById('studioMicTitle');
            const subEl = document.getElementById('studioMicSub');
            if (titleEl) titleEl.textContent = 'Live Microphone: Active (Streaming to AI)';
            if (subEl) subEl.textContent = 'Speaking / making sound will trigger live model detection';
        }

        drawLiveMicrophoneWave(wCanvas, sCanvas);

        // Stream live slices to /api/audio/live-chunk every 2 seconds
        liveMicPollTimer = setInterval(() => {
            sendLiveMicSlice();
        }, 2000);

    } catch (err) {
        alert('Microphone access was denied or is not supported: ' + err.message);
    }
}

function stopRealMicrophone(wCanvas, sCanvas) {
    if (micStream) {
        micStream.getTracks().forEach(t => t.stop());
        micStream = null;
    }
    if (liveMicPollTimer) {
        clearInterval(liveMicPollTimer);
        liveMicPollTimer = null;
    }
    isStudioLive = false;
    const toggleBox = document.getElementById('studioMicToggleBox');
    if (toggleBox) {
        toggleBox.style.borderColor = 'var(--teal-400)';
        const titleEl = document.getElementById('studioMicTitle');
        const subEl = document.getElementById('studioMicSub');
        if (titleEl) titleEl.textContent = 'Capture Live Microphone';
        if (subEl) subEl.textContent = 'Click to enable real-time browser audio stream & live AI scoring';
    }
    runIdleOscilloscope(wCanvas, sCanvas);
}

function sendLiveMicSlice() {
    if (!isStudioLive || !analyser) return;
    const pcmData = new Float32Array(1024);
    analyser.getFloatTimeDomainData(pcmData);

    // Compute basic RMS
    let sumSquares = 0;
    for (let i = 0; i < pcmData.length; i++) {
        sumSquares += pcmData[i] * pcmData[i];
    }
    const rms = Math.sqrt(sumSquares / pcmData.length);
    if (rms < 0.003) {
        // Very low energy room noise
        return;
    }

    // Send binary PCM buffer to live chunk endpoint
    fetch('/api/audio/live-chunk', {
        method: 'POST',
        headers: { 'Content-Type': 'application/octet-stream' },
        body: pcmData.buffer
    })
    .then(r => r.json())
    .then(data => {
        if (data && data.prediction) {
            const pred = data.prediction;
            const quality = data.quality || {};
            const decisionName = document.getElementById('studioDecisionName');
            const decisionMatch = document.getElementById('studioMatchStatus');
            const pyFill = document.getElementById('pyModelBarFill');
            const pyVal = document.getElementById('pyModelConfidence');
            const gtmFill = document.getElementById('gtmModelBarFill');
            const gtmVal = document.getElementById('gtmModelConfidence');

            if (decisionName && pred.final_class) decisionName.textContent = pred.final_class;
            if (decisionMatch && pred.consistency_status) {
                const diffPct = (pred.confidence_difference * 100).toFixed(1);
                decisionMatch.innerHTML = `<i class="fa-solid fa-tower-broadcast"></i> ${pred.consistency_status} (&Delta; = ${diffPct}%)`;
            }
            if (pyFill && pred.python_confidence) {
                const pct = (pred.python_confidence * 100).toFixed(1);
                pyFill.style.width = `${pct}%`;
                if (pyVal) pyVal.textContent = `${pct}%`;
            }
            if (gtmFill && pred.gtm_confidence) {
                const pct = (pred.gtm_confidence * 100).toFixed(1);
                gtmFill.style.width = `${pct}%`;
                if (gtmVal) gtmVal.textContent = `${pct}%`;
            }

            if (pred.severity === 'Critical' || pred.severity === 'High') {
                appendIncidentToFeed(pred, quality, 'Live Mic Sensor');
            }
        }
    })
    .catch(err => console.warn('Live mic stream ping:', err));
}

function drawLiveMicrophoneWave(wCanvas, sCanvas) {
    if (!isStudioLive || !analyser) return;

    const wCtx = wCanvas.getContext('2d');
    const sCtx = sCanvas.getContext('2d');
    const bufferLength = analyser.frequencyBinCount;
    const timeData = new Uint8Array(bufferLength);
    const freqData = new Uint8Array(bufferLength);

    function renderMic() {
        if (!isStudioLive) return;

        analyser.getByteTimeDomainData(timeData);
        analyser.getByteFrequencyData(freqData);

        const w = wCanvas.width = wCanvas.clientWidth;
        const h = wCanvas.height = wCanvas.clientHeight;
        wCtx.clearRect(0, 0, w, h);

        wCtx.lineWidth = 2.2;
        wCtx.strokeStyle = '#2DD4BF';
        wCtx.beginPath();

        const sliceWidth = w / bufferLength;
        let x = 0;
        for (let i = 0; i < bufferLength; i++) {
            const v = timeData[i] / 128.0;
            const y = (v * h) / 2;
            if (i === 0) wCtx.moveTo(x, y);
            else wCtx.lineTo(x, y);
            x += sliceWidth;
        }
        wCtx.stroke();

        // Spectrogram bars
        const sw = sCanvas.width = sCanvas.clientWidth;
        const sh = sCanvas.height = sCanvas.clientHeight;
        sCtx.clearRect(0, 0, sw, sh);

        const barWidth = sw / bufferLength;
        for (let i = 0; i < bufferLength; i++) {
            const barHeight = (freqData[i] / 255) * sh;
            sCtx.fillStyle = '#14B8A6';
            sCtx.fillRect(i * barWidth, sh - barHeight, barWidth - 1, barHeight);
        }

        requestAnimationFrame(renderMic);
    }
    renderMic();
}

/* Direct File Ingestion from Sandbox */
async function handleUploadedAudioFile(files, wCanvas, sCanvas) {
    if (!files || files.length === 0) return;
    const file = files[0];

    const decisionName = document.getElementById('studioDecisionName');
    const decisionMatch = document.getElementById('studioMatchStatus');

    if (decisionName) decisionName.textContent = 'Extracting 373 Acoustic Features...';
    if (decisionMatch) decisionMatch.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Running Trained SVM & GTM Pipeline...';

    animateBurst(wCanvas, sCanvas, 'gunshot');

    const formData = new FormData();
    formData.append('audio', file);

    try {
        const resp = await fetch('/api/audio/upload', {
            method: 'POST',
            body: formData
        });
        const data = await resp.json();

        if (data.success) {
            const pred = data.prediction;
            const quality = data.quality || {};

            const pyConfPct = (pred.python_confidence * 100).toFixed(1);
            const gtmConfPct = (pred.gtm_confidence * 100).toFixed(1);
            const diffPct = (pred.confidence_difference * 100).toFixed(1);
            const marginPct = (pred.top_two_margin * 100).toFixed(1);

            if (decisionName) decisionName.textContent = pred.final_class;
            if (decisionMatch) {
                decisionMatch.innerHTML = `<i class="fa-solid fa-circle-check"></i> ${pred.consistency_status} (&Delta; = ${diffPct}%)`;
            }

            const pyFill = document.getElementById('pyModelBarFill');
            const pyVal = document.getElementById('pyModelConfidence');
            const gtmFill = document.getElementById('gtmModelBarFill');
            const gtmVal = document.getElementById('gtmModelConfidence');
            const diffVal = document.getElementById('confDeltaFormula');
            const marginVal = document.getElementById('topTwoMarginFormula');
            const snrVal = document.getElementById('studioSnrVal');
            const gradeVal = document.getElementById('studioGradeVal');

            if (pyFill) pyFill.style.width = `${pyConfPct}%`;
            if (pyVal) pyVal.textContent = `${pyConfPct}%`;
            if (gtmFill) gtmFill.style.width = `${gtmConfPct}%`;
            if (gtmVal) gtmVal.textContent = `${gtmConfPct}%`;
            if (diffVal) diffVal.textContent = `${diffPct}%`;
            if (marginVal) marginVal.textContent = `${marginPct}%`;
            if (snrVal) snrVal.textContent = `${quality.snr_db || 31.4} dB`;
            if (gradeVal) gradeVal.textContent = quality.quality_grade || 'Good';

            appendIncidentToFeed(pred, quality, file.name);

            // Play preview
            const fileUrl = URL.createObjectURL(file);
            if (activeAudioElement) activeAudioElement.pause();
            activeAudioElement = new Audio(fileUrl);
            activeAudioElement.play().catch(() => {});
        } else {
            alert('Upload classification error: ' + (data.error || 'Failed to process file'));
        }
    } catch (err) {
        console.error('File upload failed:', err);
    }
}

/* ==========================================================================
   4. 10 SOUND CATEGORIES AUDITION & MODEL PREDICTION
   ========================================================================== */
function initCategoryAudition() {
    const cards = document.querySelectorAll('.sound-category-item-card');
    const waveCanvas = document.getElementById('studioWaveformCanvas');
    const specCanvas = document.getElementById('studioSpectrogramCanvas');

    cards.forEach(card => {
        card.addEventListener('click', () => {
            const key = card.getAttribute('data-category-key') || 'gunshot';

            // Active visual feedback
            cards.forEach(c => c.classList.remove('active-playing'));
            card.classList.add('active-playing');
            setTimeout(() => card.classList.remove('active-playing'), 2500);

            // Sync with preset button if exists
            const matchingPresetBtn = document.querySelector(`.preset-chip-btn[data-sound="${key}"]`);
            if (matchingPresetBtn) {
                document.querySelectorAll('.preset-chip-btn').forEach(b => b.classList.remove('active'));
                matchingPresetBtn.classList.add('active');
            }

            // Run real fast API classification and audio playback
            loadStudioPreset(key, waveCanvas, specCanvas);

            // Pop animation on card
            card.style.transform = 'scale(1.03)';
            setTimeout(() => { card.style.transform = ''; }, 200);
        });
    });
}

/* ==========================================================================
   5. LIVE INCIDENT FEED LOADER & DYNAMIC INSERTION
   ========================================================================== */
async function loadIncidentFeed() {
    try {
        const res = await fetch('/api/audio/events');
        if (!res.ok) return;
        const events = await res.json();
        if (!events || events.length === 0) return;

        const tbody = document.getElementById('incidentFeedTbody');
        if (!tbody) return;

        tbody.innerHTML = '';
        const countLabel = document.getElementById('incidentCountLabel');
        if (countLabel) countLabel.textContent = `Showing ${Math.min(events.length, 6)} latest verified acoustic detections`;

        events.slice(0, 6).forEach(ev => {
            const tr = document.createElement('tr');
            const sevClass = (ev.severity || 'Medium').toLowerCase();
            const detectedClass = ev.final_detected_class || ev.final_class || ev.predicted_class || 'Sound Event';
            const pyVal = ev.python_top_confidence != null ? ev.python_top_confidence : ev.python_confidence;
            const gtmVal = ev.gtm_top_confidence != null ? ev.gtm_top_confidence : ev.gtm_confidence;
            const pyConf = pyVal != null ? (pyVal * 100).toFixed(1) + '%' : '94.5%';
            const gtmConf = gtmVal != null ? (gtmVal * 100).toFixed(1) + '%' : '93.0%';

            tr.innerHTML = `
                <td style="font-family: var(--font-mono); font-weight: 700;">${ev.id || 'AUD-001'}</td>
                <td><i class="fa-solid fa-volume-high" style="color: var(--teal-400); margin-right: 6px;"></i> ${detectedClass}</td>
                <td><span class="badge-sev ${sevClass}">${ev.severity || 'Medium'}</span></td>
                <td>${pyConf}</td>
                <td>${gtmConf}</td>
                <td><span style="color:#10B981; font-weight:600;">${ev.quality_grade || 'Good'}</span></td>
                <td><span style="color:var(--text-muted); font-weight:600;">${ev.alert_status || 'Logged'}</span></td>
            `;
            tbody.appendChild(tr);
        });
    } catch (e) {
        console.warn('Could not load initial incident feed:', e);
    }
}

function appendIncidentToFeed(pred, quality, filename) {
    const tbody = document.getElementById('incidentFeedTbody');
    if (!tbody) return;
    const audioId = 'AUD-' + Math.random().toString(16).substring(2, 8).toUpperCase();
    const tr = document.createElement('tr');
    tr.style.background = 'rgba(45, 212, 191, 0.08)';

    const sevClass = (pred.severity || 'Medium').toLowerCase();
    const pyConf = (pred.python_confidence * 100).toFixed(1) + '%';
    const gtmConf = (pred.gtm_confidence * 100).toFixed(1) + '%';
    const qGrade = quality.quality_grade || 'Good';

    tr.innerHTML = `
        <td style="font-family: var(--font-mono); font-weight: 700;">${audioId}</td>
        <td><i class="fa-solid fa-waveform" style="color: var(--teal-400); margin-right: 6px;"></i> ${pred.final_class}</td>
        <td><span class="badge-sev ${sevClass}">${pred.severity || 'Medium'}</span></td>
        <td>${pyConf}</td>
        <td>${gtmConf}</td>
        <td><span style="color:#10B981; font-weight:600;">${qGrade}</span></td>
        <td><span style="color:var(--teal-600); font-weight:700;">${pred.consistency_status}</span></td>
    `;
    tbody.insertBefore(tr, tbody.firstChild);

    // Keep max 7 rows
    while (tbody.children.length > 7) {
        tbody.removeChild(tbody.lastChild);
    }
}

/* ==========================================================================
   6. HEADER SCROLL & MOBILE MENU
   ========================================================================== */
function initHeaderScroll() {
    const header = document.getElementById('siteHeader');
    if (!header) return;
    window.addEventListener('scroll', () => {
        if (window.scrollY > 25) {
            header.classList.add('scrolled');
        } else {
            header.classList.remove('scrolled');
        }
    });
}

function initMobileNav() {
    const mobileBtn = document.getElementById('mobileMenuBtn');
    const navMenu = document.getElementById('navPillMenu');
    if (!mobileBtn || !navMenu) return;

    mobileBtn.addEventListener('click', () => {
        const isShown = navMenu.style.display === 'flex';
        navMenu.style.display = isShown ? 'none' : 'flex';
        if (!isShown) {
            navMenu.style.position = 'absolute';
            navMenu.style.top = '76px';
            navMenu.style.left = '0';
            navMenu.style.right = '0';
            navMenu.style.background = '#FFFFFF';
            navMenu.style.padding = '20px';
            navMenu.style.boxShadow = '0 10px 30px rgba(0,0,0,0.1)';
            navMenu.style.flexDirection = 'column';
        }
    });
}
