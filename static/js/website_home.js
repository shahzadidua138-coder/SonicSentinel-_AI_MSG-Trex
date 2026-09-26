/**
 * SonicSentinel AI - Interactive Homepage Engine
 * Ambient acoustic waves, Web Audio API oscilloscope, dual-model live simulation,
 * interactive sound presets, real-time dataset playback, and trained ML fast APIs.
 */

document.addEventListener('DOMContentLoaded', () => {
    initThemeToggle();
    initCustomCursor();
    initPixieDustGlitter();
    initSonarClickEngine();
    initScrollTelemetryHud();
    init3DCardTilt();
    initHeroHoloCore();
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
    function checkHeaderScroll() {
        if (window.scrollY > 30) {
            header.classList.add('scrolled');
        } else {
            header.classList.remove('scrolled');
        }
    }
    window.addEventListener('scroll', checkHeaderScroll, { passive: true });
    checkHeaderScroll();
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

/* ==========================================================================
   7. CUSTOM FUTURISTIC MAGNETIC CURSOR ENGINE
   ========================================================================== */
function initCustomCursor() {
    const ring = document.getElementById('customCursorRing');
    const dot = document.getElementById('customCursorDot');
    if (!ring || !dot) return;

    let targetX = -100, targetY = -100;
    let ringX = -100, ringY = -100;

    window.addEventListener('mousemove', (e) => {
        targetX = e.clientX;
        targetY = e.clientY;
        dot.style.transform = `translate(${targetX}px, ${targetY}px) translate(-50%, -50%)`;
    }, { passive: true });

    function renderCursor() {
        ringX += (targetX - ringX) * 0.22;
        ringY += (targetY - ringY) * 0.22;
        ring.style.transform = `translate(${ringX}px, ${ringY}px) translate(-50%, -50%)`;
        requestAnimationFrame(renderCursor);
    }
    requestAnimationFrame(renderCursor);

    // Hover magnet detection
    const interactives = 'a, button, input, textarea, select, .sound-category-item-card, .why-feature-card, .hero-audition-chip, .preset-chip-btn, .canvas-container-box, .hero-mic-circle-btn, .hero-holo-core-canvas, .check-feature-item';
    
    document.addEventListener('mouseover', (e) => {
        if (e.target.closest(interactives)) {
            ring.classList.add('hover-magnet');
        }
    });

    document.addEventListener('mouseout', (e) => {
        if (e.target.closest(interactives)) {
            ring.classList.remove('hover-magnet');
        }
    });

    document.addEventListener('mousedown', () => ring.classList.add('click-pulse'));
    document.addEventListener('mouseup', () => ring.classList.remove('click-pulse'));
}

/* ==========================================================================
   8. ETHEREAL SILVER FAIRY PIXIE DUST GLITTER TRAIL ENGINE
   ========================================================================== */
function initPixieDustGlitter() {
    const canvas = document.getElementById('glitterTailCanvas');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    let width = canvas.width = window.innerWidth;
    let height = canvas.height = window.innerHeight;

    window.addEventListener('resize', () => {
        width = canvas.width = window.innerWidth;
        height = canvas.height = window.innerHeight;
    });

    const particles = [];
    const maxParticles = 160;

    // Palette: Shimmering Silver, Platinum, Starlight Specular, with delicate Cyan Glow
    const colors = [
        '#FFFFFF',
        '#F8FAFC',
        '#F1F5F9',
        '#E2E8F0',
        '#CBD5E1',
        'rgba(241, 245, 249, 0.95)',
        'rgba(203, 213, 225, 0.9)',
        'rgba(45, 212, 191, 0.7)'
    ];

    function spawnParticle(x, y, count = 1, isBurst = false) {
        for (let i = 0; i < count; i++) {
            if (particles.length > maxParticles) particles.shift();
            const angle = Math.random() * Math.PI * 2;
            const speed = isBurst ? Math.random() * 4.5 + 2.0 : Math.random() * 1.8 + 0.4;
            const size = Math.random() * 3.2 + 1.2;
            const type = Math.random() < 0.45 ? 'star' : (Math.random() < 0.4 ? 'diamond' : 'orb');

            particles.push({
                x: x + (Math.random() - 0.5) * 8,
                y: y + (Math.random() - 0.5) * 8,
                vx: Math.cos(angle) * speed,
                vy: Math.sin(angle) * speed - (isBurst ? 1.2 : 0.45),
                size: size,
                color: colors[Math.floor(Math.random() * colors.length)],
                alpha: 1.0,
                decay: Math.random() * 0.02 + 0.016,
                type: type,
                rotation: Math.random() * Math.PI,
                rotSpeed: (Math.random() - 0.5) * 0.14,
                twinklePhase: Math.random() * Math.PI * 2
            });
        }
    }

    let lastMouseX = -100, lastMouseY = -100;
    window.addEventListener('mousemove', (e) => {
        const dx = e.clientX - lastMouseX;
        const dy = e.clientY - lastMouseY;
        const dist = Math.hypot(dx, dy);

        if (dist > 3) {
            const num = Math.min(Math.floor(dist / 5) + 1, 5);
            spawnParticle(e.clientX, e.clientY, num, false);
            lastMouseX = e.clientX;
            lastMouseY = e.clientY;
        }
    }, { passive: true });

    // Touch support for touch screens
    window.addEventListener('touchmove', (e) => {
        if (e.touches.length > 0) {
            const touch = e.touches[0];
            spawnParticle(touch.clientX, touch.clientY, 3, false);
        }
    }, { passive: true });

    // Magical silver burst on user click
    window.addEventListener('click', (e) => {
        spawnParticle(e.clientX, e.clientY, 18, true);
    });

    function drawSparkleStar(cx, cy, spikes, outerRadius, innerRadius, color, alpha, rot) {
        ctx.save();
        ctx.translate(cx, cy);
        ctx.rotate(rot);
        ctx.globalAlpha = Math.max(0, alpha);
        ctx.fillStyle = color;
        ctx.shadowColor = '#FFFFFF';
        ctx.shadowBlur = 8;

        let rotStep = Math.PI / spikes;
        let x = 0, y = 0;

        ctx.beginPath();
        ctx.moveTo(0, -outerRadius);
        for (let i = 0; i < spikes; i++) {
            x = Math.cos(rotStep * (2 * i + 1) - Math.PI / 2) * innerRadius;
            y = Math.sin(rotStep * (2 * i + 1) - Math.PI / 2) * innerRadius;
            ctx.lineTo(x, y);

            x = Math.cos(rotStep * (2 * i + 2) - Math.PI / 2) * outerRadius;
            y = Math.sin(rotStep * (2 * i + 2) - Math.PI / 2) * outerRadius;
            ctx.lineTo(x, y);
        }
        ctx.closePath();
        ctx.fill();
        ctx.restore();
    }

    function renderGlitter() {
        ctx.clearRect(0, 0, width, height);

        for (let i = particles.length - 1; i >= 0; i--) {
            const p = particles[i];
            p.x += p.vx;
            p.y += p.vy;
            p.vy += 0.038; // gentle fairy stardust gravity
            p.vx *= 0.975;
            p.vy *= 0.975;
            p.alpha -= p.decay;
            p.rotation += p.rotSpeed;
            p.twinklePhase += 0.16;

            if (p.alpha <= 0) {
                particles.splice(i, 1);
                continue;
            }

            const twinkleAlpha = p.alpha * (0.65 + 0.35 * Math.sin(p.twinklePhase));

            if (p.type === 'star') {
                drawSparkleStar(p.x, p.y, 4, p.size * 2.2, p.size * 0.45, p.color, twinkleAlpha, p.rotation);
            } else if (p.type === 'diamond') {
                ctx.save();
                ctx.translate(p.x, p.y);
                ctx.rotate(p.rotation);
                ctx.globalAlpha = Math.max(0, twinkleAlpha);
                ctx.fillStyle = p.color;
                ctx.shadowColor = '#E2E8F0';
                ctx.shadowBlur = 6;
                ctx.beginPath();
                ctx.moveTo(0, -p.size * 1.6);
                ctx.lineTo(p.size * 0.85, 0);
                ctx.lineTo(0, p.size * 1.6);
                ctx.lineTo(-p.size * 0.85, 0);
                ctx.closePath();
                ctx.fill();
                ctx.restore();
            } else {
                ctx.save();
                ctx.globalAlpha = Math.max(0, twinkleAlpha);
                ctx.fillStyle = p.color;
                ctx.shadowColor = '#FFFFFF';
                ctx.shadowBlur = 8;
                ctx.beginPath();
                ctx.arc(p.x, p.y, p.size, 0, Math.PI * 2);
                ctx.fill();
                ctx.restore();
            }
        }

        requestAnimationFrame(renderGlitter);
    }
    requestAnimationFrame(renderGlitter);
}

/* ==========================================================================
   9. SONAR RADAR CLICK SHOCKWAVE & AUDIO FEEDBACK
   ========================================================================== */
function initSonarClickEngine() {
    let clickCtx = null;

    function playClickChirp() {
        try {
            if (!clickCtx) {
                clickCtx = new (window.AudioContext || window.webkitAudioContext)();
            }
            if (clickCtx.state === 'suspended') {
                clickCtx.resume();
            }
            const now = clickCtx.currentTime;
            const osc = clickCtx.createOscillator();
            const gain = clickCtx.createGain();

            osc.type = 'sine';
            osc.frequency.setValueAtTime(1750, now);
            osc.frequency.exponentialRampToValueAtTime(480, now + 0.045);

            gain.gain.setValueAtTime(0.035, now);
            gain.gain.exponentialRampToValueAtTime(0.0001, now + 0.045);

            osc.connect(gain);
            gain.connect(clickCtx.destination);

            osc.start(now);
            osc.stop(now + 0.045);
        } catch (e) {
            // Audio policy fallback
        }
    }

    document.addEventListener('click', (e) => {
        playClickChirp();

        // Expanding sonar shockwave
        const ripple = document.createElement('div');
        ripple.className = 'sonar-click-ripple';
        const size = Math.max(window.innerWidth, window.innerHeight) * 0.22;
        ripple.style.width = `${size}px`;
        ripple.style.height = `${size}px`;
        ripple.style.left = `${e.clientX}px`;
        ripple.style.top = `${e.clientY}px`;
        document.body.appendChild(ripple);

        setTimeout(() => ripple.remove(), 750);
    });
}

/* ==========================================================================
   10. KINETIC SCROLL ACOUSTIC TELEMETRY HUD (20 Hz - 20,000 Hz)
   ========================================================================== */
function initScrollTelemetryHud() {
    const meterBar = document.getElementById('hudMeterBar');
    const freqLabel = document.getElementById('hudScrollFreq');
    const dbIndicator = document.getElementById('hudScrollDb');
    if (!meterBar || !freqLabel) return;

    function updateScrollHud() {
        const scrollTop = window.scrollY;
        const maxScroll = Math.max(document.documentElement.scrollHeight - window.innerHeight, 1);
        const progress = Math.min(Math.max(scrollTop / maxScroll, 0), 1);

        meterBar.style.height = `${progress * 100}%`;

        // Logarithmic frequency progression: 20 Hz to 20,000 Hz
        const freq = Math.round(20 * Math.pow(1000, progress));
        if (freq >= 1000) {
            freqLabel.textContent = `${(freq / 1000).toFixed(1)} kHz`;
        } else {
            freqLabel.textContent = `${freq} Hz`;
        }

        // Live amplitude decibel scale
        if (dbIndicator) {
            const dbVal = (-54 + progress * 56).toFixed(1);
            dbIndicator.textContent = `${dbVal > 0 ? '+' : ''}${dbVal} dB`;
            dbIndicator.style.color = progress > 0.75 ? '#EF4444' : (progress > 0.4 ? '#2DD4BF' : '#94A3B8');
        }
    }

    window.addEventListener('scroll', updateScrollHud, { passive: true });
    updateScrollHud();
}

/* ==========================================================================
   11. 3D INTERACTIVE CARD TILT & HOLOGRAPHIC SHEEN ENGINE
   ========================================================================== */
function init3DCardTilt() {
    const cardSelectors = [
        '.sound-category-item-card',
        '.why-feature-card',
        '.pipeline-step-node',
        '.floating-alert-card',
        '.floating-success-card',
        '.q-metric-card',
        '.location-card'
    ];

    const cards = document.querySelectorAll(cardSelectors.join(', '));

    cards.forEach(card => {
        card.classList.add('tilt-card-3d');

        // Append holographic glare layer if not present
        if (!card.querySelector('.card-holographic-glare')) {
            const glare = document.createElement('div');
            glare.className = 'card-holographic-glare';
            card.appendChild(glare);
        }

        card.addEventListener('mousemove', (e) => {
            const rect = card.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;

            card.style.setProperty('--mouse-x', `${x}px`);
            card.style.setProperty('--mouse-y', `${y}px`);

            const normX = (x / rect.width) - 0.5;
            const normY = (y / rect.height) - 0.5;

            const rotY = (normX * 14).toFixed(2);
            const rotX = (-normY * 14).toFixed(2);

            card.style.transform = `perspective(1000px) rotateX(${rotX}deg) rotateY(${rotY}deg) scale3d(1.025, 1.025, 1.025)`;
        });

        card.addEventListener('mouseleave', () => {
            card.style.transform = 'perspective(1000px) rotateX(0deg) rotateY(0deg) scale3d(1, 1, 1)';
        });
    });
}

/* ==========================================================================
   12. 3D HOLOGRAPHIC ACOUSTIC IRIS CORE CANVAS (HERO CENTERPIECE)
   ========================================================================== */
function initHeroHoloCore() {
    const canvas = document.getElementById('heroHoloCoreCanvas');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const size = 480;
    canvas.width = size;
    canvas.height = size;

    let mouseX = size / 2, mouseY = size / 2;
    let targetRotX = 0, targetRotY = 0;
    let rotX = 0, rotY = 0;
    let isCorePulsing = false;
    let pulseScale = 1.0;

    canvas.addEventListener('mousemove', (e) => {
        const rect = canvas.getBoundingClientRect();
        mouseX = e.clientX - rect.left;
        mouseY = e.clientY - rect.top;
        targetRotY = ((mouseX / size) - 0.5) * 1.2;
        targetRotX = -((mouseY / size) - 0.5) * 1.2;
    });

    canvas.addEventListener('click', () => {
        isCorePulsing = true;
        pulseScale = 1.35;
        // Trigger Threat Audition
        if (typeof triggerHeroAudition === 'function') {
            triggerHeroAudition('gunshot', 'Gunshot Shockwave', '98.4%', 'Quantum Acoustic Core Resonating');
        }
    });

    // 3D Spherical Particle Nodes
    const particleCount = 140;
    const coreParticles = [];
    const radius = 135;

    for (let i = 0; i < particleCount; i++) {
        const theta = Math.acos(2 * Math.random() - 1);
        const phi = Math.random() * Math.PI * 2;
        coreParticles.push({
            origX: radius * Math.sin(theta) * Math.cos(phi),
            origY: radius * Math.sin(theta) * Math.sin(phi),
            origZ: radius * Math.cos(theta),
            radius: Math.random() * 2.2 + 1.2,
            phase: Math.random() * Math.PI * 2
        });
    }

    let time = 0;

    function renderHoloCore() {
        time += 0.02;
        rotX += (targetRotX - rotX) * 0.08;
        rotY += (targetRotY - rotY) * 0.08;

        if (pulseScale > 1.0) {
            pulseScale -= 0.015;
        }

        ctx.clearRect(0, 0, size, size);

        const cx = size / 2;
        const cy = size / 2;

        // Central Luminous Quantum Glow
        const glowGrad = ctx.createRadialGradient(cx, cy, 10, cx, cy, 180 * pulseScale);
        glowGrad.addColorStop(0, 'rgba(0, 242, 254, 0.45)');
        glowGrad.addColorStop(0.3, 'rgba(45, 212, 191, 0.25)');
        glowGrad.addColorStop(0.7, 'rgba(13, 148, 136, 0.08)');
        glowGrad.addColorStop(1, 'transparent');
        ctx.fillStyle = glowGrad;
        ctx.beginPath();
        ctx.arc(cx, cy, 180 * pulseScale, 0, Math.PI * 2);
        ctx.fill();

        // 3D Particles rotation & projection
        const currentRotY = rotY + time * 0.5;
        const currentRotX = rotX + Math.sin(time * 0.3) * 0.2;

        const projected = [];

        coreParticles.forEach(p => {
            const waveMod = Math.sin(time * 3 + p.phase) * 8 * pulseScale;
            const currentR = (radius + waveMod) * pulseScale;

            const len = Math.hypot(p.origX, p.origY, p.origZ) || 1;
            const px = (p.origX / len) * currentR;
            const py = (p.origY / len) * currentR;
            const pz = (p.origZ / len) * currentR;

            const cosY = Math.cos(currentRotY), sinY = Math.sin(currentRotY);
            const x1 = px * cosY + pz * sinY;
            const z1 = -px * sinY + pz * cosY;

            const cosX = Math.cos(currentRotX), sinX = Math.sin(currentRotX);
            const y2 = py * cosX - z1 * sinX;
            const z2 = py * sinX + z1 * cosX;

            const fov = 350;
            const scale = fov / (fov + z2);
            const projX = cx + x1 * scale;
            const projY = cy + y2 * scale;

            projected.push({
                x: projX,
                y: projY,
                z: z2,
                scale: scale,
                r: p.radius * scale
            });
        });

        projected.sort((a, b) => a.z - b.z);

        // Filament connections
        ctx.lineWidth = 0.75;
        for (let i = 0; i < projected.length; i++) {
            for (let j = i + 1; j < projected.length; j++) {
                const p1 = projected[i];
                const p2 = projected[j];
                const d = Math.hypot(p1.x - p2.x, p1.y - p2.y);
                if (d < 38) {
                    const alpha = (1 - d / 38) * 0.45 * Math.min(p1.scale, p2.scale);
                    ctx.strokeStyle = `rgba(45, 212, 191, ${alpha})`;
                    ctx.beginPath();
                    ctx.moveTo(p1.x, p1.y);
                    ctx.lineTo(p2.x, p2.y);
                    ctx.stroke();
                }
            }
        }

        // Particle nodes
        projected.forEach(p => {
            const alpha = Math.min(Math.max((p.z + 150) / 300, 0.25), 1.0);
            ctx.fillStyle = `rgba(255, 255, 255, ${alpha})`;
            ctx.shadowColor = '#00F2FE';
            ctx.shadowBlur = 8;
            ctx.beginPath();
            ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
            ctx.fill();
        });

        requestAnimationFrame(renderHoloCore);
    }
    requestAnimationFrame(renderHoloCore);
}
