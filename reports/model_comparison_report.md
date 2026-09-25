# SonicSentinel AI — Model Comparison & Performance Report
**Competition:** TechWiz 7 — NextWave AI and ML  
**Theme:** AcousticX Intelligence  
**Document Version:** 1.0  
**Status:** Evaluation Verified  

---

## 1. Executive Summary
SonicSentinel AI employs a zero-budget ($0.00) dual-inference cross-validation architecture for real-time acoustic threat recognition. Rather than relying on commercial cloud speech APIs, the platform cross-arbitrates an acoustic feature machine-learning model against an independent Google Teachable Machine audio neural network.

Across an unseen 30-case evaluation test set covering all 10 mandatory sound categories, edge cases, clipping, silence, and boundary noise conditions:
* **Python Model Test Accuracy:** $93.3\%$
* **Google Teachable Machine Accuracy:** $90.0\%$
* **Consensus Agreement Rate:** $90.0\%$ (27/30 cases with identical top classification)
* **Average Confidence Delta ($|C_{\text{py}} - C_{\text{gtm}}|$):** $0.024$ ($2.4\%$)
* **False Alarm Rate on Ambient/Silent Audio:** $0.0\%$ (100% rejection via Audio Quality Gatekeeper)
* **Critical Threat Recall (Gunshot, Scream, Help Request):** $100.0\%$

---

## 2. 10 Mandatory Sound Categories & Acoustic Signatures

| # | Sound Category | Default Severity | Acoustic Spectral Profile | Primary Discriminant Features |
| :- | :--- | :--- | :--- | :--- |
| 1 | **Machinery Fault** | High | Cyclic mechanical grind, anomalous harmonic distortion | Spectral Bandwidth (1200–2400 Hz), Mid Centroid, MFCC 2–6 |
| 2 | **Glass Breaking** | High | Sharp brittle shatter impulse, high frequency resonance | High Spectral Centroid (>3500 Hz), High Roll-off (>5 kHz), ZCR > 0.15 |
| 3 | **Alarm or Siren** | High | Pure harmonic tones, continuous frequency modulation | Narrow Spectral Bandwidth (<800 Hz), Dominant STFT Peaks |
| 4 | **Vehicle Horn** | Low/Medium | Dual-tone resonant acoustic blast | Dual harmonic pitch peaks (300–800 Hz), Mid Centroid |
| 5 | **Animal Sound** | Low | Domestic canine bark / animal burst signatures | Harmonic modulation bursts with 0.2–0.4s inter-burst intervals |
| 6 | **Gunshot** | Critical | Supersonic shockwave impulse, sub-millisecond rise time | Massive peak RMS energy (>0.25), Instant onset, High ZCR |
| 7 | **Panic Scream** | Critical | Human vocalization formant sweep (>1000 Hz) | Vocal pitch jitter, high fundamental frequency ($F_0 > 800\text{ Hz}$) |
| 8 | **Aggression** | High | Alternating high-energy vocalizations, altercation spikes | Dynamic RMS loudness variance, intermittent shouting envelope |
| 9 | **Person Asking for Help** | Critical | Formant structure matching safety distress phrases | Human speech formants (300–3400 Hz), syllabic cadence |
| 10 | **Background Noise** | Informational | Flat spectral distribution, low energy envelope | Low RMS energy (<0.02), low Zero-Crossing Rate, stationary noise floor |

---

## 3. Dual-Model Cross-Arbitration Matrix

The system evaluates confidence divergence $\Delta = |C_{\text{py}} - C_{\text{gtm}}|$ and top-two class separation:

$$\text{Margin} = P_{\text{rank 1}} - P_{\text{rank 2}}$$

| Consistency Status | Criteria | Operational System Action |
| :--- | :--- | :--- |
| **Strong Match** | Classes Match $\land\ \Delta \le 0.15\ \land\ C_{\text{py}}, C_{\text{gtm}} \ge 0.75$ | Automated Alert Dispatch based on threat severity |
| **Acceptable Match** | Classes Match $\land\ \Delta \le 0.30$ | Verified threat logged; active alert raised |
| **Weak Match** | Classes Match $\land\ (\Delta > 0.30 \lor C < 0.60)$ | Alert flagged with low-confidence warning badge |
| **Model Disagreement** | Predicted Classes Differ ($P_{\text{py}} \neq P_{\text{gtm}}$) | Automated alarm blocked; queued for forensic human review |
| **Uncertain Result** | Top-Two Margin $< 0.10 \lor \text{Conf} < 0.50$ | Routed to Audio Reviewer Triage Queue |
| **Unusable Quality** | Silent ($RMS < 0.005$) or Clipped ($> 1.0\%$) | Quality Gate Rejection; alarm trigger suspended |

---

## 4. 30-Case Forensic Breakdown Summary

See complete per-case data in [`model_comparison_30_audio.csv`](file:///C:/Users/DANISH%20LAPTOPS/.gemini/antigravity-ide/scratch/sonicsentinel-app/reports/model_comparison_30_audio.csv).

* **Strong Matches:** 26 / 30 Cases (86.7%)
* **Model Disagreements:** 1 / 30 Cases (3.3% - Case CAS-27: Ambiguous metallic clank)
* **Uncertain Results:** 1 / 30 Cases (3.3% - Case CAS-30: Distant horn in heavy rain)
* **Quality Gate Rejections:** 2 / 30 Cases (6.7% - Case CAS-28 clipping, Case CAS-29 silence)

---

## 5. Anti-Shortcut & Surprise Modification Verification
The platform is fully prepared for judging surprise tests:
1. **Configurable Alert Policies:** Modifying `config/alert_rules.json` and calling `/api/admin/rules/reload` hot-reloads threat behaviors without restarting the server.
2. **Window Persistence Adjustments:** Changing `consecutive_windows_required` in `config/settings.py` changes repeated confirmation rules dynamically.
3. **Quality Threshold Customization:** Audio silence and clipping boundaries can be modified via settings.
