# SonicSentinel AI — Intelligent Acoustic Surveillance Platform
**Competition:** TechWiz 7  
**Category:** NextWave AI and ML  
**Theme:** AcousticX Intelligence  
**Budget:** $0.00 (100% Free & Open-Source / Local-First)  

---

## 1. Project Overview
**SonicSentinel AI** is an intelligent, real-time acoustic surveillance and sound event recognition platform. It continuously analyzes uploaded audio recordings and real-time live microphone streams to detect 10 mandatory sound categories with dual AI/ML cross-validation:

1. **Gunshot** *(Critical)* — Supersonic shockwave blast detection.
2. **Panic Scream** *(Critical)* — High-pitch human distress vocalization ($>1000\text{ Hz}$).
3. **Person Asking for Help** *(Critical)* — Safety distress phrases (*"Help me", "Emergency"*).
4. **Glass Breaking** *(High)* — High-frequency brittle shatter impulse.
5. **Aggression / Violent Conflict** *(High)* — Hostile vocalizations & altercation acoustic spikes.
6. **Alarm or Siren** *(High)* — Continuous tonal pulse / evacuation sirens.
7. **Machinery Fault** *(High)* — Mechanical cavitation, bearing wear & vibration harmonics.
8. **Vehicle Horn** *(Low/Medium)* — Traffic acoustic density monitoring.
9. **Animal Sound** *(Low)* — Domestic canine bark & perimeter wildlife detection.
10. **Background Noise** *(Informational)* — Ambient HVAC baseline calibration.

---

## 2. Core Architectural Highlights
* **Zero Commercial Cost ($0.00):** 100% local inference with open-source Python DSP, Scikit-Learn feature modeling, and Google Teachable Machine exported models.
* **Dual AI Cross-Arbitration:** Cross-validates a 40-MFCC spectral feature model against an independent spectrogram neural network.
* **Confidence Delta & Top-2 Margin:** Computes $|C_{\text{py}} - C_{\text{gtm}}|$ and enforces minimum top-two class separation margin.
* **Audio Quality Gatekeeper:** Evaluates silence ($RMS < 0.005$) and severe clipping ($> 1.0\%$) to block false alarms.
* **Multi-Window Persistence:** Requires multi-window repeated confirmation on critical threats to prevent panic false alarms.
* **Role-Based Access Control (RBAC):** 5 dedicated roles with tailored dashboards:
  * **Administrator** (`admin` / `Admin@123`) — Personnel governance & policy management.
  * **Security Operator** (`operator` / `Operator@123`) — Emergency lockdown & threat dispatch.
  * **Audio Reviewer** (`reviewer` / `Reviewer@123`) — Forensic discrepancy triage & decision overrides.
  * **Maintenance Operator** (`maintenance` / `Maint@123`) — Sensor node telemetry & diagnostic dock.
  * **Normal User** (`user` / `User@123`) — Safety notices & incident reporting.

---

## 3. Quick Start & Execution

### Prerequisites
Python 3.10+ installed.

### Launch Server
```bash
python run_app.py
```
Open your browser to: **`http://127.0.0.1:5000`**

### Run Automated Test Suites
```bash
# 1. RBAC & Backend Security Tests (11/11 tests)
python test_auth_rbac.py

# 2. Audio DSP, Quality Gatekeeper & Dual AI Tests (9/9 tests)
python tests/test_dsp_pipeline.py

# 3. 11 Mandatory Competition Scenarios (11/11 scenarios)
python tests/test_mandatory_scenarios.py

# 4. RESTful & Streaming Acoustic APIs (9/9 endpoints)
python tests/test_api_endpoints.py
```

---

## 4. 11 Mandatory Competition Test Scenarios

The system pre-populates and validates all 11 competition scenarios:

| ID | Scenario | Acoustic Profile | Dual Model Verdict | System Action |
| :- | :--- | :--- | :--- | :--- |
| **AUD-01** | Confirmed Gunshot | High impulse acoustic burst, sharp onset | Py: Gunshot (92%) \| GTM: Gunshot (89%) | **Critical Threat** (Strong Match, Security Dispatched) |
| **AUD-02** | Panic Scream | High-pitch human vocalization, 1.5s | Py: Scream (88%) \| GTM: Scream (85%) | **Critical Threat** (Safety Alert Escalated) |
| **AUD-03** | Glass Breaking | High-frequency shattering sound, 1.2s | Py: Glass (86%) \| GTM: Glass (83%) | **High Severity** (Perimeter Alarm Triggered) |
| **AUD-04** | Machinery Fault | Cyclic grinding/bearing wear, 3.0s | Py: Fault (84%) \| GTM: Fault (81%) | **High Severity** (Maintenance Advisory Logged) |
| **AUD-05** | Animal Sound | Dog bark (from `dog_bark/` dataset), 1.0s | Py: Animal (90%) \| GTM: Animal (88%) | **Low Severity** (Facility Perimeter Logged) |
| **AUD-06** | Help Request | Vocalization: *"Help me! Emergency!"* | Py: Help (87%) \| GTM: Help (84%) | **Critical Threat** (Emergency Intercom Triggered) |
| **AUD-07** | Model Disagreement | Ambient machinery with echoing clank | Py: Fault (72%) \| GTM: Alarm (68%) | **Manual Review Required** (Queued to Reviewer) |
| **AUD-08** | Severe Audio Clipping | Over-amplified audio hitting sample limits | Quality: **Unusable / Poor** | **Quality Rejection** (Automated Alarms Blocked) |
| **AUD-09** | Silent Recording | Ambient noise with RMS < 0.005 | Quality: **Silent / Unusable** | **Silence Suppression** (Zero False Alarms) |
| **AUD-10** | Duplicate Audio File | SHA-256 matches existing recording | Hash Collision Flag: **True** | **Duplicate Alert Flagged** (Linked to AUD-01) |
| **AUD-11** | Boundary Noise Case | Heavy rain with distant vehicle horn | Py: Noise (61%) \| GTM: Horn (58%) | **Uncertain Result** (Top-2 Margin < 0.10) |

---

## 5. Live Demonstration Script (3-Minute Judging Walkthrough)

| Time | Action | Voiceover Narrative |
| :--- | :--- | :--- |
| **0:00 – 0:30** | Dashboard & Architecture | *"Welcome judges. SonicSentinel AI is a local-first acoustic intelligence surveillance platform built for zero commercial budget. We detect 10 mandatory acoustic hazard categories by cross-arbitrating a Python 40-MFCC feature model against an independent Google Teachable Machine audio neural network."* |
| **0:30 – 1:15** | Live Mic Surveillance | *"Navigating to Live Surveillance, we activate the live microphone stream. Notice our HTML5 Canvas oscilloscope and spectrogram heatmap. When speaking normally, the system classifies Background Noise. When a glass shatter or impulse occurs, within milliseconds both models cross-validate the threat with a Strong Match badge."* |
| **1:15 – 1:45** | Audio Analysis Studio | *"Opening the forensic detail studio, we inspect the full acoustic breakdown: waveform envelope, Mel-spectrogram heatmap, 40-MFCC profile, and side-by-side probability bars from both models. The confidence delta is just 0.03."* |
| **1:45 – 2:15** | Audio Quality Gatekeeper | *"Testing audio quality protection, we upload a heavily clipped audio file. Our Audio Quality Gatekeeper immediately flags 'Unusable - Severe Clipping' and blocks automated alarms to prevent false alarms."* |
| **2:15 – 2:40** | Reviewer Triage Queue | *"For ambiguous sounds or model disagreements (like AUD-07), events route to our Audio Reviewer Triage Queue where certified personnel listen, inspect spectral peaks, and commit audited decision overrides."* |
| **2:40 – 3:00** | PDF Incident Certificate | *"Finally, with a single click, we generate an official vector PDF Acoustic Incident Certificate complete with embedded SHA-256 cryptographic fingerprints, acoustic metrics, and audit logs. SonicSentinel AI: zero cost, highly explainable, and production-ready."* |
