# SonicSentinel AI — AI & Tool Declaration Log
**Competition:** TechWiz 7 — NextWave AI and ML  
**Theme:** AcousticX Intelligence  
**Declaration Version:** 1.0 (Mandatory Compliance)  

---

## 1. Zero-Budget & Open-Source Compliance Statement
SonicSentinel AI was designed and constructed under a strict **$0.00 commercial software budget**. No paid third-party APIs (e.g., OpenAI Whisper, AWS Transcribe, Google Cloud Speech, or commercial audio monitoring SDKs) were used in development or runtime inference.

All inference runs 100% locally and on-device utilizing open-source Python acoustic engineering libraries, NumPy digital signal processing, Scikit-Learn tabular feature modeling, and Google Teachable Machine exported neural architectures.

---

## 2. AI Tools & Assistant Usage Declaration

| AI Tool / Library | Role & Application in Project | Stage Used |
| :--- | :--- | :--- |
| **Antigravity AI Assistant** | Code structure scaffolding, architectural blueprint drafting, testing scripts, and CSS glassmorphism token organization. | Design & Implementation |
| **Google Teachable Machine (Audio)** | Training and exporting a 10-class spectrogram neural network for secondary cross-validation. | Model Training & Inference |
| **NumPy & SciPy DSP** | Short-Time Fourier Transform (STFT), Mel Filterbank calculation, RMS frame power, Zero-Crossing Rate, and SNR estimation. | Core DSP Engineering |
| **Scikit-Learn** | Feature vector scaling and classifier decision boundaries across acoustic descriptors. | Python ML Engine |
| **ReportLab** | Forensic vector PDF Acoustic Incident Certificate layout and rendering. | Incident Reporting |
| **Chart.js** | Client-side dynamic incident timeline and class distribution visualizations. | Presentation Layer |

---

## 3. Training & Dataset Provenance
* **Animal Sound Category:** Trained utilizing the local 300-clip domestic canine acoustic recording dataset (`dog_bark/`).
* **Remaining Sound Categories:** Acoustic profiles modeled after standard acoustic benchmark corpora (UrbanSound8K, ESC-50, and synthesized impulse-formant signatures).
* **Audio Preprocessing Standardization:** 22,050 Hz Mono, amplitude normalized, with 40 MFCC coefficients, spectral centroid, spectral bandwidth, roll-off, and RMS energy.
