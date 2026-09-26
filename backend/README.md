# SonicSentinel AI — Backend & Machine Learning Pipeline
**Theme:** AcousticX Intelligence | **Category:** NextWave AI & ML

SonicSentinel AI is an intelligent acoustic event detection and monitoring system built to detect critical sounds (machinery faults, gunshots, screams, alarms, aggression, break-ins, calls for help) in real-time using deep learning and machine learning with multi-model validation.

---

## 📋 Table of Contents
1. [SRS Requirements Compliance](#-srs-requirements-compliance)
2. [Acoustic Sound Categories](#-acoustic-sound-categories)
3. [Architecture Overview](#-architecture-overview)
4. [Machine Learning Models & Overfitting Safeguards](#-machine-learning-models--overfitting-safeguards)
5. [Google Colab Training Notebook](#-google-colab-training-notebook)
6. [REST API Endpoints](#-rest-api-endpoints)
7. [Installation & Local Setup](#-installation--local-setup)
8. [Running Tests](#-running-tests)

---

## 🎯 SRS Requirements Compliance

| SRS Requirement | Implementation Detail | Status |
|-----------------|----------------------|--------|
| **10 Mandatory Categories** | All 10 sound categories implemented across preprocessing, models, and alerts | ✅ Fully Compliant |
| **3,000 Audio Samples** | Dataset partitioned into 300 samples per class with Stratified 70/15/15 splits | ✅ Fully Compliant |
| **Audio Preprocessing** | Amplitude normalization, noise suppression, 22050 Hz resampling, 3s segmentation, clipping/silence detection, SNR calculation | ✅ Fully Compliant |
| **Data Augmentation** | Pitch shifting (±2 semitones), time stretching (0.8x-1.2x), Gaussian noise, SpecAugment time/freq masking | ✅ Fully Compliant |
| **Feature Extraction** | MFCC (40 coeffs + deltas), Mel Spectrogram (128 bands), Chroma (12 pitch classes), Spectral Centroid, Rolloff, Contrast, ZCR, RMS | ✅ Fully Compliant |
| **Model Architectures** | CNN (2D Mel Spectrograms), Random Forest (300 trees), Support Vector Machine (RBF kernel with probabilities) | ✅ Fully Compliant |
| **Anti-Overfitting Protection** | 5-Fold Stratified Cross-Validation, Dropout (0.2-0.4), Batch Normalization, L2 Regularization, EarlyStopping with restore_best_weights, ReduceLROnPlateau | ✅ Fully Compliant |
| **Model Verification** | Consistency checker comparing Python model predictions against secondary/GTM predictions with confidence margin thresholds | ✅ Fully Compliant |
| **Alert Rules Engine** | Per-category thresholds, severity routing (Critical, High, Medium, Low), consecutive detection requirements, emergency recommendations | ✅ Fully Compliant |
| **Colab Training Notebook** | `backend/notebooks/SonicSentinel_AI_Training.ipynb` with full GPU execution, EDA, training, tuning, evaluation, confusion matrix, export | ✅ Fully Compliant |
| **REST API & Visualization** | Flask REST API with JWT authentication, file upload, live microphone inference, waveform and spectrogram generation, manual review queue | ✅ Fully Compliant |

---

## 🔊 Acoustic Sound Categories

The system monitors and classifies 10 sound categories:

1. **Machinery Fault** (`machinery_fault`): High-risk industrial bearing failures, engine knocks, mechanical grinding.
2. **Glass Breaking** (`glass_breaking`): High-frequency impact spikes representing security breaches or vandalism.
3. **Alarm or Siren** (`alarm_siren`): Modulated tones from fire alarms, industrial sirens, or emergency vehicles.
4. **Vehicle Horn** (`vehicle_horn`): Sustained tonal acoustic signatures from traffic and transportation.
5. **Animal Sound** (`animal_sound`): Dogs barking, wildlife calls, livestock disturbances.
6. **Gunshot** (`gunshot`): High-energy impulsive acoustic blast wave followed by rapid decay.
7. **Panic Scream** (`panic_scream`): High-pitched chaotic vocalizations indicating acute human distress.
8. **Aggression or Violent Conflict** (`aggression`): Raised combative voices, blunt physical impacts, verbal confrontation.
9. **Person Asking for Help** (`person_asking_for_help`): Urgent spoken distress calls ("Help", "Call police").
10. **Background Noise** (`background_noise`): Urban ambience, wind, traffic hum, HVAC, construction rumble.

---

## 🏗️ Architecture Overview

```
SONIC SENTINEL PROJECT/
├── audio data/                         # 3,000 curated audio clips (300 per class)
│   ├── aggression/
│   ├── alarm_siren/
│   ├── animal_sound/
│   ├── background_noise/
│   ├── glass_breaking/
│   ├── gunshot/
│   ├── machinery_fault/
│   ├── panic_scream/
│   ├── person_asking_for_help/
│   └── vehicle_horn/
└── backend/
    ├── alert_rules/
    │   └── alert_rules.json            # Category severity, thresholds, actions
    ├── audio_preprocessing/
    │   └── preprocessor.py             # Audio validation, filtering, SNR, chunking
    ├── augmentation/
    │   └── augmentor.py                # Pitch shift, stretch, noise, SpecAugment
    ├── config/
    │   └── settings.py                 # Paths, constants, hyperparams, thresholds
    ├── feature_extraction/
    │   └── feature_extractor.py        # MFCC, Mel spec, Chroma, spectral features
    ├── notebooks/
    │   └── SonicSentinel_AI_Training.ipynb # Google Colab notebook for GPU training
    ├── python_models/                  # Serialized trained model artifacts
    │   ├── best_model.pkl
    │   ├── scaler.pkl
    │   ├── label_encoder.pkl
    │   └── model_metadata.json
    ├── src/
    │   ├── database/
    │   │   └── models.py               # SQLAlchemy ORM models (Events, Alerts, Users)
    │   ├── models/
    │   │   ├── cnn_model.py            # Deep CNN architecture for Mel Spectrograms
    │   │   └── trainer.py              # Sklearn SVM, RF, and GBDT training pipeline
    │   ├── services/
    │   │   └── prediction_service.py   # Unified inference, validation, alerts
    │   └── utils/
    │       ├── dataset_builder.py      # Stratified splitting and feature caching
    │       └── visualization.py        # Dark-theme waveform and spectrogram generator
    ├── tests/
    │   ├── test_preprocessing.py       # Audio processing unit tests
    │   ├── test_feature_extraction.py  # Feature vector correctness tests
    │   ├── test_alert_rules.py         # Decision logic & threshold tests
    │   └── test_api.py                 # Full Flask REST API test suite
    ├── app.py                          # Flask application server
    ├── requirements.txt                # Production dependencies
    └── train_initial_models.py         # Local model training script
```

---

## 🧠 Machine Learning Models & Overfitting Safeguards

### 1. 2D Convolutional Neural Network (CNN)
- **Input:** Log Mel Spectrogram $(128 \times 130 \times 1)$
- **Blocks:** 4 Convolutional blocks $(32 \to 64 \to 128 \to 256)$ with $(3 \times 3)$ kernels
- **Anti-Overfitting:**
  - $L_2$ kernel regularization $(10^{-4})$
  - Batch Normalization after every convolution
  - Progressive Dropout $(0.20 \to 0.25 \to 0.30 \to 0.40)$
  - Global Average Pooling (drastically reduces parameters vs flatten)
  - `EarlyStopping` (patience = 8, `restore_best_weights = True`)
  - `ReduceLROnPlateau` (factor = 0.5, patience = 3, min_lr = $10^{-6}$)
  - Label Smoothing $(0.05)$ to prevent overconfident erroneous predictions

### 2. Random Forest Classifier
- 300 - 500 estimators with `max_features='sqrt'`
- `min_samples_leaf=2`, `min_samples_split=3` to avoid fitting noise
- `class_weight='balanced'` for robust multi-class representation
- 5-Fold Stratified Cross-Validation

### 3. Support Vector Machine (SVM)
- RBF Kernel: $K(x, x') = \exp(-\gamma ||x - x'||^2)$
- Soft margin parameter $C=10.0$ to balance margin width and training error
- Standardized feature scaling (zero mean, unit variance)
- Platt scaling (`probability=True`) for calibrated confidence scores

---

## ⚡ REST API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/health` | Backend status, version, and model loading state |
| `POST` | `/api/auth/register` | Register new user / security operator |
| `POST` | `/api/auth/login` | JWT authentication and role assignment |
| `POST` | `/api/audio/upload` | Upload audio file for full classification and alert checking |
| `POST` | `/api/live/classify` | Send live microphone chunk (base64 or binary) for real-time inference |
| `POST` | `/api/live/session/start` | Start live microphone monitoring session |
| `POST` | `/api/live/session/<id>/stop` | Stop live microphone session and return session summary |
| `GET` | `/api/audio/<id>` | Fetch full details and predictions for an audio event |
| `GET` | `/api/audio/<id>/waveform` | Retrieve rendered waveform visualization image |
| `GET` | `/api/audio/<id>/spectrogram` | Retrieve rendered Mel spectrogram visualization image |
| `GET` | `/api/alerts` | List generated alerts with severity, status, and filter options |
| `POST` | `/api/alerts/<id>/acknowledge` | Acknowledge/resolve critical alert |
| `GET` | `/api/review/queue` | List low-confidence or contradictory events awaiting manual review |
| `POST` | `/api/review/<audio_id>` | Submit human operator verified ground-truth label |
| `GET` | `/api/dashboard/stats` | Summary metrics: total events, active alerts, class breakdown |
| `GET` | `/api/events` | Paginated list of all sound events |

---

## 🚀 Running the Colab Training Notebook

1. Open [Google Colab](https://colab.research.google.com).
2. Upload `backend/notebooks/SonicSentinel_AI_Training.ipynb`.
3. Set runtime to **GPU** (`Runtime > Change runtime type > T4 GPU`).
4. Mount your Google Drive or upload the `audio data` zip archive.
5. Run all cells sequentially. The notebook performs:
   - Automated library installation
   - Dataset loading and class distribution validation
   - Exploratory Data Analysis (EDA) with waveforms and spectrograms
   - Stratified train/val/test splitting
   - Audio preprocessing & data augmentation
   - Comprehensive multi-domain feature extraction
   - SVM, Random Forest, and CNN training
   - 5-Fold cross-validation
   - Confusion matrices and per-class classification reports
   - Serialization and export of `best_model.pkl`, `cnn_model.h5`, `scaler.pkl`, `label_encoder.pkl`.

---

## 💻 Running the Flask Server Locally

```bash
# 1. Navigate to backend directory
cd backend

# 2. Train and export initial models from local dataset
python train_initial_models.py

# 3. Start the Flask server
python app.py
```

Server will run at `http://127.0.0.1:5000` with interactive API health at `/api/health`.
