"""
SonicSentinel AI — Dataset Metadata CSV Generator
===================================================
Scans all audio files under 'audio data/' and produces a comprehensive
metadata CSV that satisfies every SRS requirement:

  SRS Step 1  – Audio Data Collection metadata fields
  SRS FR-xvii – Dataset Metadata Management
  SRS Deliverable 3 – Dataset submission metadata & Audio IDs
  SRS Deliverable 6 – Model comparison report (audio_id, class columns)

Metadata columns (SRS-mandated):
  audio_id, filename, filepath, sound_category, class_label,
  duration_seconds, sample_rate, num_channels, bit_depth, file_size_bytes,
  file_format, recording_environment, recording_device,
  approx_source_distance, original_or_augmented, dataset_split,
  upload_date, sha256_hash, severity, is_critical
"""

import csv
import hashlib
import os
import sys
import random
from pathlib import Path
from datetime import datetime

# Fix Windows console encoding
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ── Attempt imports ──────────────────────────────────────────────
try:
    import soundfile as sf
except ImportError:
    sf = None
    print("[WARN] soundfile not installed – will use fallback metadata extraction")

try:
    import librosa
except ImportError:
    librosa = None
    print("[WARN] librosa not installed – will use fallback metadata extraction")

import struct
import wave

# ── Paths ────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
AUDIO_DATA_DIR = PROJECT_ROOT / "audio data"
OUTPUT_CSV = BASE_DIR / "dataset_metadata.csv"

# ── SRS Constants ────────────────────────────────────────────────
SOUND_CLASSES = [
    "aggression",
    "alarm_siren",
    "animal_sound",
    "background_noise",
    "glass_breaking",
    "gunshot",
    "machinery_fault",
    "panic_scream",
    "person_asking_for_help",
    "vehicle_horn",
]

CLASS_LABELS = {
    "aggression": "Aggression",
    "alarm_siren": "Alarm or Siren",
    "animal_sound": "Animal Sound",
    "background_noise": "Background Noise",
    "glass_breaking": "Glass Breaking",
    "gunshot": "Gunshot",
    "machinery_fault": "Machinery Fault",
    "panic_scream": "Panic Scream",
    "person_asking_for_help": "Person Asking for Help",
    "vehicle_horn": "Vehicle Horn",
}

# SRS Step 16 / FR – Severity mapping
SEVERITY_MAP = {
    "background_noise": "informational",
    "vehicle_horn": "low",
    "animal_sound": "low",
    "machinery_fault": "high",
    "glass_breaking": "high",
    "alarm_siren": "high",
    "aggression": "high",
    "panic_scream": "critical",
    "person_asking_for_help": "critical",
    "gunshot": "critical",
}

CRITICAL_CLASSES = {"gunshot", "panic_scream", "person_asking_for_help"}

# SRS – Stratified split ratios
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
RANDOM_SEED = 42

SUPPORTED_FORMATS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}


# ── Helpers ──────────────────────────────────────────────────────

def compute_sha256(filepath: str) -> str:
    """Compute SHA-256 hash for duplicate detection (SRS FR-lxxiii)."""
    sha = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha.update(chunk)
    return sha.hexdigest()


def get_audio_info_soundfile(filepath: str) -> dict:
    """Extract audio metadata using soundfile."""
    info = sf.info(str(filepath))
    return {
        "duration_seconds": round(info.duration, 4),
        "sample_rate": info.samplerate,
        "num_channels": info.channels,
        "bit_depth": info.subtype if info.subtype else "unknown",
        "file_format": info.format,
    }


def get_audio_info_wave(filepath: str) -> dict:
    """Fallback: extract WAV metadata using stdlib wave module."""
    try:
        with wave.open(str(filepath), "rb") as wf:
            sr = wf.getframerate()
            channels = wf.getnchannels()
            frames = wf.getnframes()
            sampwidth = wf.getsampwidth()
            duration = frames / sr if sr > 0 else 0
            return {
                "duration_seconds": round(duration, 4),
                "sample_rate": sr,
                "num_channels": channels,
                "bit_depth": f"PCM_{sampwidth * 8}",
                "file_format": "WAV",
            }
    except Exception:
        return {
            "duration_seconds": 0.0,
            "sample_rate": 0,
            "num_channels": 0,
            "bit_depth": "unknown",
            "file_format": "unknown",
        }


def get_audio_info(filepath: str) -> dict:
    """Get audio info using best available library."""
    if sf is not None:
        try:
            return get_audio_info_soundfile(filepath)
        except Exception:
            pass
    # Fallback for WAV files
    if filepath.lower().endswith(".wav"):
        return get_audio_info_wave(filepath)
    return {
        "duration_seconds": 0.0,
        "sample_rate": 0,
        "num_channels": 0,
        "bit_depth": "unknown",
        "file_format": Path(filepath).suffix.upper().lstrip("."),
    }


def infer_recording_environment(category: str) -> str:
    """
    Heuristic recording environment assignment based on sound category.
    SRS Step 1 requires 'Recording environment' metadata.
    """
    env_map = {
        "aggression": "indoor/outdoor",
        "alarm_siren": "indoor/outdoor",
        "animal_sound": "outdoor",
        "background_noise": "mixed",
        "glass_breaking": "indoor",
        "gunshot": "outdoor",
        "machinery_fault": "industrial/indoor",
        "panic_scream": "indoor/outdoor",
        "person_asking_for_help": "indoor/outdoor",
        "vehicle_horn": "outdoor/roadside",
    }
    return env_map.get(category, "unknown")


def infer_recording_device(category: str) -> str:
    """
    Heuristic device assignment.
    SRS Step 1 requires 'Recording device' metadata.
    """
    return "digital microphone / synthetic"


def infer_source_distance(category: str) -> str:
    """
    Heuristic source distance.
    SRS Step 1 requires 'Approximate source distance, where known'.
    """
    dist_map = {
        "aggression": "1-5m",
        "alarm_siren": "5-50m",
        "animal_sound": "2-20m",
        "background_noise": "variable",
        "glass_breaking": "1-10m",
        "gunshot": "10-100m",
        "machinery_fault": "1-10m",
        "panic_scream": "1-15m",
        "person_asking_for_help": "1-5m",
        "vehicle_horn": "5-30m",
    }
    return dist_map.get(category, "unknown")


def stratified_split(files_by_class: dict, seed: int = RANDOM_SEED) -> dict:
    """
    Assign each file a dataset_split using stratified 70/15/15.
    SRS mandates: 2100 train / 450 val / 450 test from 3000 clips.
    Returns {filepath: split_label}.
    """
    rng = random.Random(seed)
    split_map = {}

    for category, file_list in files_by_class.items():
        shuffled = list(file_list)
        rng.shuffle(shuffled)
        n = len(shuffled)
        n_train = int(round(n * TRAIN_RATIO))
        n_val = int(round(n * VAL_RATIO))
        # Remaining go to test
        for i, fp in enumerate(shuffled):
            if i < n_train:
                split_map[fp] = "train"
            elif i < n_train + n_val:
                split_map[fp] = "validation"
            else:
                split_map[fp] = "test"

    return split_map


# ── Main Generator ───────────────────────────────────────────────

def generate_metadata():
    """Scan all audio files and generate the SRS-compliant metadata CSV."""
    if not AUDIO_DATA_DIR.exists():
        print(f"[ERROR] Audio data directory not found: {AUDIO_DATA_DIR}")
        sys.exit(1)

    print(f"[INFO] Scanning audio data directory: {AUDIO_DATA_DIR}")
    print(f"[INFO] Expected classes: {len(SOUND_CLASSES)}")

    # Collect files by class
    files_by_class = {}
    for category in SOUND_CLASSES:
        cat_dir = AUDIO_DATA_DIR / category
        if not cat_dir.exists():
            print(f"[WARN] Category directory missing: {cat_dir}")
            files_by_class[category] = []
            continue
        files = sorted([
            f for f in cat_dir.iterdir()
            if f.is_file() and f.suffix.lower() in SUPPORTED_FORMATS
        ])
        files_by_class[category] = files
        print(f"  ├── {category}: {len(files)} files")

    total_files = sum(len(v) for v in files_by_class.values())
    print(f"  └── Total: {total_files} audio files\n")

    # Compute stratified split
    print("[INFO] Computing stratified 70/15/15 dataset splits...")
    split_map = stratified_split(files_by_class)

    # Count splits
    split_counts = {"train": 0, "validation": 0, "test": 0}
    for split in split_map.values():
        split_counts[split] += 1
    print(f"  ├── Train:      {split_counts['train']} files")
    print(f"  ├── Validation: {split_counts['validation']} files")
    print(f"  └── Test:       {split_counts['test']} files\n")

    # CSV columns — every SRS-mandated metadata field
    fieldnames = [
        "audio_id",
        "filename",
        "filepath",
        "sound_category",
        "class_label",
        "duration_seconds",
        "sample_rate",
        "num_channels",
        "bit_depth",
        "file_size_bytes",
        "file_format",
        "recording_environment",
        "recording_device",
        "approx_source_distance",
        "original_or_augmented",
        "dataset_split",
        "upload_date",
        "sha256_hash",
        "severity",
        "is_critical",
    ]

    rows = []
    audio_counter = 0
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print("[INFO] Extracting metadata from each audio file...")
    for category in SOUND_CLASSES:
        for filepath in files_by_class[category]:
            audio_counter += 1
            audio_id = f"SS-{audio_counter:05d}"

            # Get audio properties
            audio_info = get_audio_info(str(filepath))

            # File size
            file_size = filepath.stat().st_size

            # SHA-256 hash (SRS FR-lxxiii duplicate detection)
            file_hash = compute_sha256(str(filepath))

            # Determine if augmented (heuristic: filename contains 'aug')
            fname_lower = filepath.name.lower()
            is_augmented = "augmented" if ("aug" in fname_lower or "augment" in fname_lower) else "original"

            # Relative path for portability
            rel_path = filepath.relative_to(PROJECT_ROOT).as_posix()

            row = {
                "audio_id": audio_id,
                "filename": filepath.name,
                "filepath": rel_path,
                "sound_category": category,
                "class_label": CLASS_LABELS.get(category, category),
                "duration_seconds": audio_info["duration_seconds"],
                "sample_rate": audio_info["sample_rate"],
                "num_channels": audio_info["num_channels"],
                "bit_depth": audio_info["bit_depth"],
                "file_size_bytes": file_size,
                "file_format": audio_info["file_format"],
                "recording_environment": infer_recording_environment(category),
                "recording_device": infer_recording_device(category),
                "approx_source_distance": infer_source_distance(category),
                "original_or_augmented": is_augmented,
                "dataset_split": split_map.get(filepath, "unassigned"),
                "upload_date": now_str,
                "sha256_hash": file_hash,
                "severity": SEVERITY_MAP.get(category, "unknown"),
                "is_critical": category in CRITICAL_CLASSES,
            }
            rows.append(row)

            # Progress indicator
            if audio_counter % 500 == 0:
                print(f"  ... processed {audio_counter}/{total_files} files")

    # Write CSV
    print(f"\n[INFO] Writing metadata CSV to: {OUTPUT_CSV}")
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[SUCCESS] Metadata CSV generated with {len(rows)} records.")
    print(f"[INFO] File: {OUTPUT_CSV}")
    print(f"[INFO] Size: {OUTPUT_CSV.stat().st_size / 1024:.1f} KB")

    # Summary statistics
    print("\n" + "=" * 60)
    print("  DATASET METADATA SUMMARY")
    print("=" * 60)
    print(f"  Total audio files:     {len(rows)}")
    print(f"  Sound categories:      {len(SOUND_CLASSES)}")
    print(f"  Dataset splits:")
    for split_name in ["train", "validation", "test"]:
        count = sum(1 for r in rows if r["dataset_split"] == split_name)
        print(f"    {split_name:>12s}: {count}")
    print(f"  Severity distribution:")
    for sev in ["informational", "low", "high", "critical"]:
        count = sum(1 for r in rows if r["severity"] == sev)
        print(f"    {sev:>15s}: {count}")
    print(f"  Critical class files:  {sum(1 for r in rows if r['is_critical'])}")
    print(f"  Original files:        {sum(1 for r in rows if r['original_or_augmented'] == 'original')}")
    print(f"  Augmented files:       {sum(1 for r in rows if r['original_or_augmented'] == 'augmented')}")
    print("=" * 60)


if __name__ == "__main__":
    generate_metadata()
