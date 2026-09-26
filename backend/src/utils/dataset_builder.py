"""
SonicSentinel AI - Dataset Builder
Scans audio data folder, creates features CSV, handles train/val/test split.
"""
import os, csv, uuid
import numpy as np
import librosa
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings("ignore")

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config.settings import (
    AUDIO_DATA_DIR, SAMPLE_RATE, SOUND_CLASSES,
    TRAIN_SPLIT, VAL_SPLIT, TEST_SPLIT, RANDOM_SEED, MODELS_DIR
)
from audio_preprocessing.preprocessor import AudioPreprocessor
from feature_extraction.feature_extractor import FeatureExtractor


class DatasetBuilder:
    """Builds train/val/test feature datasets from the audio data directory."""

    def __init__(self, data_dir: Path = AUDIO_DATA_DIR):
        self.data_dir = Path(data_dir)
        self.preprocessor = AudioPreprocessor()
        self.extractor = FeatureExtractor()

    def scan_files(self) -> pd.DataFrame:
        """Scan audio data directory and create file manifest."""
        records = []
        for cls in SOUND_CLASSES:
            cls_dir = self.data_dir / cls
            if not cls_dir.exists():
                print(f"Warning: Class directory not found: {cls_dir}")
                continue
            files = list(cls_dir.glob("*.wav")) + list(cls_dir.glob("*.mp3"))
            for fp in files:
                records.append({
                    "audio_id": f"AUD-{uuid.uuid4().hex[:8]}",
                    "filename": fp.name,
                    "filepath": str(fp),
                    "class_label": cls,
                    "source": "original",
                    "split": None,
                })
        df = pd.DataFrame(records)
        print(f"Found {len(df)} audio files across {df['class_label'].nunique()} classes")
        print(df["class_label"].value_counts())
        return df

    def create_stratified_split(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create stratified 70/15/15 split."""
        df = df.copy()
        X = df["filepath"]
        y = df["class_label"]

        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=(VAL_SPLIT + TEST_SPLIT),
            random_state=RANDOM_SEED, stratify=y
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp,
            test_size=TEST_SPLIT / (VAL_SPLIT + TEST_SPLIT),
            random_state=RANDOM_SEED, stratify=y_temp
        )

        df.loc[X_train.index, "split"] = "train"
        df.loc[X_val.index, "split"] = "val"
        df.loc[X_test.index, "split"] = "test"

        print(f"\nDataset splits:")
        print(f"  Train: {len(X_train)} ({len(X_train)/len(df)*100:.1f}%)")
        print(f"  Val:   {len(X_val)} ({len(X_val)/len(df)*100:.1f}%)")
        print(f"  Test:  {len(X_test)} ({len(X_test)/len(df)*100:.1f}%)")
        return df

    def extract_features_for_split(
        self, df: pd.DataFrame, split: str, verbose: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Extract features for all files in a split."""
        split_df = df[df["split"] == split].reset_index(drop=True)
        X_list, y_list = [], []
        skipped = 0

        for i, row in split_df.iterrows():
            if verbose and i % 50 == 0:
                print(f"  [{split}] {i}/{len(split_df)} - {row['class_label']}")
            try:
                y_audio, _ = librosa.load(row["filepath"], sr=SAMPLE_RATE, mono=True)
                y_audio = self.preprocessor.normalize_amplitude(y_audio)
                y_audio = self.preprocessor.pad_or_truncate(
                    y_audio, int(3.0 * SAMPLE_RATE)
                )
                features = self.extractor.extract_all(y_audio)
                X_list.append(features)
                y_list.append(row["class_label"])
            except Exception as e:
                print(f"  Skipping {row['filename']}: {e}")
                skipped += 1

        if skipped > 0:
            print(f"  Skipped {skipped} files in {split} split")

        return np.array(X_list, dtype=np.float32), np.array(y_list)

    def build_full_dataset(
        self,
        save_dir: Path = None,
        verbose: bool = True,
    ) -> Dict[str, np.ndarray]:
        """Build complete feature dataset."""
        if save_dir is None:
            save_dir = MODELS_DIR

        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        print("=" * 60)
        print("SonicSentinel AI - Building Feature Dataset")
        print("=" * 60)

        df = self.scan_files()
        df = self.create_stratified_split(df)

        # Save manifest
        manifest_path = save_dir / "dataset_manifest.csv"
        df.to_csv(manifest_path, index=False)
        print(f"\nManifest saved: {manifest_path}")

        print("\nExtracting features for train split...")
        X_train, y_train = self.extract_features_for_split(df, "train", verbose)
        print(f"Train shape: {X_train.shape}")

        print("\nExtracting features for val split...")
        X_val, y_val = self.extract_features_for_split(df, "val", verbose)
        print(f"Val shape: {X_val.shape}")

        print("\nExtracting features for test split...")
        X_test, y_test = self.extract_features_for_split(df, "test", verbose)
        print(f"Test shape: {X_test.shape}")

        # Save numpy arrays
        np.save(save_dir / "X_train.npy", X_train)
        np.save(save_dir / "y_train.npy", y_train)
        np.save(save_dir / "X_val.npy", X_val)
        np.save(save_dir / "y_val.npy", y_val)
        np.save(save_dir / "X_test.npy", X_test)
        np.save(save_dir / "y_test.npy", y_test)

        print(f"\n✓ Dataset saved to {save_dir}")
        return {
            "X_train": X_train, "y_train": y_train,
            "X_val": X_val, "y_val": y_val,
            "X_test": X_test, "y_test": y_test,
            "manifest": df,
        }
