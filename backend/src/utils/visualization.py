"""
SonicSentinel AI - Waveform and Spectrogram Visualization Utilities
"""
import numpy as np
import librosa
import librosa.display
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from typing import Optional
import warnings
warnings.filterwarnings("ignore")

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config.settings import SAMPLE_RATE, N_MELS, HOP_LENGTH, N_FFT

# Color palette
BG_COLOR = "#0a0e1a"
WAVEFORM_COLOR = "#00d4ff"
SPEC_CMAP = "magma"
GRID_COLOR = "#1e2d4a"


def generate_waveform(
    file_path: str,
    audio_id: str,
    output_dir: str,
    sr: int = SAMPLE_RATE,
) -> Optional[str]:
    """
    Generate a professional waveform visualization.
    Returns output file path.
    """
    try:
        y, _ = librosa.load(file_path, sr=sr, mono=True)
        duration = len(y) / sr
        time_axis = np.linspace(0, duration, len(y))

        fig, ax = plt.subplots(figsize=(12, 3), facecolor=BG_COLOR)
        ax.set_facecolor(BG_COLOR)

        # Fill waveform
        ax.fill_between(time_axis, y, alpha=0.7, color=WAVEFORM_COLOR, linewidth=0)
        ax.plot(time_axis, y, color=WAVEFORM_COLOR, linewidth=0.5, alpha=0.9)

        # Reference lines
        ax.axhline(y=0, color="#2d4a6b", linewidth=0.8, linestyle="--", alpha=0.6)
        ax.axhline(y=0.5, color="#1e3a5f", linewidth=0.5, linestyle=":", alpha=0.4)
        ax.axhline(y=-0.5, color="#1e3a5f", linewidth=0.5, linestyle=":", alpha=0.4)

        # Styling
        ax.set_xlabel("Time (s)", color="#94a3b8", fontsize=10)
        ax.set_ylabel("Amplitude", color="#94a3b8", fontsize=10)
        ax.tick_params(colors="#94a3b8", labelsize=9)
        ax.spines[:].set_color(GRID_COLOR)
        ax.set_xlim([0, duration])
        ax.set_ylim([-1.05, 1.05])
        ax.grid(True, color=GRID_COLOR, linewidth=0.5, alpha=0.5)

        # RMS envelope
        frame_rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
        rms_times = librosa.frames_to_time(
            np.arange(len(frame_rms)), sr=sr, hop_length=512
        )
        ax.plot(rms_times, frame_rms, color="#ff6b35", linewidth=1.5,
                alpha=0.8, label="RMS Energy")
        ax.legend(loc="upper right", fontsize=8,
                  facecolor="#0d1526", edgecolor="#1e2d4a", labelcolor="#94a3b8")

        plt.tight_layout(pad=0.5)

        output_path = Path(output_dir) / f"{audio_id}_waveform.png"
        plt.savefig(str(output_path), dpi=150, bbox_inches="tight",
                    facecolor=BG_COLOR, edgecolor="none")
        plt.close(fig)
        return str(output_path)

    except Exception as e:
        print(f"Waveform generation error: {e}")
        return None


def generate_spectrogram(
    file_path: str,
    audio_id: str,
    output_dir: str,
    sr: int = SAMPLE_RATE,
    n_mels: int = N_MELS,
) -> Optional[str]:
    """
    Generate a Mel spectrogram visualization.
    Returns output file path.
    """
    try:
        y, _ = librosa.load(file_path, sr=sr, mono=True)

        mel = librosa.feature.melspectrogram(
            y=y, sr=sr, n_mels=n_mels, n_fft=N_FFT, hop_length=HOP_LENGTH
        )
        mel_db = librosa.power_to_db(mel, ref=np.max)

        fig, ax = plt.subplots(figsize=(12, 4), facecolor=BG_COLOR)
        ax.set_facecolor(BG_COLOR)

        img = librosa.display.specshow(
            mel_db,
            x_axis="time",
            y_axis="mel",
            sr=sr,
            hop_length=HOP_LENGTH,
            cmap=SPEC_CMAP,
            ax=ax,
        )

        cbar = plt.colorbar(img, ax=ax, format="%+2.0f dB", pad=0.01)
        cbar.ax.yaxis.set_tick_params(color="#94a3b8", labelsize=8)
        cbar.ax.set_ylabel("dB", color="#94a3b8", fontsize=9)

        ax.set_xlabel("Time (s)", color="#94a3b8", fontsize=10)
        ax.set_ylabel("Frequency (Hz)", color="#94a3b8", fontsize=10)
        ax.tick_params(colors="#94a3b8", labelsize=9)
        ax.spines[:].set_color(GRID_COLOR)

        plt.tight_layout(pad=0.5)

        output_path = Path(output_dir) / f"{audio_id}_spectrogram.png"
        plt.savefig(str(output_path), dpi=150, bbox_inches="tight",
                    facecolor=BG_COLOR, edgecolor="none")
        plt.close(fig)
        return str(output_path)

    except Exception as e:
        print(f"Spectrogram generation error: {e}")
        return None


def generate_confidence_chart(
    confidence_scores: dict,
    predicted_class: str,
    audio_id: str,
    output_dir: str,
) -> Optional[str]:
    """Generate a horizontal bar chart of confidence scores."""
    try:
        sorted_items = sorted(confidence_scores.items(), key=lambda x: x[1], reverse=True)
        classes = [item[0].replace("_", " ").title() for item in sorted_items]
        scores = [item[1] * 100 for item in sorted_items]

        colors = [
            "#00d4ff" if item[0] == predicted_class else "#1e3d5c"
            for item in sorted_items
        ]

        fig, ax = plt.subplots(figsize=(10, 6), facecolor=BG_COLOR)
        ax.set_facecolor(BG_COLOR)

        bars = ax.barh(classes, scores, color=colors, edgecolor="none", height=0.6)

        # Add value labels
        for bar, score in zip(bars, scores):
            ax.text(
                min(score + 1, 98), bar.get_y() + bar.get_height() / 2,
                f"{score:.1f}%",
                va="center", ha="left", color="#94a3b8", fontsize=9,
            )

        ax.set_xlabel("Confidence (%)", color="#94a3b8", fontsize=10)
        ax.set_xlim([0, 105])
        ax.tick_params(colors="#94a3b8", labelsize=9)
        ax.spines[:].set_color(GRID_COLOR)
        ax.grid(axis="x", color=GRID_COLOR, linewidth=0.5, alpha=0.5)
        ax.invert_yaxis()

        plt.tight_layout(pad=0.5)

        output_path = Path(output_dir) / f"{audio_id}_confidence.png"
        plt.savefig(str(output_path), dpi=130, bbox_inches="tight",
                    facecolor=BG_COLOR, edgecolor="none")
        plt.close(fig)
        return str(output_path)

    except Exception as e:
        print(f"Confidence chart error: {e}")
        return None


def generate_confusion_matrix_plot(
    cm: np.ndarray,
    class_names: list,
    output_path: str,
    title: str = "Confusion Matrix",
) -> None:
    """Generate and save a confusion matrix heatmap."""
    import seaborn as sns

    fig, ax = plt.subplots(figsize=(12, 10), facecolor=BG_COLOR)
    ax.set_facecolor(BG_COLOR)

    # Normalize to percentages
    cm_norm = cm.astype("float") / (cm.sum(axis=1, keepdims=True) + 1e-9)

    sns.heatmap(
        cm_norm, annot=True, fmt=".2f",
        xticklabels=[c.replace("_", " ").title() for c in class_names],
        yticklabels=[c.replace("_", " ").title() for c in class_names],
        cmap="Blues", ax=ax,
        linewidths=0.5, linecolor=BG_COLOR,
        annot_kws={"size": 9},
    )

    ax.set_xlabel("Predicted", color="#94a3b8", fontsize=11)
    ax.set_ylabel("Actual", color="#94a3b8", fontsize=11)
    ax.set_title(title, color="#e2e8f0", fontsize=13, pad=15)
    ax.tick_params(colors="#94a3b8", labelsize=8, rotation=45)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight",
                facecolor=BG_COLOR, edgecolor="none")
    plt.close(fig)
