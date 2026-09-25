"""
src/utils/audio_hash.py - SHA-256 Binary Audio Fingerprinting
Used for duplicate audio detection and cryptographic incident auditing.
"""

import hashlib


def compute_audio_hash(audio_bytes: bytes) -> str:
    """Computes SHA-256 cryptographic hash of raw audio byte stream."""
    return hashlib.sha256(audio_bytes).hexdigest()
