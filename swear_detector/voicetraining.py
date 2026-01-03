"""
Speaker Training System - Create Voice Signatures
==================================================
This script processes pre-recorded WAV files to create speaker signatures.
Run this ONCE to train the system on Mike and James's voices.

Requirements:
1. pip install vosk numpy scipy wave
2. vosk-model-en-us-0.22 (speech recognition)
3. vosk-model-spk-0.4 (speaker identification)
4. mike.wav and james.wav in same folder

Output: mike_signature.npy and james_signature.npy

Author: FIFA Swear Counter Project
"""

# =============================================================================
# IMPORTS
# =============================================================================
print("=" * 70)
print("SPEAKER TRAINING - Creating Voice Signatures")
print("=" * 70)

import vosk
import wave
import json
import numpy as np
import os

print("✓ Libraries imported\n")

# =============================================================================
# CONFIGURATION
# =============================================================================
print("=" * 70)
print("CONFIGURATION")
print("=" * 70)

# Model paths
SPEECH_MODEL_PATH = "vosk-model-en-us-0.22"
SPEAKER_MODEL_PATH = "vosk-model-spk-0.4"

# Training file paths
MIKE_WAV = "mike.wav"
JAMES_WAV = "james.wav"

# Output signature files
MIKE_SIGNATURE_FILE = "mike_signature.npy"
JAMES_SIGNATURE_FILE = "james_signature.npy"

print(f"Speech model: {SPEECH_MODEL_PATH}")
print(f"Speaker model: {SPEAKER_MODEL_PATH}")
print(f"Mike's training file: {MIKE_WAV}")
print(f"James's training file: {JAMES_WAV}")

# =============================================================================
# CHECK FILES EXIST
# =============================================================================
print("\n" + "=" * 70)
print("CHECKING FILES")
print("=" * 70)

# Check models
if not os.path.exists(SPEECH_MODEL_PATH):
    print(f"✗ ERROR: Speech model not found: {SPEECH_MODEL_PATH}")
    exit(1)
print(f"✓ Speech model found")

if not os.path.exists(SPEAKER_MODEL_PATH):
    print(f"✗ ERROR: Speaker model not found: {SPEAKER_MODEL_PATH}")
    exit(1)
print(f"✓ Speaker model found")

# Check WAV files
if not os.path.exists(MIKE_WAV):
    print(f"✗ ERROR: Mike's WAV file not found: {MIKE_WAV}")
    exit(1)
print(f"✓ {MIKE_WAV} found")

if not os.path.exists(JAMES_WAV):
    print(f"✗ ERROR: James's WAV file not found: {JAMES_WAV}")
    exit(1)
print(f"✓ {JAMES_WAV} found")

# =============================================================================
# LOAD MODELS
# =============================================================================
print("\n" + "=" * 70)
print("LOADING MODELS")
print("=" * 70)

print("Loading speech model...")
speech_model = vosk.Model(SPEECH_MODEL_PATH)
print("✓ Speech model loaded")

print("Loading speaker model...")
speaker_model = vosk.SpkModel(SPEAKER_MODEL_PATH)
print("✓ Speaker model loaded")


# =============================================================================
# FUNCTION: EXTRACT SPEAKER VECTORS FROM WAV FILE
# =============================================================================

def extract_speaker_vectors(wav_filename, speaker_name):
    """
    Extract all speaker vectors from a WAV file.

    Args:
        wav_filename: Path to WAV file
        speaker_name: Name for display (e.g., "MIKE")

    Returns:
        List of speaker vectors (voice fingerprints)
    """
    print(f"\n--- Processing {speaker_name}'s voice ---")
    print(f"File: {wav_filename}")

    # Open the WAV file
    wf = wave.open(wav_filename, "rb")

    # Check audio format
    sample_rate = wf.getframerate()
    channels = wf.getnchannels()

    print(f"Sample rate: {sample_rate} Hz")
    print(f"Channels: {channels}")

    # Vosk requires 16000 Hz
    if sample_rate != 16000:
        print(f"⚠️  WARNING: File is {sample_rate} Hz, Vosk works best at 16000 Hz")
        print(f"   Detection may still work, but consider re-converting the file")

    # Create recognizer for this file
    recognizer = vosk.KaldiRecognizer(speech_model, sample_rate, speaker_model)
    recognizer.SetWords(True)

    # Process the entire file
    vectors = []
    word_count = 0
    chunk_size = 4000  # Process in chunks

    print("Processing audio...")

    while True:
        # Read chunk of audio
        data = wf.readframes(chunk_size)
        if len(data) == 0:
            break

        # Process chunk
        if recognizer.AcceptWaveform(data):
            # Got a complete phrase
            result = json.loads(recognizer.Result())

            # Extract speaker vector if available
            if "spk" in result:
                vectors.append(result["spk"])

            # Count words
            if "text" in result and result["text"]:
                words = result["text"].split()
                word_count += len(words)
                print(f"  Words processed: {word_count}", end='\r')

    # Get final result
    final_result = json.loads(recognizer.FinalResult())
    if "spk" in final_result:
        vectors.append(final_result["spk"])

    wf.close()

    print(f"\n✓ Extracted {len(vectors)} voice samples")
    print(f"✓ Processed ~{word_count} words")

    return vectors


# =============================================================================
# EXTRACT VECTORS FROM BOTH FILES
# =============================================================================
print("\n" + "=" * 70)
print("EXTRACTING VOICE SIGNATURES")
print("=" * 70)

# Process Mike's file
mike_vectors = extract_speaker_vectors(MIKE_WAV, "MIKE")

# Process James's file
james_vectors = extract_speaker_vectors(JAMES_WAV, "JAMES")

# =============================================================================
# CREATE AND SAVE SIGNATURES
# =============================================================================
print("\n" + "=" * 70)
print("CREATING SIGNATURES")
print("=" * 70)

# Check we got enough data
if len(mike_vectors) == 0:
    print("✗ ERROR: No voice samples extracted from Mike's file!")
    print("  Make sure the WAV file contains clear speech")
    exit(1)

if len(james_vectors) == 0:
    print("✗ ERROR: No voice samples extracted from James's file!")
    print("  Make sure the WAV file contains clear speech")
    exit(1)

# Calculate average vector for each person
# This creates their unique "voice signature"
mike_signature = np.mean(mike_vectors, axis=0)
james_signature = np.mean(james_vectors, axis=0)

print(f"✓ Mike's signature created from {len(mike_vectors)} samples")
print(f"✓ James's signature created from {len(james_vectors)} samples")

# Save signatures to disk
np.save(MIKE_SIGNATURE_FILE, mike_signature)
np.save(JAMES_SIGNATURE_FILE, james_signature)

print(f"\n✓ Saved: {MIKE_SIGNATURE_FILE}")
print(f"✓ Saved: {JAMES_SIGNATURE_FILE}")

# =============================================================================
# QUALITY CHECK
# =============================================================================
print("\n" + "=" * 70)
print("QUALITY CHECK")
print("=" * 70)

# Calculate similarity between Mike and James
# (Lower = more different = better for identification)
from scipy.spatial.distance import cosine

similarity = cosine(mike_signature, james_signature)

print(f"Voice difference score: {similarity:.3f}")

if similarity < 0.3:
    print("⚠️  WARNING: Voices are very similar (score < 0.3)")
    print("   Speaker identification may be less accurate")
elif similarity < 0.5:
    print("✓ Voices are somewhat different (score 0.3-0.5)")
    print("  Speaker identification should work reasonably well")
else:
    print("✓✓ Voices are very different (score > 0.5)")
    print("  Speaker identification should work excellently!")

# =============================================================================
# DONE
# =============================================================================
print("\n" + "=" * 70)
print("TRAINING COMPLETE!")
print("=" * 70)
print("\nSignature files created:")
print(f"  - {MIKE_SIGNATURE_FILE}")
print(f"  - {JAMES_SIGNATURE_FILE}")
print("\nYou can now run the live swear detector!")
print("=" * 70)