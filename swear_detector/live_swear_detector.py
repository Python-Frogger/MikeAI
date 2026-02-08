"""
Live FIFA Swear Word Detector - MAXIMUM RESPONSIVENESS MODE
=============================================================
Optimized for fastest possible detection with Vosk.
Processes smallest possible phrases and uses partial results.

Author: FIFA Swear Counter Project - Final Optimization
"""

# =============================================================================
# IMPORTS
# =============================================================================
print("=" * 70)
print("FIFA SWEAR DETECTOR - Maximum Responsiveness Mode")
print("=" * 70)

import pyaudio
import vosk
import json
import numpy as np
import os
from scipy.spatial.distance import cosine
from datetime import datetime

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

# Signature files
MIKE_SIGNATURE_FILE = "mike_signature.npy"
JAMES_SIGNATURE_FILE = "james_signature.npy"

# OPTIMIZED Audio settings for maximum responsiveness
SAMPLE_RATE = 16000
CHUNK_SIZE = 2000  # SMALLER chunks = faster processing (was 4000)

# SWEAR WORD LIST
SWEAR_WORDS = {
    "fuck", "fucking", "fucker", "fucked", "fucks",
    "shit", "shitting", "shitty", "shitter",
    "bastard", "bastards",
    "bollocks",
    "cunt", "cunts",
    "wanker", "wankers",
    "piss", "pissed", "pissing",
    "arse", "arsehole", "arseholes",
    "twat", "twats",
    "cock", "cocks",
    "dickhead", "dickheads",
    "bloody",
    "damn", "damned",
    "hell"
}

print(f"Speech model: {SPEECH_MODEL_PATH}")
print(f"Speaker model: {SPEAKER_MODEL_PATH}")
print(f"Monitoring {len(SWEAR_WORDS)} swear words")
print(f"Chunk size: {CHUNK_SIZE} (smaller = faster)")

# =============================================================================
# CHECK FILES EXIST
# =============================================================================
print("\n" + "=" * 70)
print("CHECKING FILES")
print("=" * 70)

if not os.path.exists(SPEECH_MODEL_PATH):
    print(f"✗ ERROR: Speech model not found")
    exit(1)
print(f"✓ Speech model found")

if not os.path.exists(SPEAKER_MODEL_PATH):
    print(f"✗ ERROR: Speaker model not found")
    exit(1)
print(f"✓ Speaker model found")

if not os.path.exists(MIKE_SIGNATURE_FILE):
    print(f"✗ ERROR: Mike's signature not found")
    exit(1)
print(f"✓ Mike's signature found")

if not os.path.exists(JAMES_SIGNATURE_FILE):
    print(f"✗ ERROR: James's signature not found")
    exit(1)
print(f"✓ James's signature found")

# =============================================================================
# LOAD MODELS AND SIGNATURES
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

print("Loading voice signatures...")
mike_signature = np.load(MIKE_SIGNATURE_FILE)
james_signature = np.load(JAMES_SIGNATURE_FILE)
print("✓ Signatures loaded")

# =============================================================================
# SET UP MICROPHONE
# =============================================================================
print("\n" + "=" * 70)
print("SETTING UP MICROPHONE")
print("=" * 70)

pa = pyaudio.PyAudio()

try:
    stream = pa.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=SAMPLE_RATE,
        input=True,
        frames_per_buffer=CHUNK_SIZE
    )
    print("✓ Microphone opened")
except Exception as e:
    print(f"✗ Error opening microphone: {e}")
    pa.terminate()
    exit(1)

# Create recognizer with OPTIMIZED settings
recognizer = vosk.KaldiRecognizer(speech_model, SAMPLE_RATE, speaker_model)
recognizer.SetWords(True)  # Word-level timestamps

# CRITICAL OPTIMIZATION: Make Vosk more aggressive about finishing phrases
# This makes it split on shorter pauses
recognizer.SetMaxAlternatives(0)  # Don't waste time on alternatives
recognizer.SetPartialWords(True)  # Get partial word results faster

print("✓ Recognizer created with aggressive settings")

# =============================================================================
# SCOREBOARD
# =============================================================================
mike_swear_count = 0
james_swear_count = 0
total_swears = 0

# Track last few detections to avoid duplicates
recent_detections = []  # Store (word, timestamp) tuples


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def identify_speaker(speaker_vector):
    """Identify speaker from voice vector."""
    distance_to_mike = cosine(speaker_vector, mike_signature)
    distance_to_james = cosine(speaker_vector, james_signature)

    if distance_to_mike < distance_to_james:
        confidence = 1 - distance_to_mike
        return "MIKE", confidence
    else:
        confidence = 1 - distance_to_james
        return "JAMES", confidence


def is_duplicate_detection(word, timestamp):
    """Check if we already detected this word in last 2 seconds."""
    global recent_detections

    # Clean old detections (older than 2 seconds)
    current_time = datetime.now()
    recent_detections = [
        (w, t) for w, t in recent_detections
        if (current_time - t).total_seconds() < 2
    ]

    # Check if this word was just detected
    for recent_word, recent_time in recent_detections:
        if recent_word == word and (timestamp - recent_time).total_seconds() < 1:
            return True

    return False


def process_result(result, is_partial=False):
    """
    Process Vosk result and detect swears.
    Works with both full and partial results.
    """
    global mike_swear_count, james_swear_count, total_swears, recent_detections

    # For partial results, just check the partial text
    if is_partial:
        if "partial" not in result:
            return

        partial_text = result["partial"].lower()
        words = partial_text.split()

        # Check each word
        for word in words:
            clean_word = word.strip('.,!?;:')
            if clean_word in SWEAR_WORDS:
                timestamp = datetime.now()

                # Avoid duplicates
                if is_duplicate_detection(clean_word, timestamp):
                    continue

                # Can't identify speaker in partial results
                # But we can at least flag the swear
                print(
                    f"⚡ [{timestamp.strftime('%H:%M:%S')}] SWEAR DETECTED: '{clean_word}' (speaker unknown - waiting...)")

    else:
        # Full result - has speaker info
        if "result" not in result or "spk" not in result:
            return

        speaker_vector = result["spk"]
        words = result["result"]

        # Identify speaker for this phrase
        speaker, confidence = identify_speaker(speaker_vector)

        # Check each word
        for word_obj in words:
            word = word_obj["word"].lower()

            if word in SWEAR_WORDS:
                timestamp = datetime.now()

                # Avoid duplicates
                if is_duplicate_detection(word, timestamp):
                    continue

                # Record this detection
                recent_detections.append((word, timestamp))

                # Update scoreboard
                total_swears += 1
                if speaker == "MIKE":
                    mike_swear_count += 1
                else:
                    james_swear_count += 1

                # Display
                print(f"🚨 [{timestamp.strftime('%H:%M:%S')}] {speaker}: '{word}' ({confidence * 100:.0f}%)")
                print(f"   MIKE={mike_swear_count} | JAMES={james_swear_count}")
                print("-" * 70)


# =============================================================================
# MAIN DETECTION LOOP - MAXIMUM RESPONSIVENESS
# =============================================================================
print("\n" + "=" * 70)
print("STARTING MAXIMUM RESPONSIVENESS MODE")
print("=" * 70)
print("\n🎮 FIFA SWEAR DETECTOR - HYPER MODE!")
print("Detecting swears with maximum speed...")
print("Smaller chunks + partial results = faster detection")
print("Press Ctrl+C to stop\n")
print("-" * 70)

try:
    while True:
        # Read SMALLER audio chunk for faster processing
        audio_data = stream.read(CHUNK_SIZE, exception_on_overflow=False)

        # Try to process as full result first
        if recognizer.AcceptWaveform(audio_data):
            # Got a complete phrase
            result = json.loads(recognizer.Result())
            process_result(result, is_partial=False)

        else:
            # No complete phrase yet - check partial results
            # This gives us early warning of swears
            partial = json.loads(recognizer.PartialResult())
            process_result(partial, is_partial=True)

except KeyboardInterrupt:
    print("\n\n⚠️  Detection stopped")

# =============================================================================
# CLEANUP
# =============================================================================
print("\n" + "=" * 70)
print("FINAL RESULTS")
print("=" * 70)

stream.stop_stream()
stream.close()
pa.terminate()

print(f"\nFinal scoreboard:")
print(f"  MIKE:  {mike_swear_count} swears")
print(f"  JAMES: {james_swear_count} swears")
print(f"  TOTAL: {total_swears} swears")

if mike_swear_count > james_swear_count:
    print(f"\n🏆 MIKE is the swear champion! 😬")
elif james_swear_count > mike_swear_count:
    print(f"\n🏆 JAMES is the swear champion! 😬")
else:
    print(f"\n🤝 Perfect tie!")

print("\n" + "=" * 70)


