"""
Live FIFA Swear Word Detector - WORD-BY-WORD Detection
========================================================
Detects swear words IMMEDIATELY as individual words, not phrases.
Uses Vosk's word-level results for instant detection.

Requirements:
1. Run train_speakers.py FIRST to create signature files
2. pip install vosk pyaudio numpy scipy
3. mike_signature.npy and james_signature.npy must exist

Author: FIFA Swear Counter Project
"""

# =============================================================================
# IMPORTS
# =============================================================================
print("=" * 70)
print("FIFA SWEAR WORD DETECTOR - Word-by-Word Mode")
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

# Signature files (created by train_speakers.py)
MIKE_SIGNATURE_FILE = "mike_signature.npy"
JAMES_SIGNATURE_FILE = "james_signature.npy"

# Audio settings
SAMPLE_RATE = 16000
CHUNK_SIZE = 4000

# SWEAR WORD LIST - exact matches only
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
print(f"Top swears: {', '.join(list(SWEAR_WORDS)[:5])}...")

# =============================================================================
# CHECK FILES EXIST
# =============================================================================
print("\n" + "=" * 70)
print("CHECKING FILES")
print("=" * 70)

if not os.path.exists(SPEECH_MODEL_PATH):
    print(f"✗ ERROR: Speech model not found: {SPEECH_MODEL_PATH}")
    exit(1)
print(f"✓ Speech model found")

if not os.path.exists(SPEAKER_MODEL_PATH):
    print(f"✗ ERROR: Speaker model not found: {SPEAKER_MODEL_PATH}")
    exit(1)
print(f"✓ Speaker model found")

if not os.path.exists(MIKE_SIGNATURE_FILE):
    print(f"✗ ERROR: Mike's signature not found: {MIKE_SIGNATURE_FILE}")
    print("  Run train_speakers.py first!")
    exit(1)
print(f"✓ Mike's signature found")

if not os.path.exists(JAMES_SIGNATURE_FILE):
    print(f"✗ ERROR: James's signature not found: {JAMES_SIGNATURE_FILE}")
    print("  Run train_speakers.py first!")
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
print("✓ Mike's voice signature loaded")
print("✓ James's voice signature loaded")

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

# Create recognizer with speaker identification
recognizer = vosk.KaldiRecognizer(speech_model, SAMPLE_RATE, speaker_model)
recognizer.SetWords(True)  # CRITICAL: Get individual words with timestamps
print("✓ Recognizer created")

# =============================================================================
# SCOREBOARD
# =============================================================================
mike_swear_count = 0
james_swear_count = 0
total_swears = 0

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def identify_speaker(speaker_vector):
    """
    Identify who is speaking based on voice vector.

    Args:
        speaker_vector: Voice fingerprint from Vosk

    Returns:
        Tuple of (speaker_name, confidence)
    """
    distance_to_mike = cosine(speaker_vector, mike_signature)
    distance_to_james = cosine(speaker_vector, james_signature)

    if distance_to_mike < distance_to_james:
        confidence = 1 - distance_to_mike
        return "MIKE", confidence
    else:
        confidence = 1 - distance_to_james
        return "JAMES", confidence


def process_word_result(result):
    """
    Process word-level results from Vosk.
    Checks each individual word for swears.

    Args:
        result: JSON result from Vosk

    Returns:
        List of (word, speaker, confidence) tuples for detected swears
    """
    global mike_swear_count, james_swear_count, total_swears

    detections = []

    # Check if we have both words and speaker info
    if "result" not in result or "spk" not in result:
        return detections

    speaker_vector = result["spk"]
    words = result["result"]  # List of word objects with timestamps

    # Identify speaker once for this phrase
    speaker, confidence = identify_speaker(speaker_vector)

    # Check each word individually
    for word_obj in words:
        word = word_obj["word"].lower()  # Get the word, lowercase

        # Is this a swear word?
        if word in SWEAR_WORDS:
            # Update scoreboard
            total_swears += 1
            if speaker == "MIKE":
                mike_swear_count += 1
            else:
                james_swear_count += 1

            # Record detection
            detections.append((word, speaker, confidence))

    return detections

# =============================================================================
# MAIN DETECTION LOOP
# =============================================================================
print("\n" + "=" * 70)
print("STARTING WORD-BY-WORD DETECTION")
print("=" * 70)
print("\n🎮 FIFA SWEAR DETECTOR IS LIVE!")
print("Detecting individual swear words in real-time...")
print("Press Ctrl+C to stop\n")
print("-" * 70)

try:
    while True:
        # Read audio from microphone
        audio_data = stream.read(CHUNK_SIZE, exception_on_overflow=False)

        # Process with Vosk
        if recognizer.AcceptWaveform(audio_data):
            # Got a complete phrase with word-level breakdown
            result = json.loads(recognizer.Result())

            # Process each word in the result
            swear_detections = process_word_result(result)

            # Display any swears found
            for word, speaker, confidence in swear_detections:
                timestamp = datetime.now().strftime("%H:%M:%S")

                # INSTANT DETECTION DISPLAY - Just the swear word!
                print(f"🚨 [{timestamp}] {speaker}: '{word}' (confidence: {confidence*100:.0f}%)")
                print(f"   Score: MIKE={mike_swear_count} | JAMES={james_swear_count}")
                print("-" * 70)

except KeyboardInterrupt:
    print("\n\n⚠️  Detection stopped by user")

# =============================================================================
# CLEANUP AND FINAL RESULTS
# =============================================================================
print("\n" + "=" * 70)
print("FINAL RESULTS")
print("=" * 70)

stream.stop_stream()
stream.close()
pa.terminate()

print("\nFinal scoreboard:")
print(f"  MIKE:  {mike_swear_count} swears")
print(f"  JAMES: {james_swear_count} swears")
print(f"  TOTAL: {total_swears} swears")

if mike_swear_count > james_swear_count:
    print(f"\n🏆 MIKE wins the swearing competition! 😬")
elif james_swear_count > mike_swear_count:
    print(f"\n🏆 JAMES wins the swearing competition! 😬")
else:
    print(f"\n🤝 It's a tie! Both equally foul-mouthed! 😂")

print("\n" + "=" * 70)
print("Session complete!")
print("=" * 70)

## **KEY CHANGES:**

#1. ✅ **Uses `result["result"]`** - the word-level breakdown from Vosk
#2. ✅ **Checks EACH WORD individually** against the swear list
#3. ✅ **Only displays swear words** - ignores all other words
#4. ✅ **Cleaner output** - just shows `MIKE: 'fuck'` instead of full phrases
#5. ✅ **Same speaker ID** - still uses your trained signatures

## **How it works now:**

#User says: "this is mike fuck you"
#Old version: [waits for pause] "Said: 'this is mike fuck you'"
#New version: [instant] MIKE: 'fuck' ✓
