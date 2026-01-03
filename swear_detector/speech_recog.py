"""
Speaker Identification System - Step 1: Basic Detection
========================================================
This script detects WHO is speaking using Vosk speaker recognition.
Goal: Display "MIKE" or "JAMES" when each person talks.

Requirements:
1. pip install vosk pyaudio numpy scipy
2. Download vosk-model-en-us-0.22 (speech recognition)
3. Download vosk-model-spk-0.4 (speaker identification)
   From: https://alphacephei.com/vosk/models

Author: CS Teaching Example - FIFA Swear Counter Project
"""

# =============================================================================
# STEP 1: IMPORT LIBRARIES
# =============================================================================
print("=" * 70)
print("SPEAKER IDENTIFICATION - STEP 1: BASIC DETECTION")
print("=" * 70)
print("\nImporting libraries...\n")

import pyaudio  # For microphone input
import vosk  # For speech recognition AND speaker identification
import json  # For parsing Vosk results
import numpy as np  # For audio calculations
import os  # For checking files exist
import time  # For timing
from scipy.spatial.distance import cosine  # For comparing speaker "fingerprints"

print("✓ All libraries imported!\n")

# =============================================================================
# STEP 2: CONFIGURATION
# =============================================================================
print("=" * 70)
print("STEP 2: CONFIGURATION")
print("=" * 70)

# Model paths - CHANGE THESE to match your setup
SPEECH_MODEL_PATH = "vosk-model-en-us-0.22"  # Speech recognition model
SPEAKER_MODEL_PATH = "vosk-model-spk-0.4"  # Speaker identification model

# Audio settings
SAMPLE_RATE = 16000  # 16 kHz - required by Vosk
CHUNK_SIZE = 4000  # Read 0.25 seconds at a time

print(f"Speech model: {SPEECH_MODEL_PATH}")
print(f"Speaker model: {SPEAKER_MODEL_PATH}")
print(f"Sample rate: {SAMPLE_RATE} Hz")

# =============================================================================
# STEP 3: CHECK MODELS EXIST
# =============================================================================
print("\n" + "=" * 70)
print("STEP 3: CHECKING MODELS")
print("=" * 70)

# Check speech model
if not os.path.exists(SPEECH_MODEL_PATH):
    print(f"\n✗ ERROR: Speech model not found at '{SPEECH_MODEL_PATH}'")
    print("Download from: https://alphacephei.com/vosk/models")
    exit(1)
print(f"✓ Speech model found")

# Check speaker model
if not os.path.exists(SPEAKER_MODEL_PATH):
    print(f"\n✗ ERROR: Speaker model not found at '{SPEAKER_MODEL_PATH}'")
    print("Download 'vosk-model-spk-0.4' from: https://alphacephei.com/vosk/models")
    print("Extract it to the same folder as this script")
    exit(1)
print(f"✓ Speaker model found")

# =============================================================================
# STEP 4: LOAD MODELS
# =============================================================================
print("\n" + "=" * 70)
print("STEP 4: LOADING MODELS")
print("=" * 70)

print("Loading speech model... (may take a few seconds)")
speech_model = vosk.Model(SPEECH_MODEL_PATH)
print("✓ Speech model loaded")

print("Loading speaker model...")
speaker_model = vosk.SpkModel(SPEAKER_MODEL_PATH)  # Special speaker model!
print("✓ Speaker model loaded")

# =============================================================================
# STEP 5: SET UP MICROPHONE
# =============================================================================
print("\n" + "=" * 70)
print("STEP 5: SETTING UP MICROPHONE")
print("=" * 70)

# Create PyAudio interface
pa = pyaudio.PyAudio()

# Open microphone stream
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

# Create recognizer WITH speaker recognition enabled
recognizer = vosk.KaldiRecognizer(speech_model, SAMPLE_RATE, speaker_model)
recognizer.SetWords(True)  # Get individual words
print("✓ Recognizer created with speaker identification enabled")

# =============================================================================
# STEP 6: COLLECT SPEAKER SAMPLES (ENROLLMENT)
# =============================================================================
print("\n" + "=" * 70)
print("STEP 6: SPEAKER ENROLLMENT")
print("=" * 70)
print("\nWe need to learn what Mike and James sound like!")
print("Each person will speak for 5 seconds.\n")

# Storage for speaker "fingerprints" (called vectors)
mike_vectors = []  # Mike's voice fingerprints
james_vectors = []  # James's voice fingerprints


def collect_speaker_sample(speaker_name, duration=10):
    """
    Collect voice samples from one speaker.

    speaker_name: "MIKE" or "JAMES"
    duration: how many seconds to record

    Returns: list of speaker vectors (voice fingerprints)
    """
    print(f"\n--- {speaker_name}'S TURN ---")
    print(f"{speaker_name}: Please speak continuously for {duration} seconds")
    print("IMPORTANT: Keep talking non-stop! Count numbers, recite alphabet, anything!")
    print("The more you talk, the better the detection will be.")
    print("\nStarting in 3 seconds...")
    time.sleep(3)

    print(f"\n🎤 {speaker_name} - SPEAK NOW! KEEP TALKING!")
    print("=" * 50)

    vectors = []  # Store voice fingerprints here
    partial_results = []  # Track partial results too
    start_time = time.time()
    last_update = start_time

    # Record for specified duration
    while time.time() - start_time < duration:
        # Read audio chunk
        audio_data = stream.read(CHUNK_SIZE, exception_on_overflow=False)

        # Process with Vosk - check BOTH full results and partial results
        if recognizer.AcceptWaveform(audio_data):
            # Full result (end of phrase)
            result = json.loads(recognizer.Result())

            # Check if we got a speaker vector
            if "spk" in result:
                vector = result["spk"]  # This is the "voice fingerprint"!
                vectors.append(vector)
                print(f"  ✓ Got voice sample #{len(vectors)}")  # Show progress!

                # Show what was said
                text = result.get("text", "")
                if text:
                    print(f"    Said: '{text}'")
            else:
                # Vosk recognized speech but didn't give us a speaker vector
                text = result.get("text", "")
                if text:
                    print(f"  ⚠ Speech detected but no voice sample: '{text}'")
        else:
            # Partial result (Vosk is still processing)
            partial = json.loads(recognizer.PartialResult())
            partial_text = partial.get("partial", "")
            if partial_text:
                partial_results.append(partial_text)

        # Show time remaining every second
        current_time = time.time()
        if current_time - last_update >= 1.0:
            elapsed = int(current_time - start_time)
            remaining = duration - elapsed
            print(f"  ⏱️  {remaining}s remaining - Keep talking! Got {len(vectors)} samples so far")
            last_update = current_time

    # Get final result
    final = json.loads(recognizer.FinalResult())
    if "spk" in final:
        vectors.append(final["spk"])
        print(f"  ✓ Got final voice sample #{len(vectors)}")

    print(f"\n✓ Collected {len(vectors)} voice samples from {speaker_name}")

    # Diagnostic info
    if len(vectors) == 0:
        print(f"  ⚠️  WARNING: No speaker vectors collected!")
        print(f"     Speech was detected: {len(partial_results) > 0}")
        print(f"     This might mean:")
        print(f"       - Speech segments were too short")
        print(f"       - Need to speak louder/clearer")
        print(f"       - Microphone issue")
    elif len(vectors) < 3:
        print(f"  ⚠️  WARNING: Only got {len(vectors)} samples (ideally want 5+)")
        print(f"     Try speaking more continuously next time")

    return vectors


# Collect samples from both speakers
print("\nFirst, we'll record MIKE's voice:")
mike_vectors = collect_speaker_sample("MIKE", duration=10)

print("\n\nNow, we'll record JAMES's voice:")
james_vectors = collect_speaker_sample("JAMES", duration=10)

# Calculate average vector for each person
# (Average of all their voice fingerprints = their "signature")
print("\n" + "=" * 70)
print("CREATING VOICE SIGNATURES")
print("=" * 70)

if len(mike_vectors) == 0:
    print("\n✗ CRITICAL ERROR: No voice samples from MIKE!")
    print("\nTROUBLESHOOTING:")
    print("  1. Speak CONTINUOUSLY - don't pause between words")
    print("  2. Speak LOUDER - make sure mic can hear you clearly")
    print("  3. Get closer to the microphone")
    print("  4. Try counting: 'one two three four five...' non-stop")
    stream.close()
    pa.terminate()
    exit(1)

if len(james_vectors) == 0:
    print("\n✗ CRITICAL ERROR: No voice samples from JAMES!")
    print("\nTROUBLESHOOTING:")
    print("  1. Speak CONTINUOUSLY - don't pause between words")
    print("  2. Speak LOUDER - make sure mic can hear you clearly")
    print("  3. Get closer to the microphone")
    print("  4. Try counting: 'one two three four five...' non-stop")
    stream.close()
    pa.terminate()
    exit(1)

# Calculate signatures
mike_signature = np.mean(mike_vectors, axis=0)
james_signature = np.mean(james_vectors, axis=0)

print(f"✓ MIKE's signature: {len(mike_vectors)} samples")
print(f"✓ JAMES's signature: {len(james_vectors)} samples")

# Warn if we have very few samples
if len(mike_vectors) < 3:
    print(f"⚠️  WARNING: Mike only has {len(mike_vectors)} samples (ideally 5+)")
    print("   Accuracy might be lower - consider re-running")

if len(james_vectors) < 3:
    print(f"⚠️  WARNING: James only has {len(james_vectors)} samples (ideally 5+)")
    print("   Accuracy might be lower - consider re-running")
# =============================================================================
# STEP 7: TEST SPEAKER IDENTIFICATION (20 seconds)
# =============================================================================
print("\n" + "=" * 70)
print("STEP 7: TESTING SPEAKER IDENTIFICATION")
print("=" * 70)
print("\nNow both of you can speak!")
print("The system will try to identify who is talking.")
print("\nTest for 20 seconds - take turns speaking")
print("Starting in 3 seconds...\n")
time.sleep(3)

print("🎤 LISTENING - SPEAK NOW!")
print("=" * 70)

start_time = time.time()
detection_count = 0

try:
    while time.time() - start_time < 20:
        # Read audio
        audio_data = stream.read(CHUNK_SIZE, exception_on_overflow=False)

        # Process with Vosk
        if recognizer.AcceptWaveform(audio_data):
            result = json.loads(recognizer.Result())

            # Check if we got speech with speaker info
            if "spk" in result and result.get("text"):
                detection_count += 1
                current_vector = result["spk"]  # Current speaker's fingerprint
                text = result["text"]

                # Compare current voice to Mike and James signatures
                # Using cosine distance (0 = identical, 2 = completely different)
                distance_to_mike = cosine(current_vector, mike_signature)
                distance_to_james = cosine(current_vector, james_signature)

                # Who is closer?
                if distance_to_mike < distance_to_james:
                    speaker = "MIKE"
                    confidence = 1 - distance_to_mike  # Convert distance to confidence
                else:
                    speaker = "JAMES"
                    confidence = 1 - distance_to_james

                # Display result
                print(f"\n🎯 {speaker} said: '{text}'")
                print(f"   Confidence: {confidence * 100:.1f}%")
                print(f"   Distance to Mike: {distance_to_mike:.3f}")
                print(f"   Distance to James: {distance_to_james:.3f}")

        # Show elapsed time
        elapsed = int(time.time() - start_time)
        remaining = 20 - elapsed
        if remaining > 0 and elapsed % 5 == 0:  # Every 5 seconds
            print(f"\n⏱️  {remaining} seconds remaining...")

except KeyboardInterrupt:
    print("\n\n⚠️  Stopped by user")

# =============================================================================
# STEP 8: CLEANUP
# =============================================================================
print("\n" + "=" * 70)
print("STEP 8: CLEANUP AND SUMMARY")
print("=" * 70)

stream.stop_stream()
stream.close()
pa.terminate()
print("✓ Microphone closed")

print(f"\nTotal detections: {detection_count}")
print("\n" + "=" * 70)
print("NEXT STEPS:")
print("=" * 70)
print("If the speaker identification worked well:")
print("  1. We'll add swear word detection")
print("  2. Create a scoreboard")
print("  3. Add voice-activated reset")
print("\nIf it didn't work well:")
print("  - Try speaking more clearly during enrollment")
print("  - Make sure microphone is close to speakers")
print("  - Re-run the script to re-enroll")
print("=" * 70)