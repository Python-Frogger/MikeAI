"""
Simple Vosk Speech Detection - A-Level CS Teaching Example
============================================================
This script detects ANY spoken words using Vosk speech recognition.
Perfect for learning how speech detection works!

Before running, you need to:
1. Install packages: pip install vosk pyaudio
2. Download a small Vosk model from: https://alphacephei.com/vosk/models
   Recommended: vosk-model-small-en-us-0.15 (40MB)
3. Extract the model folder next to this script

Author: Teaching example for GCSE/A-Level Computer Science
"""

# =============================================================================
# STEP 1: IMPORT LIBRARIES
# =============================================================================
print("=" * 70)
print("STEP 1: IMPORTING LIBRARIES")
print("=" * 70)

try:
    import pyaudio  # For capturing microphone audio

    print("✓ PyAudio imported - for recording from microphone")
except ImportError:
    print("✗ PyAudio not found! Install with: pip install pyaudio")
    exit(1)

try:
    import vosk  # For speech recognition

    print("✓ Vosk imported - for speech recognition")
except ImportError:
    print("✗ Vosk not found! Install with: pip install vosk")
    exit(1)

try:
    import json  # For parsing Vosk's JSON responses

    print("✓ JSON imported - for reading Vosk results")
except ImportError:
    print("✗ JSON not found (should be built-in to Python!)")
    exit(1)

try:
    import time  # For timing our 10-second test

    print("✓ Time imported - for timing")
except ImportError:
    print("✗ Time not found (should be built-in to Python!)")
    exit(1)

try:
    import numpy as np  # For calculating audio volume levels

    print("✓ NumPy imported - for audio calculations")
except ImportError:
    print("✗ NumPy not found! Install with: pip install numpy")
    exit(1)

import os  # For checking if model folder exists

print("✓ OS imported - for file operations")

print("\n✓✓✓ All libraries imported successfully!\n")
time.sleep(1)  # Pause so student can read

# =============================================================================
# STEP 2: LOAD THE VOSK MODEL
# =============================================================================
print("=" * 70)
print("STEP 2: LOADING VOSK MODEL")
print("=" * 70)

# Tell user what we're looking for
MODEL_PATH = "vosk-model-small-en-us-0.15"  # Change this to match YOUR model folder name

print(f"Looking for model in: {MODEL_PATH}")

# Check if the model exists
if not os.path.exists(MODEL_PATH):
    print("\n✗ ERROR: Model not found!")
    print("\nPlease download a model from: https://alphacephei.com/vosk/models")
    print("Recommended: vosk-model-small-en-us-0.15 (40MB)")
    print(f"Extract it to the same folder as this script.")
    print(f"Make sure the folder is named: {MODEL_PATH}")
    exit(1)

print("✓ Model folder found!")

# Load the model (this takes a few seconds)
print("Loading model... (this takes a few seconds)")
try:
    model = vosk.Model(MODEL_PATH)
    print("✓✓✓ Model loaded successfully!\n")
except Exception as e:
    print(f"✗ Error loading model: {e}")
    exit(1)

time.sleep(1)

# =============================================================================
# STEP 3: SET UP THE MICROPHONE
# =============================================================================
print("=" * 70)
print("STEP 3: SETTING UP MICROPHONE")
print("=" * 70)

# Vosk works best with 16000 Hz (16 kHz) sample rate
SAMPLE_RATE = 16000  # Number of audio samples per second
CHUNK_SIZE = 4000  # How many samples to read at once (0.25 seconds worth)

print(f"Sample rate: {SAMPLE_RATE} Hz (16 kHz)")
print(f"Chunk size: {CHUNK_SIZE} samples (~0.25 seconds)")

# Create a PyAudio object (this manages audio)
pa = pyaudio.PyAudio()

# Open the microphone stream
try:
    stream = pa.open(
        format=pyaudio.paInt16,  # 16-bit audio (standard quality)
        channels=1,  # Mono audio (1 microphone)
        rate=SAMPLE_RATE,  # 16000 samples per second
        input=True,  # We're READing from mic (not playing audio)
        frames_per_buffer=CHUNK_SIZE  # Read this many samples at a time
    )
    print("✓ Microphone opened successfully!")
except Exception as e:
    print(f"✗ Error opening microphone: {e}")
    print("\nTroubleshooting:")
    print("- Check your microphone is plugged in")
    print("- Check Windows microphone permissions")
    print("- Try restarting the script")
    pa.terminate()
    exit(1)

# Create the recognizer (this is what detects speech)
recognizer = vosk.KaldiRecognizer(model, SAMPLE_RATE)
recognizer.SetMaxAlternatives(0)  # We only want the best guess
recognizer.SetWords(True)  # We want individual words, not just the full phrase

print("✓ Speech recognizer created!")
print("\n✓✓✓ Microphone setup complete!\n")
time.sleep(1)

# =============================================================================
# STEP 4: TEST MICROPHONE VOLUME
# =============================================================================
print("=" * 70)
print("STEP 4: TESTING MICROPHONE (3 seconds)")
print("=" * 70)
print("Speak now to test your microphone!")
print("(The volume bar should jump when you talk)\n")

# Test for 3 seconds
for i in range(3):
    # Read audio from microphone
    audio_data = stream.read(CHUNK_SIZE, exception_on_overflow=False)

    # Convert to numpy array so we can calculate volume
    audio_array = np.frombuffer(audio_data, dtype=np.int16)

    # Calculate volume (how loud the audio is)
    # We use "max absolute value" - the loudest point in this chunk
    volume = np.max(np.abs(audio_array))

    # Create a visual bar (like a volume meter)
    bar_length = int(volume / 1000)  # Scale down so it fits on screen
    bar = "█" * min(bar_length, 50)  # Cap at 50 characters

    print(f"Second {i + 1}/3 | Volume: {volume:5d} | {bar}")
    time.sleep(1)

print("\n✓✓✓ Microphone test complete!")
print("(If you saw the bars move when speaking, your mic works!)\n")
time.sleep(1)

# =============================================================================
# STEP 5: LISTEN FOR SPEECH (10 SECONDS)
# =============================================================================
print("=" * 70)
print("STEP 5: LISTENING FOR SPEECH (10 seconds)")
print("=" * 70)
print("Say ANYTHING - Vosk will detect any English words!")
print("Try saying: hello, test, microphone, computer, stop, etc.")
print("The easier words are: yes, no, hello, stop, test")
print("\nStarting in 2 seconds...\n")
time.sleep(2)

# Variables to track what we've detected
start_time = time.time()
last_print_time = start_time
detection_count = 0
words_detected = []

print("🎤 LISTENING NOW! (Speak clearly)")
print("=" * 70)

# Main listening loop - runs for 10 seconds
try:
    while time.time() - start_time < 10:
        # Read audio from the microphone
        audio_data = stream.read(CHUNK_SIZE, exception_on_overflow=False)

        # Calculate current volume (for diagnostics)
        audio_array = np.frombuffer(audio_data, dtype=np.int16)
        volume = np.max(np.abs(audio_array))

        # Send audio to Vosk for recognition
        if recognizer.AcceptWaveform(audio_data):
            # Vosk has completed recognizing a phrase!
            result = json.loads(recognizer.Result())
            text = result.get("text", "")

            if text:  # If Vosk detected any words
                detection_count += 1
                words_detected.append(text)
                print(f"\n🎉 DETECTED #{detection_count}: '{text}'")
                print(f"   Volume: {volume}")

        # Print a diagnostic every second
        current_time = time.time()
        if current_time - last_print_time >= 1.0:
            elapsed = int(current_time - start_time)
            remaining = 10 - elapsed

            # Volume bar
            bar_length = int(volume / 1000)
            bar = "█" * min(bar_length, 30)

            print(f"[{elapsed}s] Volume: {volume:5d} | {bar} | {remaining}s remaining")
            last_print_time = current_time

except KeyboardInterrupt:
    print("\n\n⚠️  Stopped by user (Ctrl+C pressed)")

# =============================================================================
# STEP 6: CLEANUP AND RESULTS
# =============================================================================
print("\n" + "=" * 70)
print("STEP 6: CLEANUP AND RESULTS")
print("=" * 70)

# Get any final partial results
final_result = json.loads(recognizer.FinalResult())
final_text = final_result.get("text", "")
if final_text and final_text not in words_detected:
    words_detected.append(final_text)
    detection_count += 1
    print(f"Final detection: '{final_text}'")

# Close the microphone
stream.stop_stream()
stream.close()
pa.terminate()
print("✓ Microphone closed")

# Print summary
print("\n" + "=" * 70)
print("SUMMARY OF RESULTS")
print("=" * 70)
print(f"Total time: 10 seconds")
print(f"Total detections: {detection_count}")
print(f"\nWords/phrases detected:")
if words_detected:
    for i, phrase in enumerate(words_detected, 1):
        print(f"  {i}. '{phrase}'")
else:
    print("  (none - try speaking louder or more clearly)")

print("\n" + "=" * 70)
print("WHAT TO TRY NEXT:")
print("=" * 70)
print("Easy words to test with:")
print("  - Single syllable: yes, no, stop, go, test, one, two")
print("  - Common words: hello, computer, microphone, working")
print("  - For your project: Try saying swear words!")
print("\nFor swear detection, modify this code to:")
print("  1. Check if detected text contains 'fuck', 'shit', etc.")
print("  2. Play a sound or show a warning")
print("  3. Log the detection with timestamp")
print("=" * 70)