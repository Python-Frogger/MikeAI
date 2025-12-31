import openwakeword
from openwakeword.model import Model
import pyaudio
import numpy as np
import time

print("=" * 70)
print("OPENWAKEWORD WINDOWS TEST - FIXED")
print("=" * 70)

# Step 1: Download models
print("\n[1/5] Downloading models...")
openwakeword.utils.download_models()

# Step 2: Initialize with EXPLICIT ONNX framework for Windows
print("\n[2/5] Loading model with ONNX inference (Windows compatible)...")
model = Model(
    wakeword_models=["hey_jarvis"],
    inference_framework='onnx'  # CRITICAL for Windows!
)
print(f"✓ Model loaded with ONNX framework")
print(f"  Available models: {list(model.models.keys())}")

# Step 3: Setup audio
print("\n[3/5] Setting up microphone...")
RATE = 16000
CHUNK = 1280  # 80ms at 16kHz
FORMAT = pyaudio.paInt16
CHANNELS = 1

pa = pyaudio.PyAudio()
stream = pa.open(
    rate=RATE,
    channels=CHANNELS,
    format=FORMAT,
    input=True,
    frames_per_buffer=CHUNK
)
print("✓ Microphone ready")

# Step 4: Listen
print("\n[4/5] Listening for 20 seconds...")
print("=" * 70)
print("🎤 SAY: 'HEY JARVIS' clearly!")
print("=" * 70)

THRESHOLD = 0.5
start_time = time.time()
frame_count = 0
max_level = 0
max_score = 0
detection_count = 0

try:
    while time.time() - start_time < 20:
        # Read audio
        audio_data = stream.read(CHUNK, exception_on_overflow=False)
        audio_array = np.frombuffer(audio_data, dtype=np.int16)
        audio_float = audio_array.astype(np.float32) / 32768.0

        # Check level
        level = np.max(np.abs(audio_float))
        max_level = max(max_level, level)

        # Get prediction
        prediction = model.predict(audio_float)
        score = prediction.get("hey_jarvis", 0.0)
        max_score = max(max_score, score)

        frame_count += 1

        # Print every 10 frames (~0.8s)
        if frame_count % 10 == 0:
            elapsed = time.time() - start_time
            print(f"Time: {elapsed:5.1f}s | Level: {level:.4f} | Score: {score:.6f}")

        # Detection!
        if score > THRESHOLD:
            detection_count += 1
            print("\n" + "🎉" * 35)
            print(f"✓✓✓ DETECTED 'HEY JARVIS' (#{detection_count}) ✓✓✓")
            print("🎉" * 35 + "\n")
            time.sleep(0.5)

except KeyboardInterrupt:
    print("\n\nStopped by user")

finally:
    stream.stop_stream()
    stream.close()
    pa.terminate()

# Step 5: Results
print("\n[5/5] RESULTS")
print("=" * 70)
print(f"Duration: {time.time() - start_time:.1f}s")
print(f"Frames processed: {frame_count}")
print(f"Detections: {detection_count}")
print(f"Max audio level: {max_level:.4f}")
print(f"Max prediction score: {max_score:.6f}")
print(f"Threshold: {THRESHOLD}")

print("\n📊 COMPARISON:")
print(f"  Porcupine:     ✓ WORKING (7 detections)")
print(f"  OpenWakeWord:  {'✓ WORKING' if detection_count > 0 else '✗ NOT WORKING'} ({detection_count} detections)")

if detection_count > 0:
    print("\n✓✓✓ SUCCESS! OpenWakeWord now works on Windows!")
elif max_score > 0.01:
    print(f"\n⚠️  Model responding but scores too low (max: {max_score:.6f})")
    print("   Try speaking more clearly or adjusting threshold")
else:
    print("\n✗ Model still not responding properly")
    print("  Scores are too low - possible model/framework mismatch")

print("=" * 70)