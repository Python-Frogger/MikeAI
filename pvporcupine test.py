import pvporcupine
import pyaudio
import struct
import time

print("=" * 70)
print("PORCUPINE 'TERMINATOR' DETECTION TEST")
print("=" * 70)

ACCESS_KEY = 'LFvu9jHaT12Y/ovdTMXg8CcJobGh75cZOFugv0cqacRFsJBSPedztA=='

# Initialize Porcupine
print("\n[1/3] Initializing Porcupine...")
porcupine = pvporcupine.create(
    access_key=ACCESS_KEY,
    keywords=['terminator']
)
print(f"✓ Porcupine ready")
print(f"  Sample rate: {porcupine.sample_rate} Hz")
print(f"  Frame length: {porcupine.frame_length} samples")

# Setup microphone
print("\n[2/3] Opening microphone...")
pa = pyaudio.PyAudio()
stream = pa.open(
    rate=porcupine.sample_rate,
    channels=1,
    format=pyaudio.paInt16,
    input=True,
    frames_per_buffer=porcupine.frame_length
)
print("✓ Microphone ready")

# Listen
print("\n[3/3] Listening for 20 seconds...")
print("=" * 70)
print("🎤 SAY: 'TERMINATOR' clearly!")
print("=" * 70)

start_time = time.time()
frame_count = 0
max_level = 0
detection_count = 0

try:
    while time.time() - start_time < 20:
        # Read audio
        pcm = stream.read(porcupine.frame_length, exception_on_overflow=False)
        pcm_unpacked = struct.unpack_from("h" * porcupine.frame_length, pcm)

        # Check level
        level = max(abs(x) for x in pcm_unpacked) / 32768.0
        max_level = max(max_level, level)

        frame_count += 1

        # Show status every ~0.5 seconds
        if frame_count % 25 == 0:
            print(f"Time: {time.time() - start_time:5.1f}s | Level: {level:.4f} | Listening...")

        # Detect wake word
        keyword_index = porcupine.process(pcm_unpacked)

        if keyword_index >= 0:
            detection_count += 1
            print("\n" + "🎉" * 35)
            print(f"✓✓✓ DETECTED 'TERMINATOR' (#{detection_count}) ✓✓✓")
            print("🎉" * 35 + "\n")
            time.sleep(0.5)

except KeyboardInterrupt:
    print("\n\nStopped by user")

finally:
    stream.stop_stream()
    stream.close()
    pa.terminate()
    porcupine.delete()

# Results
print("\n" + "=" * 70)
print("RESULTS")
print("=" * 70)
print(f"Duration: {time.time() - start_time:.1f}s")
print(f"Frames processed: {frame_count}")
print(f"Detections: {detection_count}")
print(f"Max audio level: {max_level:.4f}")

if detection_count > 0:
    print("\n✓✓✓ SUCCESS! Porcupine is working on Windows!")
elif max_level < 0.1:
    print("\n⚠️  Mic level low - try speaking louder")
else:
    print("\n⚠️  No detection - try saying 'TERMINATOR' more clearly")

print("=" * 70)