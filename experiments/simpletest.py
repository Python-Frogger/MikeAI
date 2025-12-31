import pyaudio      # Audio library.
import numpy as np  # Math operations.
import time         # Delays.

print("Testing Windows DEFAULT mic...")

pa = pyaudio.PyAudio()
stream = pa.open(   # Uses Windows default mic automatically.
    rate=16000,     # Speech standard rate.
    channels=1,     # Mono.
    format=pyaudio.paInt16,  # Windows standard.
    input=True,     # Microphone input.
    frames_per_buffer=1024
)

print("🎤 SPEAK LOUDLY FOR 5 SECONDS NOW! 🎤")
levels = []

for i in range(80):  # 80 chunks = 5 seconds.
    try:
        data = stream.read(1024, exception_on_overflow=False)
        audio = np.frombuffer(data, dtype=np.int16).astype(np.float32)
        level = np.max(np.abs(audio)) / 32768.0  # Volume 0.0 to 1.0.
        levels.append(level)
        print(f"Level: {level:.3f} {'🔴' if level<0.01 else '🟢'}", end='\r')
        time.sleep(0.06)
    except:
        print("READ ERROR")
        break

avg_level = np.mean([l for l in levels if l > 0])
print(f"\n🎯 FINAL: Average speech level = {avg_level:.3f}")

stream.stop_stream()
stream.close()
pa.terminate()

if avg_level > 0.05:
    print("✅ MIC WORKS! Ready for openWakeWord.")
else:
    print("❌ NO AUDIO - Check Windows mic settings + Run PyCharm as Admin")
