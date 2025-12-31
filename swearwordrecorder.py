import pyaudio
import numpy as np
import soundfile as sf
import os
import time

os.makedirs("swear_clips", exist_ok=True)

SWEAR_WORD = "fuck"
clips_per_word = 20

pa = pyaudio.PyAudio()
stream = pa.open(rate=16000, channels=1, format=pyaudio.paInt16, input=True, frames_per_buffer=1024)

print(f"🤬 RECORDING '{SWEAR_WORD.upper()}' x{clips_per_word} (**2s clips**)")

try:
    for i in range(clips_per_word):
        print(f"\n📢 SAY '{SWEAR_WORD.upper()}' NOW! [{i + 1}/{clips_per_word}]")
        time.sleep(1.5)  # Prep time

        print("🎤 RECORDING 2s... ", end='', flush=True)

        # 2s = 32 chunks x 1024 samples = 32768 samples
        audio_chunks = []
        for _ in range(32):
            chunk = stream.read(1024, exception_on_overflow=False)
            audio_chunks.append(np.frombuffer(chunk, dtype=np.int16).astype(np.float32))

        audio = np.concatenate(audio_chunks) / 32768.0  # Full 2s

        max_level = np.max(np.abs(audio))
        avg_level = np.mean(np.abs(audio))

        if max_level > 0.02:  # Has speech
            filename = f"swear_clips/{SWEAR_WORD}_{i:02d}.wav"
            sf.write(filename, audio, 16000)
            print(f"✅ {filename} (peak:{max_level:.2f} avg:{avg_level:.3f})")
        else:
            print("❌ NO SPEECH - yell louder!")

        time.sleep(0.5)

except KeyboardInterrupt:
    print("\n🛑 Stopped!")

finally:
    stream.close()
    pa.terminate()
    print("🎉 2s clips ready! Reply 'clips done'")
