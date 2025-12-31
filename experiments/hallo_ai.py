'''import pyttsx3  # Imports the pyttsx3 library for offline text-to-speech conversion.

engine = pyttsx3.init()  # Initializes the TTS engine using your Windows SAPI voices (no internet needed).
engine.setProperty('rate', 150)  # Sets speaking speed to 150 words/min (default 200; lower = slower).
voices = engine.getProperty('voices')  # Gets list of available voices on your system.
engine.setProperty('voice', voices[0].id)  # Sets female voice (index 0=male, 1=female usually; test both).

engine.say("Swear detector online. Say your wake word.")  # Queues text to be spoken.
engine.runAndWait()  # Plays queued speech and waits until done before continuing.
'''

import openwakeword
import pyaudio
import numpy as np
import time

openwakeword.utils.download_models()
oww_model = openwakeword.Model()

pa = pyaudio.PyAudio()
stream = pa.open(rate=16000, channels=1, format=pyaudio.paInt16, input=True, frames_per_buffer=1280)

print("🔊 SENSITIVE MODE: Threshold = 0.1 (was 0.5)")
print("📢 SCREAM 'HEY JARVIS' x3 - watch scores RISE!")

detection_count = 0

try:
    while True:
        audio_bytes = stream.read(1280, exception_on_overflow=False)
        audio = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0

        level = np.max(np.abs(audio))
        predictions = oww_model.predict(audio)

        # LOWERED THRESHOLD + SHOW hey_jarvis SPECIFICALLY
        jarvis_score = predictions.get('hey_jarvis', 0)
        alexa_score = predictions.get('alexa', 0)

        if jarvis_score > 0.01 or alexa_score > 0.01:  # Much lower!
            detection_count += 1
            print(f"\n🎯 DETECTED #{detection_count}: jarvis={jarvis_score:.3f}, alexa={alexa_score:.3f}")

        # Live debug
        print(f"Level:{level:.2f} jarvis:{jarvis_score:.4f} alexa:{alexa_score:.4f}", end='\r')
        time.sleep(0.01)

except KeyboardInterrupt:
    print(f"\n✅ Total: {detection_count}")
finally:
    stream.close()
    pa.terminate()
