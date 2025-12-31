import openwakeword                                        # Core library.
import numpy as np                                        # Audio processing.
import soundfile as sf                                    # Read/write WAV files.
import os                                                 # File operations.

print("🤬 SIMPLE LOCAL 'fuck' TRAINING")

# Load your fuck clips
positive_paths = [f"swear_clips/{f}" for f in os.listdir("swear_clips") if f.startswith("fuck_")]
print(f"✅ {len(positive_paths)} fuck clips loaded")

# Create 40 silence negatives (fuck ≠ silence)
print("⚖️ Creating silence negatives...")
for i in range(40):
    silence = np.zeros(32000)  # 2s silence @ 16kHz.
    sf.write(f"swear_clips/neg_{i:02d}.wav", silence, 16000)

negative_paths = [f"swear_clips/neg_{i:02d}.wav" for i in range(40)]

# Use openwakeword's INTERNAL trainer (no datasets module needed)
print("🧠 Training with basic trainer...")
from openwakeword.model import train_model                # Direct training function.

model = train_model(
    positive_examples=positive_paths,                    # Your clips = positive class.
    negative_examples=negative_paths,                     # Silence = negative class.
    target_classes=["fuck"],                             # Model name.
    epochs=20                                            # Quick training.
)

model.save("fuck_detector.onnx")                         # Save ONNX model.
print("🎉 fuck_detector.onnx READY!")

# Test load
oww = openwakeword.Model("fuck_detector.onnx")
print("✅ SUCCESS! Ready for live detector.")
