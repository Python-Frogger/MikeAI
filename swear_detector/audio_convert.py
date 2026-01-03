"""
Convert M4A to proper WAV format for Vosk
==========================================
Converts audio files to exactly what Vosk needs:
- 16000 Hz sample rate
- Mono (1 channel)
- WAV format

Requirements: pip install pydub
"""

from pydub import AudioSegment
import os

print("=" * 70)
print("AUDIO CONVERTER FOR VOSK")
print("=" * 70)cd s

# Input files (change these to match your files)
INPUT_FILES = [
    ("mike.m4a", "mike.wav"),
    ("james.m4a", "james.wav")
]

for input_file, output_file in INPUT_FILES:
    if not os.path.exists(input_file):
        print(f"⚠️  Skipping {input_file} - file not found")
        continue

    print(f"\nConverting {input_file}...")

    # Load the audio file
    audio = AudioSegment.from_file(input_file)

    print(f"  Original: {audio.frame_rate} Hz, {audio.channels} channel(s)")

    # Convert to exactly what Vosk needs
    audio = audio.set_frame_rate(16000)  # 16 kHz
    audio = audio.set_channels(1)  # Mono

    # Export as WAV
    audio.export(output_file, format="wav")

    print(f"  ✓ Converted to: 16000 Hz, 1 channel")
    print(f"  ✓ Saved as: {output_file}")

print("\n" + "=" * 70)
print("CONVERSION COMPLETE!")
print("=" * 70)
print("\nNext steps:")
print("1. Delete the old .npy signature files")
print("2. Run: python train_speakers.py")
print("3. Run: python live_swear_detector.py")
print("=" * 70)