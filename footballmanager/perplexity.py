import pytesseract
from PIL import ImageGrab, Image, ImageEnhance, ImageOps
import win32clipboard
import io
import re
import json

# Tesseract path
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


def get_clipboard_image():
    """Robust clipboard image grab (fixes PIL bugs)"""
    try:
        # Method 1: PIL (works 80% time)
        img = ImageGrab.grabclipboard()
        if isinstance(img, Image.Image):
            return img
    except:
        pass

    try:
        # Method 2: Win32 direct (handles PNG32bit, DIB)
        win32clipboard.OpenClipboard()
        if win32clipboard.IsClipboardFormatAvailable(win32clipboard.CF_DIB) or \
                win32clipboard.IsClipboardFormatAvailable(win32clipboard.CF_BITMAP):
            data = win32clipboard.GetClipboardData()
            img = Image.open(io.BytesIO(data))
            win32clipboard.CloseClipboard()
            return img
        win32clipboard.CloseClipboard()
    except:
        pass

    print("❌ No image in clipboard. Try:")
    print("  1. PrintScreen → Ctrl+V")
    print("  2. Snipping Tool → Ctrl+V")
    print("  3. Alt+PrintScreen (active window)")
    return None


# [fm_preprocess and parse_fm_stats functions unchanged from previous]
def fm_preprocess(img):
    img = img.resize((img.width * 2, img.height * 2), Image.Resampling.LANCZOS)
    img = img.convert('RGB')
    img = img.quantize(colors=4, method=Image.Quantize.FASTOCTREE).convert('L')
    img = ImageOps.invert(img)
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(3.0)
    img = img.filter(Image.Filter.SHARPEN)
    img = img.point(lambda x: 0 if x < 128 else 255, '1')
    return img


def parse_fm_stats(text):
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    name_age = ' '.join(lines[:2])
    name_match = re.search(r'([A-Z][a-z]+ [A-Z][a-z]+).*?(\d{1,2})', name_age)
    name = name_match.group(1) if name_match else "Unknown"
    age = int(name_match.group(2)) if name_match else 0
    all_numbers = re.findall(r'\b(\d{1,2})\b', text)
    tech_stats = ["Corners", "Crossing", "Dribbling", "Finishing", "First Touch",
                  "Free Kick Taking", "Heading", "Long Shots", "Long Throws",
                  "Marking", "Passing", "Penalty Taking", "Tackling", "Technique"]
    mental_stats = ["Aggression", "Anticipation", "Bravery", "Composure",
                    "Concentration", "Decisions", "Determination", "Flair",
                    "Leadership", "Off The Ball", "Positioning", "Teamwork", "Vision", "Work Rate"]
    phys_stats = ["Acceleration", "Agility", "Balance", "Jumping Reach",
                  "Natural Fitness", "Pace", "Stamina", "Strength"]
    stats = {
        "Technical": dict(zip(tech_stats, all_numbers[:14])),
        "Mental": dict(zip(mental_stats, all_numbers[14:28])),
        "Physical": dict(zip(phys_stats, all_numbers[28:]))
    }
    return {"name": name, "age": age, "stats": stats}


if __name__ == "__main__":
    img = get_clipboard_image()
    if img is None:
        exit(1)

    processed = fm_preprocess(img)
    processed.save("fm_processed.png")

    text = pytesseract.image_to_string(processed, config='--psm 6')
    print("RAW OCR:\n", repr(text))  # repr() shows hidden chars

    result = parse_fm_stats(text)
    print("\n✅ FM STATS:")
    print(json.dumps(result, indent=2))
