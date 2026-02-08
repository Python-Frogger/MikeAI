import re
import os
import json
import csv
import pyttsx3
import winsound
import numpy as np
import easyocr
import time

from PIL import Image, ImageGrab
from typing import Dict, Optional

# --- SILENCE AMD GPU WARNINGS ---
os.environ['HIP_VISIBLE_DEVICES'] = '0'
os.environ['MIOPEN_LOG_LEVEL'] = '0'


def get_clipboard_image() -> Optional[Image.Image]:
    try:
        content = ImageGrab.grabclipboard()
        if isinstance(content, Image.Image): return content
        if isinstance(content, list) and len(content) > 0: return Image.open(content[0])
    except Exception as e:
        print(f"⚠️ Clipboard Error: {e}")
    return None


def extract_fm_stats(img: Image.Image, reader: easyocr.Reader) -> Dict:
    start_time = time.time()

    # SPEED HACK: If image is huge (4K), resize it by half.
    # This makes the AI 4x faster with minimal accuracy loss.
    if img.width > 2000:
        img = img.resize((img.width // 2, img.height // 2), resample=Image.LANCZOS)

    print("🔍 Scanning image...")
    # Standard settings are usually best for ROCm compatibility
    results = reader.readtext(np.array(img))



    player_data = {
        'Name': 'Unknown', 'Age': '?', 'Positions': '?',
        'Technical': {}, 'Mental': {}, 'Physical': {}
    }

    stats_map = {
        'technical': ['Corners', 'Crossing', 'Dribbling', 'Finishing', 'First Touch', 'Free Kick', 'Heading',
                      'Long Shots', 'Long Throws', 'Marking', 'Passing', 'Penalty', 'Tackling', 'Technique'],
        'mental': ['Aggression', 'Anticipation', 'Bravery', 'Composure', 'Concentration', 'Decisions', 'Determination',
                   'Flair', 'Leadership', 'Off The Ball', 'Positioning', 'Teamwork', 'Vision', 'Work Rate'],
        'physical': ['Acceleration', 'Agility', 'Balance', 'Jumping', 'Fitness', 'Pace', 'Stamina', 'Strength']
    }

    text_data = []
    for (bbox, text, conf) in results:
        box_height = bbox[2][1] - bbox[0][1]
        y_center = (bbox[0][1] + bbox[2][1]) / 2
        x_left = bbox[0][0]
        text_data.append({'y': y_center, 'x': x_left, 'h': box_height, 'text': text})

    # --- NAME DETECTION (TARGETED) ---
    header_area = [d for d in text_data if d['y'] < 130]
    header_area.sort(key=lambda k: k['h'], reverse=True)
    excluded = ['FM', 'HOME', 'WORLD', 'SEARCH', 'CONTINUE', 'SAVE', 'GAME']

    for item in header_area:
        txt = item['text'].strip().upper()
        if len(txt) > 3 and not any(ex in txt for ex in excluded):
            player_data['Name'] = txt
            break

    full_blob = " ".join([d['text'] for d in text_data])

    # Age & Positions
    age_match = re.search(r'(\d{2})\s?years?', full_blob, re.IGNORECASE)
    if age_match: player_data['Age'] = age_match.group(1)

    pos_match = re.search(r'([A-Z]{1,2}\s?\(.[RLC]?\))', full_blob)
    if pos_match: player_data['Positions'] = pos_match.group(1)

    # Stats Extraction
    for category, stats in stats_map.items():
        for stat_name in stats:
            for item in text_data:
                if stat_name.lower() in item['text'].lower():
                    for val_item in text_data:
                        if abs(val_item['y'] - item['y']) < 15:
                            if val_item['x'] > item['x'] and val_item['text'].isdigit():
                                val = int(val_item['text'])
                                if 1 <= val <= 20:
                                    player_data[category.capitalize()][stat_name] = val
                                    break
    print(f"⏱️ Scan completed in {time.time() - start_time:.2f} seconds.")
    return player_data


def save_data(new_player_data: dict, json_filename='fm_database.json', csv_filename='fm_database.csv'):
    # 1. JSON
    database = []
    if os.path.exists(json_filename):
        try:
            with open(json_filename, 'r') as f:
                database = json.load(f)
        except:
            database = []

    database.append(new_player_data)
    with open(json_filename, 'w') as f:
        json.dump(database, f, indent=4)

    # 2. CSV
    file_exists = os.path.exists(csv_filename)
    flat_row = {'Name': new_player_data['Name'], 'Age': new_player_data['Age'],
                'Positions': new_player_data['Positions']}
    for cat in ['Technical', 'Mental', 'Physical']:
        for stat, val in new_player_data.get(cat, {}).items():
            flat_row[f"{cat}_{stat}"] = val

    with open(csv_filename, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(flat_row.keys()))
        if not file_exists: writer.writeheader()
        writer.writerow(flat_row)


def main():
    # Initialize engines
    engine = pyttsx3.init()
    engine.setProperty('rate', 190)

    print("🚀 INITIALIZING FM SCOUT ON RX 9070...")
    # Add 'quantize=False' - sometimes AMD cards struggle with quantized models
    reader = easyocr.Reader(['en'], gpu=True)

    print("\n✅ SYSTEM READY.")

    while True:
        if input("> ").lower() == 'q': break

        img = get_clipboard_image()
        if img:
            results = extract_fm_stats(img, reader)
            save_data(results)

            # Voice feedback
            winsound.Beep(1000, 150)
            engine.say(f"Added {results['Name']}")
            engine.runAndWait()
        else:
            print("❌ Clipboard empty!")


if __name__ == "__main__":
    main()