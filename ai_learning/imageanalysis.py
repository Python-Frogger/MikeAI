# One-time setup
# !pip install torch torchvision pillow -q

import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import matplotlib.pyplot as plt
import requests
from io import BytesIO
import json

print("🤖 Loading AI Brain...")
model = models.mobilenet_v2(pretrained=True)
model.eval()

# Get labels
LABELS_URL = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
labels = json.loads(requests.get(LABELS_URL).text)
print("✅ AI Ready!\n")

def show_me(url):
    """Paste an image URL and watch AI identify it!"""

    # Download image
    print(f"📥 Downloading image...")
    response = requests.get(url)
    img = Image.open(BytesIO(response.content)).convert('RGB')

    # Prepare for AI
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    img_tensor = transform(img).unsqueeze(0)

    # AI THINKS...
    print("🧠 AI analyzing...")
    with torch.no_grad():
        output = model(img_tensor)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        top3_prob, top3_idx = torch.topk(probabilities, 3)

    # SHOW RESULTS!
    plt.figure(figsize=(10, 8))
    plt.imshow(img)
    plt.axis('off')

    result = f"🤖 AI SAYS:\n\n"
    result += f"#{1} {labels[top3_idx[0]].upper()}: {top3_prob[0]*100:.0f}%\n"
    result += f"#{2} {labels[top3_idx[1]]}: {top3_prob[1]*100:.0f}%\n"
    result += f"#{3} {labels[top3_idx[2]]}: {top3_prob[2]*100:.0f}%"

    plt.title(result, fontsize=14, loc='left', fontweight='bold',
              bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
    plt.tight_layout()
    plt.show()

    print(result)
    print("\n" + "="*50)

# ============================================
# PASTE IMAGE URLS BELOW AND RUN!
# ============================================

# Golden Retriever
show_me("https://images.unsplash.com/photo-1633722715463-d30f4f325e24?w=600")

# Cat
show_me("https://images.unsplash.com/photo-1574158622682-e40e69881006?w=600")

# Pizza
show_me("https://images.unsplash.com/photo-1513104890138-7c749659a591?w=600")

# Ferrari
show_me("https://images.unsplash.com/photo-1583121274602-3e2820c69888?w=600")

show_me("https://www.shutterstock.com/editorial/image-editorial/MeTdA1y7N8j8gc37MzQwNTE=/brighton--hove-albion-fan-seagull-mask-550nw-10189208k.jpg")