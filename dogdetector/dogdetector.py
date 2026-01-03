"""
Stuffed Animal Detector - What does the AI think it is?
Shows TOP 5 predictions with confidence percentages
Press 'q' to quit
"""

# ==================== IMPORTS ====================
# Import libraries we need
import cv2  # OpenCV - for webcam and image processing
import torch  # PyTorch - the AI framework
import torchvision.transforms as transforms  # Image preprocessing tools
import torchvision.models as models  # Pre-trained AI models
import json  # To load the list of object names

# ==================== STEP 1: LOAD CLASS LABELS ====================
# ImageNet has 1000 different object categories (dog, cat, airplane, etc.)
# We need to load the names so we can show "dog" instead of just "207"
print("Loading ImageNet class labels...")

# This downloads a JSON file with all 1000 category names
import urllib.request

url = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
with urllib.request.urlopen(url) as f:
    imagenet_labels = json.load(f)

print(f"Loaded {len(imagenet_labels)} categories")

# ==================== STEP 2: LOAD THE AI MODEL ====================
# MobileNetV2 is a lightweight, fast image classifier
# It's been trained on millions of images to recognize 1000 different things
print("Loading MobileNetV2 AI model...")
print("(This might take a minute on first run - downloading model...)")

model = models.mobilenet_v2(pretrained=True)  # Download pre-trained model
model.eval()  # Set to "evaluation mode" (not training)

# Move model to GPU if available (your RX 9070!)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
print(f"Model loaded on: {device}")

# ==================== STEP 3: IMAGE PREPROCESSING ====================
# AI models are picky! They need images in a specific format:
# - Resize to 224x224 pixels
# - Convert to tensor (AI-readable format)
# - Normalize colors (make them consistent)
preprocess = transforms.Compose([
    transforms.ToPILImage(),  # Convert OpenCV image to PIL format
    transforms.Resize(256),  # Resize to 256 pixels
    transforms.CenterCrop(224),  # Crop center 224x224 square
    transforms.ToTensor(),  # Convert to tensor
    transforms.Normalize(  # Normalize colors (ImageNet standard)
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])


# ==================== STEP 4: FUNCTION TO GET TOP 5 PREDICTIONS ====================
def get_top5_predictions(frame):
    """
    Takes a webcam frame and returns the top 5 things the AI thinks it sees

    Args:
        frame: Image from webcam (OpenCV format)

    Returns:
        List of tuples: [(label, confidence), (label, confidence), ...]
    """
    # Preprocess the image for the AI model
    input_tensor = preprocess(frame)
    input_batch = input_tensor.unsqueeze(0)  # Add batch dimension

    # Move to GPU if available
    input_batch = input_batch.to(device)

    # Run the AI model (no gradient calculation needed for inference)
    with torch.no_grad():
        output = model(input_batch)

    # Convert raw output to probabilities (percentages)
    probabilities = torch.nn.functional.softmax(output[0], dim=0)

    # Get top 5 predictions
    top5_prob, top5_indices = torch.topk(probabilities, 5)

    # Convert to readable format with labels and percentages
    results = []
    for i in range(5):
        label = imagenet_labels[top5_indices[i].item()]
        confidence = top5_prob[i].item() * 100  # Convert to percentage
        results.append((label, confidence))

    return results


# ==================== STEP 5: OPEN WEBCAM ====================
print("\nOpening webcam...")
print("(This might take 1-2 minutes on AMD GPUs - please wait...)")
webcam = cv2.VideoCapture(1, cv2.CAP_DSHOW)  # CHANGE ME IF HAVE THE CAPTURE CARD IN!
# 0 = capture card , 1 = webcam

if not webcam.isOpened():
    print("ERROR: Could not open webcam!")
    exit()

print("Webcam ready!")
print("\n=== CONTROLS ===")
print("Press 'q' to quit")
print("Press 's' to save a screenshot")
print("Rotate your stuffed animals to see different predictions!")
print("================\n")

# Counter for screenshot filenames
screenshot_count = 0

# ==================== STEP 6: MAIN LOOP ====================
frame_count = 0  # Track frames for performance
predictions = []  # Store current predictions

while True:
    # Read frame from webcam
    ret, frame = webcam.read()

    if not ret:
        print("ERROR: Failed to grab frame")
        break

    # Only run AI every 15 frames to reduce flashing
    # This makes predictions more stable and easier to read
    # (15 frames ≈ every 0.5 seconds at 30fps)
    if frame_count % 15 == 0:
        # Get top 5 predictions
        predictions = get_top5_predictions(frame)

    # Display predictions on the frame (every frame, not just when updated!)
    # This keeps the text stable and prevents flashing
    if predictions:  # Only draw if we have predictions
        y_position = 30  # Starting y position for text

        # Add semi-transparent black background for better text visibility
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (450, 200), (0, 0, 0), -1)
        # Blend overlay with frame for semi-transparency
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        # Draw each prediction
        for i, (label, confidence) in enumerate(predictions):
            # Color code by confidence:
            # Green for high confidence (>50%)
            # Yellow for medium (20-50%)
            # Red for low (<20%)
            if confidence > 50:
                color = (0, 255, 0)  # Green
            elif confidence > 20:
                color = (0, 255, 255)  # Yellow
            else:
                color = (0, 100, 255)  # Orange/Red

            # Format: "1. wolf (45.2%)"
            text = f"{i + 1}. {label}: {confidence:.1f}%"
            cv2.putText(frame, text, (20, y_position),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            y_position += 30

    # Show the frame
    cv2.imshow('Animal Detector - Press Q to quit, S to screenshot', frame)

    # Check for key presses
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        print("\nQuitting...")
        break
    elif key == ord('s'):
        # Save screenshot
        filename = f"screenshot_{screenshot_count:03d}.png"
        cv2.imwrite(filename, frame)
        print(f"📸 Screenshot saved: {filename}")
        screenshot_count += 1

    frame_count += 1

# ==================== STEP 7: CLEANUP ====================
webcam.release()
cv2.destroyAllWindows()
print("Done! Hope you had fun seeing what the AI thinks! 🐺🦊🐵")