import torch
from transformers import ViTForImageClassification, ViTFeatureExtractor
from PIL import Image
import cv2
import warnings
import torch.nn.functional as F

warnings.filterwarnings("ignore")

# Load the model and feature extractor
model = ViTForImageClassification.from_pretrained('EdBianchi/vit-fire-detection')
feature_extractor = ViTFeatureExtractor.from_pretrained('EdBianchi/vit-fire-detection')

model.eval()

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Load the video
cap = cv2.VideoCapture('data/fire.mp4')  # Replace with your video path
frame_interval = 5  # Predict every 5th frame
frame_count = 0
probs_display = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    if frame_count % frame_interval == 0:
        # Convert OpenCV BGR frame to PIL RGB image
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # Preprocess the image
        inputs = feature_extractor(images=image, return_tensors="pt").to(device)

        # Perform inference
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = F.softmax(logits, dim=-1).cpu().squeeze().numpy()

            # Get label names from the config
            id2label = model.config.id2label

            # Determine label: "fire" if fire probability > 0.1%, else "normal"
            label = "fire" if probs[0] > 0.001 else "normal"

    # Overlay label on frame
    color = (0, 0, 255) if label == "fire" else (0, 255, 0)
    cv2.putText(frame, label, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 3)

    # Show the frame
    cv2.imshow('Fire Detection - Probabilities', frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_count += 1

cap.release()
cv2.destroyAllWindows()
