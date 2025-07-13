import torch
from transformers import ViTForImageClassification, ViTFeatureExtractor
from PIL import Image

import warnings
warnings.filterwarnings("ignore")

# Load the model and feature extractor
model = ViTForImageClassification.from_pretrained('EdBianchi/vit-fire-detection')
feature_extractor = ViTFeatureExtractor.from_pretrained('EdBianchi/vit-fire-detection')

# Load an image
image = Image.open('not_fire.jpg')

# Preprocess the image
inputs = feature_extractor(images=image, return_tensors="pt")

# Perform inference
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class_idx = logits.argmax(-1).item()

# Print the predicted class
prediction = model.config.id2label[predicted_class_idx]
print(f"Predicted class: {prediction}")
