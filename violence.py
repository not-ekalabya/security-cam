import torch
from transformers import ViTForImageClassification, ViTFeatureExtractor
from PIL import Image

# Load the model and feature extractor
model = ViTForImageClassification.from_pretrained('jaranohaal/vit-base-violence-detection')
feature_extractor = ViTFeatureExtractor.from_pretrained('jaranohaal/vit-base-violence-detection')

# Load an image
image = Image.open('data/fight.jpg')

# Preprocess the image
inputs = feature_extractor(images=image, return_tensors="pt")

# Perform inference
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class_idx = logits.argmax(-1).item()

# Print the predicted class
print("Predicted class:", model.config.id2label[predicted_class_idx])
