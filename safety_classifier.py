from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, RedirectResponse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import os
import io
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
from typing import List

# Create FastAPI app
app = FastAPI()

# Load the pre-trained SafetyCNN model
class SafetyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=12, kernel_size=5, stride=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=12, out_channels=24, kernel_size=5, stride=1)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(157 * 157 * 24, 64)
        self.fc2 = nn.Linear(64, 1)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Initialize model and load pre-trained weights
safety_model = SafetyCNN()
safety_model.load_state_dict(torch.load("safety_model.pth", map_location=torch.device('cpu')))
safety_model.eval()

# Define transformations
tta_transforms = [
    transforms.Compose([]),
    transforms.Compose([transforms.RandomHorizontalFlip(p=1.0)]),
    transforms.Compose([transforms.RandomRotation(degrees=30)]),
    transforms.Compose([transforms.RandomResizedCrop(size=(640, 640), scale=(0.8, 1.0))])
]

# Utility function to classify image with TTA
def classify_image_with_tta(image: Image.Image, model, tta_transforms: List[transforms.Compose], num_tta=4):
    # List to accumulate predictions from TTA versions
    augmented_predictions = []

    # Apply each TTA transformation to the image
    for i in range(num_tta):
        tta_transform = tta_transforms[i % len(tta_transforms)]
        augmented_image = tta_transform(image)
        
        # Preprocess the image for the model
        transform = transforms.Compose([
            transforms.Resize((640, 640)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        input_tensor = transform(augmented_image).unsqueeze(0)  # Add batch dimension (1, 3, 640, 640)

        # Run the model on the augmented image
        with torch.no_grad():
            output = model(input_tensor).squeeze(1)

        # Apply sigmoid to get the probability
        prob = torch.sigmoid(output).item()
        augmented_predictions.append(prob)

    # Average predictions over all TTA versions
    avg_prob = np.mean(augmented_predictions)

    # Set a threshold of 0.5 for binary classification
    prediction = 1 if avg_prob > 0.5 else 0

    return prediction, avg_prob

@app.get("/")
async def docs():
    return RedirectResponse(url="/docs")

# FastAPI endpoint to upload an image and classify it
@app.post("/classify")
async def classify_image(file: UploadFile = File(...)):
    try:
        # Read the uploaded file
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail="Invalid image file")
    
    # Classify the image using the model
    prediction, avg_prob = classify_image_with_tta(image, safety_model, tta_transforms, num_tta=4)
    
    # Create a response message
    result = {
        "prediction": "Unsafe" if prediction == 1 else "Safe",
        "probability": avg_prob
    }
    return JSONResponse(content=result)

# Optional: Run with uvicorn if needed
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)

# To run the server:
# uvicorn safety_classifier:app --reload
