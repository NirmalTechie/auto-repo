import torch
import cv2
import numpy as np
from torchvision import transforms
from train import model

def predict(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    
    image = transform(image).unsqueeze(0)
    output = model(image)
    _, predicted = torch.max(output, 1)

    return "Real" if predicted.item() == 0 else "Fake"

print(predict("../dataset/fake/example.jpg"))
