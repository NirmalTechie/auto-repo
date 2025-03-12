import cv2
import torch
import numpy as np
from torchvision import transforms
from model import FaceSpoofModel  # Replace with your actual model class

# Load the trained model
model = FaceSpoofModel()  # Initialize your model
model.load_state_dict(torch.load("best_model.pth", map_location=torch.device('cpu')))  # Load trained model weights
model.eval()  # Set to evaluation mode

# Define transformations
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to RGB and preprocess
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_tensor = transform(rgb_frame).unsqueeze(0)  # Add batch dimension

    # Make prediction
    with torch.no_grad():
        output = model(img_tensor)
        prediction = torch.sigmoid(output).item()  # Get probability

    # Determine label
    label = "Real" if prediction > 0.5 else "Fake"
    color = (0, 255, 0) if label == "Real" else (0, 0, 255)

    # Display label on frame
    cv2.putText(frame, label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    # Show frame
    cv2.imshow("Face Spoofing Detection", frame)

    # Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
