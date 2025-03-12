import os
import cv2
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image

class SpoofDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []

        # Check if dataset path exists
        if not os.path.exists(root_dir):
            raise FileNotFoundError(f"❌ Dataset folder not found at: {root_dir}")

        # Iterate through 'real' and 'fake' folders
        for label, sub_dir in enumerate(['real', 'fake']):
            sub_path = os.path.join(root_dir, sub_dir)
            
            if not os.path.exists(sub_path):
                raise FileNotFoundError(f"❌ Missing folder: {sub_path}")
            
            for img_name in os.listdir(sub_path):
                img_path = os.path.join(sub_path, img_name)

                # Ensure only image files are considered
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.image_paths.append(img_path)
                    self.labels.append(label)

        # Check if dataset is empty
        if len(self.image_paths) == 0:
            raise ValueError("❌ No images found in dataset. Please check your dataset folders!")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = cv2.imread(img_path)
        
        # Check if the image loaded correctly
        if image is None:
            raise ValueError(f"❌ Error loading image: {img_path}")
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB
        image = Image.fromarray(image)  # Convert OpenCV image to PIL format
        
        if self.transform:
            image = self.transform(image)

        label = self.labels[idx]
        return image, label

# Define transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),  
    transforms.ToTensor(),
])

# Define dataset path
dataset_path = "../dataset"


# Load dataset
try:
    dataset = SpoofDataset(dataset_path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    print(f"✅ Dataset loaded successfully with {len(dataset)} images.")
except Exception as e:
    print(str(e))

# Test loading a batch
try:
    for images, labels in dataloader:
        print(f"✅ Loaded batch with {images.shape[0]} images")
        break  # Stop after one batch
except Exception as e:
    print(f"❌ Error loading DataLoader: {e}")
