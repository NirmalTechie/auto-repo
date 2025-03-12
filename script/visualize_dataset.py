import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import os
import cv2
import torch
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

# 🔹 Fix OpenMP Runtime Error
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Define Spoof Dataset Class
class SpoofDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []

        for label, sub_dir in enumerate(['real', 'fake']):
            sub_path = os.path.join(root_dir, sub_dir)
            if not os.path.exists(sub_path):
                print(f"⚠️ Warning: '{sub_path}' does not exist!")
                continue
            for img_name in os.listdir(sub_path):
                self.image_paths.append(os.path.join(sub_path, img_name))
                self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.transform:
            image = self.transform(image)

        label = self.labels[idx]
        return image, label

# 🔹 Define image transformations
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# 🔹 Load dataset
dataset_path = "../dataset"  # Make sure the dataset is correctly placed
dataset = SpoofDataset(dataset_path, transform=transform)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# 🔹 Visualize a Batch of Images
def visualize_batch(dataloader):
    images, labels = next(iter(dataloader))
    
    fig, axes = plt.subplots(4, 4, figsize=(10, 10))
    for i, ax in enumerate(axes.flat):
        img = images[i].permute(1, 2, 0).numpy()  # Convert tensor to image
        label = "Real" if labels[i] == 0 else "Fake"
        ax.imshow(img)
        ax.set_title(label)
        ax.axis("off")
    
    plt.show()

# 🔹 Run visualization
if len(dataset) > 0:
    print(f"✅ Dataset loaded successfully with {len(dataset)} images.")
    visualize_batch(dataloader)
else:
    print("❌ No images found in the dataset. Check the dataset folder path.")
import matplotlib.pyplot as plt
plt.show()
