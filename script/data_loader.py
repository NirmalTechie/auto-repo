import os
import cv2
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

class SpoofDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []

        for label, sub_dir in enumerate(['real', 'fake']):
            sub_path = os.path.join(root_dir, sub_dir)
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

# Define transforms
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Example usage
dataset_path = "../dataset"
dataset = SpoofDataset(dataset_path, transform=transform)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
