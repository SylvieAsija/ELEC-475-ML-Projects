from PIL import Image
from torch.utils.data import Dataset
import os


class KittiROITestDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(os.listdir(self.image_dir))

    def __getitem__(self, idx):
        image_path = f"{self.image_dir}/image_{idx}.png"
        image = Image.open(image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, -1