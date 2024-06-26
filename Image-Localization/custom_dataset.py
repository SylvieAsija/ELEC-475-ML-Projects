import csv
import ast
import torch
import os
from PIL import Image
from torch.utils.data import Dataset


class NoseCoordinateDataset(Dataset):
    def __init__(self, dir, training=True, transform=None):
        self.nose_data = []
        self.dir = dir
        self.training = training

        self.mode = 'train'
        if self.training == False:
            self.mode = 'test'

        self.img_dir = os.path.join(dir, self.mode, 'resized_images')
        self.label_dir = os.path.join(dir, self.mode, 'resized_labels', 'labels.txt')

        print(self.img_dir)
        print(self.label_dir)

        self.transform = transform

        with open(self.label_dir, 'r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                image_name = row['image']
                nose_coord = ast.literal_eval(row['nose'])
                self.nose_data.append((image_name, nose_coord))

    def __len__(self):
        return len(self.nose_data)

    def __getitem__(self, idx):
        img_name, nose_coord = self.nose_data[idx]
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path)

        x, y = nose_coord
        x_float = float(x)
        y_float = float(y)

        # Convert to a tensor or use them in your model as needed
        coordinates = torch.tensor([x_float, y_float])

        if self.transform:
            image = self.transform(image)

        return image, coordinates

