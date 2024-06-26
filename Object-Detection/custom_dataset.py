import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image, ImageFile
import fnmatch
import cv2


class KittiROITrainDataset(Dataset):
    def __init__(self, dir, training=True, transform=None):
        self.dir = dir
        self.training = training
        self.mode = 'train'
        if self.training == False:
            self.mode = 'test'
        self.img_dir = os.path.join(dir, self.mode, 'image')
        self.label_dir = os.path.join(dir, self.mode, 'label')
        self.transform = transform
        self.num = 0
        self.img_files = []
        for file in os.listdir(self.img_dir):
            if fnmatch.fnmatch(file, '*.png'):
                self.img_files += [file]

        self.max = len(self)

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        filename = os.path.splitext(self.img_files[idx])[0]
        img_path = os.path.join(self.img_dir, self.img_files[idx])
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)

        label_path = os.path.join(self.label_dir, 'labels.txt')
        with open(label_path) as label_file:
            # Read all lines from the file
            labels_string = label_file.readlines()

            # Find the line corresponding to the current image
            current_line = None
            for line in labels_string:
                if filename in line:
                    current_line = line
                    break

            if current_line is None:
                # Handle the case where label information for the current image is not found
                # You may choose to skip the image or handle it differently based on your needs
                raise ValueError(f"Label information not found for image: {filename}")

            # Parse the label information from the line
            label_info = current_line.split()

            # Assuming the label is the third value in the line
            label = label_info[-1].rstrip('\n')  # Remove the newline character

            # Map the label to an integer using your class_label dictionary
            label_id = self.class_label[label]

            label = torch.tensor([label_id], dtype=torch.float32)

            image = self.transform(image)

        return image, label

    def __iter__(self):
        self.num = 0
        return self

    def __next__(self):
        if (self.num >= self.max):
            raise StopIteration
        else:
            self.num += 1
            return self.__getitem__(self.num-1)


    class_label = {'NoCar': 0, 'Car': 1}

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
