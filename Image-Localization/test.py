import numpy
import torchvision.models as models
import torch.nn as nn
from torchvision.models import ResNet18_Weights
import torch
import argparse
from torchvision import transforms
from custom_dataset import NoseCoordinateDataset
from torch.utils.data import DataLoader
import math
import csv

def test_transforms():
    transform_list = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ]
    return transforms.Compose(transform_list)


def parse_arguments():
    parser = argparse.ArgumentParser(description='Arguments to pass to the train module')
    parser.add_argument('-d', type=str, default='./data', help='train file directory')
    parser.add_argument('-cuda', type=str, default='cuda:0', help='device')
    parser.add_argument('-l', type=str, default='model3.pth', help='classification head weight path')

    argsUsed = parser.parse_args()
    return argsUsed


class NoseLocalizationModel(nn.Module):
    def __init__(self):
        super(NoseLocalizationModel, self).__init__()
        resnet = models.resnet18(weights=ResNet18_Weights.DEFAULT)  # Load pre-trained ResNet
        self.features = nn.Sequential(*list(resnet.children())[:-1])  # Remove last fully connected layer
        self.regressor = nn.Sequential(
            nn.Linear(resnet.fc.in_features, 256),
            nn.ReLU(),
            nn.Linear(256, 2)  # Output 2 coordinates for x, y
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.regressor(x)
        return x


def euclidian_distance(label, prediction):
    pred_x, pred_y = prediction[0]
    label_x, label_y = label[0]
    x_2 = pow(label_x - pred_x, 2)
    y_2 = pow(label_y - pred_y, 2)
    distance = math.sqrt(x_2 + y_2)
    return distance


if __name__ == '__main__':
    args = parse_arguments()

    deviceUsed = torch.device(args.cuda)
    print(torch.cuda.get_device_name(torch.cuda.current_device()))

    model = NoseLocalizationModel()
    model.load_state_dict(torch.load(args.l))
    model = model.to(deviceUsed)

    test_transform = test_transforms()
    test_dataset = NoseCoordinateDataset(dir=args.d, training=False, transform=test_transform)
    test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=1)

    min_dist = 1000.0
    max_dist = 0.0
    mean_dist = 0.0
    st_dev = 0.0

    listed = []
    preds = []
    images = []
    noses = []

    label_file = f"./data/test/resized_labels/labels.txt"
    with open(label_file) as csvfile:
        reader = csv.DictReader(csvfile)

        for row in reader:
            images.append(row['image'])
            noses.append(row['nose'])

    with torch.no_grad():
        model.eval()
        i = 0
        for img, coords in test_dataloader:
            img = img.to(deviceUsed)
            coords = coords.to(deviceUsed)
            outputs = model(img)

            # Print statements for debugging, not required to run code
            # print(outputs)
            # print(coords)
            dist = euclidian_distance(coords, outputs)
            listed.append(dist)
            preds.append(outputs)
            if dist < min_dist:
                min_dist = dist
            if dist > max_dist:
                max_dist = dist
            i += 1

        with (open('data/test/predictions/predictionss.txt', "w") as f):
            for count in range(i):
                outputs = preds[count]
                x, y = outputs[0]
                toWrite = images[count] + ',' + '"(' + str(int(x)) + ',' + str(int(y)) + ')"' + '\n'
                f.write(toWrite)

        mean_dist = sum(listed)/len(listed)
        st_dev = numpy.std(listed)

        print('Min Euclidian Distance: ', min_dist)
        print('Max Euclidian Distance: ', max_dist)
        print('Mean Euclidian Distance: ', mean_dist)
        print('Standard Deviation of Euclidian Distances: ', st_dev)




