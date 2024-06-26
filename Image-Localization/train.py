import torch
import argparse
import datetime
import torchsummary
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torchvision.models as models

from torchvision import transforms
from torch_lr_finder import LRFinder
from torch.utils.data import DataLoader
from torchvision.models import ResNet18_Weights
from custom_dataset import NoseCoordinateDataset


def train_transform():
    transform_list = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ]
    return transforms.Compose(transform_list)


def parse_arguments():
    parser = argparse.ArgumentParser(description='Arguments to pass to the train module')
    parser.add_argument('-d', type=str, default='./data', help='train file directory')
    parser.add_argument('-cuda', type=str, default='cuda:0', help='device')
    parser.add_argument('-lr', type=float, default=0.0001, help='initial learning rate')
    parser.add_argument('-s', type=str, default='modelEXP.pth', help='classification head weight path')
    parser.add_argument('-e', type=int, default=40, help='number of epochs')
    parser.add_argument('-p', type=str, default='loss_plotEXP.png', help='Path to save the loss plot')

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


def train(n_epochs, optimizer, model, loss_fn, train_loader, scheduler, device):
    print('Training Starting: ', datetime.datetime.now())
    losses_train = []
    for epoch in range(1, n_epochs + 1):
        running_loss = 0.0
        for img, coords in train_loader:
            img = img.to(device)
            coords = coords.to(device)

            optimizer.zero_grad()
            outputs = model(img)
            loss = loss_fn(outputs, coords)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        scheduler.step(running_loss)

        losses_train.append(running_loss / len(train_loader))
        print('{} Epoch {}. Training Loss {}'.format(
            datetime.datetime.now(), epoch, running_loss / len(train_loader)))

        plt.plot(losses_train)
        plt.savefig(args.p)

    return 'Training Complete'


if __name__ == '__main__':
    args = parse_arguments()

    deviceUsed = torch.device(args.cuda)
    print(torch.cuda.get_device_name(torch.cuda.current_device()))

    model = NoseLocalizationModel()
    model = model.to(deviceUsed)

    transform = train_transform()
    train_set = NoseCoordinateDataset(dir=args.d, training=True, transform=transform)
    train_dataloader = DataLoader(train_set, batch_size=24, shuffle=True)

    opt = optim.Adam(params=model.parameters(), lr=args.lr, amsgrad=True)
    sched = optim.lr_scheduler.ReduceLROnPlateau(optimizer=opt, factor=0.1, patience=5, threshold=0.1, verbose=True)
    criterion = torch.nn.MSELoss()

    # Both torchsummary and lr_finder are for aiding with finding hyperparams
    # or debugging, not required to run code
    # 
    # input_size = (3, 224, 224)
    # torchsummary.summary(model, input_size)

    # lr_finder = LRFinder(model, opt, criterion)
    # lr_finder.range_test(train_dataloader, end_lr=1, num_iter=1000, step_mode='linear')
    # lr_finder.plot()

    check = train(args.e, opt, model, criterion, train_dataloader, sched, deviceUsed)
    print(check)

    model_state_dict = model.state_dict()
    torch.save(model_state_dict, args.s)

    print("Model Saved")
