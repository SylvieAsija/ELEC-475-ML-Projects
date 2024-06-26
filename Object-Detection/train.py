import datetime
import matplotlib.pyplot as plt
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import data
from torchvision import transforms
import numpy as np
from torch_lr_finder import LRFinder
import model as net
from torch.utils.data import DataLoader
from custom_dataset import KittiROITrainDataset
import torchsummary


def train_transform():
    transform_list = [
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ]
    return transforms.Compose(transform_list)


def val_transform():
    transform_list = [
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ]
    return transforms.Compose(transform_list)


def InfiniteSampler(n):
    # i = 0
    i = n - 1
    order = np.random.permutation(n)
    while True:
        yield order[i]
        i += 1
        if i >= n:
            np.random.seed()
            order = np.random.permutation(n)
            i = 0


class InfiniteSamplerWrapper(data.sampler.Sampler):
    def __init__(self, data_source):
        super().__init__(data_source)
        self.num_samples = len(data_source)

    def __iter__(self):
        return iter(InfiniteSampler(self.num_samples))

    def __len__(self):
        return 2 ** 31


def parse_arguments():
    parser = argparse.ArgumentParser(description='Arguments to pass to the train module')
    parser.add_argument('-lr', type=float, default=0.0001, help='initial learning rate')
    parser.add_argument('-vb', type=int, default=1700, help='number of validation iterations per epoch')
    parser.add_argument('-e', type=int, default=40, help='number of epochs')
    parser.add_argument('-b', type=int, default=7000, help='number of train iterations per epoch')
    parser.add_argument('-cuda', type=str, default='cuda:0', help='device')
    parser.add_argument('-d', type=str, default='./data/Kitti8_ROIs/', help='train file directory')
    parser.add_argument('-s', type=str, default='classification_head.pth', help='classification head weight path')
    parser.add_argument('-p', type=str, default='loss_plot.png', help='Path to save the loss plot')

    arsUsed = parser.parse_args()
    return arsUsed


def train(n_epochs, batch_size, val_batch_size, model, train_loader, validation_loader, loss_fn, optimizer, scheduler, device):
    losses_train = []
    losses_val = []
    train_num_iter = 0
    val_num_iter = 0
    print('training starting: ', datetime.datetime.now())
    for epoch in range(1, 1 + n_epochs):
        loss_train = 0.0
        model.train()
        for train_batch in range(batch_size):
            inputs, labels = next(train_loader)
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            loss_train += loss.item()
            train_num_iter = (epoch - 1) * batch_size + train_batch

            # print(batch)

        scheduler.step(loss_train)

        model.eval()
        loss_val = 0.0
        with torch.no_grad():
            for val_batch in range(val_batch_size):
                inputs, labels = next(validation_loader)
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                loss = loss_fn(outputs, labels)

                loss_val += loss.item()
                val_num_iter = (epoch - 1) * val_batch_size + val_batch

        losses_train += [loss_train / train_num_iter]
        losses_val += [loss_val / val_num_iter]

        print('{} Epoch {}. Training Loss {}. Validation Loss {}'.format(
            datetime.datetime.now(), epoch, loss_train / train_num_iter, loss_val / val_num_iter))

    plt.plot(losses_train, label='Training Loss')
    plt.plot(losses_val, label='Validation Loss')
    plt.legend()
    plt.savefig(args.p)
    plt.show()

    return "Training Complete"


if __name__ == "__main__":
    args = parse_arguments()
    deviceUsed = torch.device(args.cuda)
    print(torch.cuda.get_device_name(torch.cuda.current_device()))

    encoder = net.encoder_classify.encoder
    encoder = encoder.to(deviceUsed)

    classifier = net.encoder_classify.simple_classification
    classifier = classifier.to(deviceUsed)

    modelUsed = net.CustomNetwork(encoder, classifier)
    modelUsed = modelUsed.to(deviceUsed)

    transform = train_transform()
    train_dataset = KittiROITrainDataset(dir=args.d, training=True, transform=transform)
    train_sampler = InfiniteSamplerWrapper(train_dataset)
    train_dataloader = iter(DataLoader(train_dataset, batch_size=48, sampler=train_sampler))

    transform = val_transform()
    val_dataset = KittiROITrainDataset(dir=args.d, training=False, transform=transform)
    val_sampler = InfiniteSamplerWrapper(val_dataset)
    val_dataloader = iter(DataLoader(val_dataset, batch_size=48, sampler=val_sampler))

    opt = optim.Adam(params=modelUsed.classification.parameters(), lr=args.lr, amsgrad=True)
    sched = optim.lr_scheduler.ReduceLROnPlateau(optimizer=opt, factor=0.1, patience=5, threshold=0.01, verbose=True)
    criterion = nn.BCEWithLogitsLoss()

    # TorchSummary was used for bug fixing and testing, not required to run train.py
    #
    # input_size = (3, 224, 224)
    # torchsummary.summary(modelUsed, input_size)

    # The learning rate finder was used to find the optimal intiial learning rate
    # it was not required after that, the code has been provided for clarity and
    # since it was referenced in the report
    #
    # lr_finder = LRFinder(modelUsed, opt, criterion)
    # lr_finder.range_test(train_dataloader, end_lr=1, num_iter=1000, step_mode='linear')
    # lr_finder.plot()

    check = train(args.e, args.b, args.vb, modelUsed, train_dataloader,
                  val_dataloader, criterion, opt, sched, deviceUsed)
    print(check)

    classifier_state_dict = net.encoder_classify.simple_classification.state_dict()
    torch.save(classifier_state_dict, args.s)

    print("Model Saved")
