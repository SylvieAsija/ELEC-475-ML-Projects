import os
import argparse
import datetime
import torch
import network as net
import matplotlib.pyplot as plt
import torch.nn as nn
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100
from torch.utils.data import DataLoader
import torchsummary
from torch import optim

# Set CUDA_LAUNCH_BLOCKING=1
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

def train_transform():
    transform_list = [
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ]
    return transforms.Compose(transform_list)


def parse_arguments():
    parser = argparse.ArgumentParser(description='Arguments to pass to the train module')
    parser.add_argument('-lr', type=float, default=0.001, help='initial learning rate')
    parser.add_argument('-decay', type=float, default=0.0, help='learning rate decay')
    parser.add_argument('-e', type=int, default=20, help='number of epochs')
    parser.add_argument('-b', type=int, default=512, help='batch size')
    parser.add_argument('-cuda', type=str, default='cuda:0', help='device')
    parser.add_argument('-l', type=str, default='encoder.pth', help='encoder weight path')
    parser.add_argument('-s', type=str, default='classification_head_10.pth', help='classification head weight path')
    parser.add_argument('-p', type=str, default='loss_plot_10.png', help='Path to save the loss plot')
    parser.add_argument('-c', type=int, default=10, help='CIFAR 10 or 100 dataset')
    parser.add_argument('-mod', type=int, default=0, help='Modded architecture (1) or vanilla (0)')

    argsUsed = parser.parse_args()
    return argsUsed


def train(n_epochs, optimizer, model, loss_fn, train_loader, scheduler, device):
    print('training...')
    losses_train = []

    for epoch in range(1, n_epochs + 1):
        loss_train = 0.0
        for img, labels in train_loader:
            img = img.to(device=device)
            labels = labels.to(device=device)
            optimizer.zero_grad()
            outputs = model(img)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            loss_train += loss.item()

        scheduler.step(loss_train)

        losses_train += [loss_train / len(train_loader)]

        print('{} Epoch {}. Training Loss {}'.format(
            datetime.datetime.now(), epoch, loss_train / len(train_loader)))

        plt.savefig(args.p)
        plt.plot(losses_train)


    return "Training Complete"


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    args = parse_arguments()
    if args.mod == 1:
        classification_head = net.encoder_classification.modded_classification_head
    else:
        classification_head = net.encoder_classification.classification_head

    encoder = net.encoder_classification.encoder
    encoder.load_state_dict(torch.load(args.l))
    modelUsed = net.network(encoder, classification_head)
    modelUsed.train()
    modelUsed.to(device)

    if args.c == 100:
        train_set = CIFAR100('./data/CIFAR100', train=True, download=True, transform=train_transform())
    else:
        train_set = CIFAR10('./data/CIFAR10', train=True, download=True, transform=train_transform())

    train_load = DataLoader(train_set, batch_size=args.b, shuffle=True)

    lr = args.lr
    opt = optim.Adam(params=modelUsed.classification.parameters(), lr=lr, amsgrad=True)
    sched = optim.lr_scheduler.ReduceLROnPlateau(optimizer=opt, factor=0.1, patience=5, threshold=0.01, verbose=True)

    check = train(args.e, opt, modelUsed, nn.CrossEntropyLoss(), train_load, sched, args.cuda)

    if args.mod == 1:
        encoder_state_dict = net.encoder_classification.modded_classification_head.state_dict()
    else:
        encoder_state_dict = net.encoder_classification.classification_head.state_dict()
    torch.save(encoder_state_dict, args.s)

    print("model saved")
