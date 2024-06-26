import datetime
import torch
import torchsummary
import argparse
from matplotlib import pyplot as plt
from torch import optim, nn
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from model import (autoencoderMLP4Layer)
from torchvision import transforms


def main():
    print("You're in main")


# Get inputs from command line
def parse_arguments():
    parser = argparse.ArgumentParser(description='Arguments to pass to the train module')
    parser.add_argument('-z', '--bottleneck_size', type=int, default=8, help='Size of the bottleneck layer')
    parser.add_argument('-e', '--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('-b', '--batch_size', type=int, default=2048, help='Batch size for training')
    parser.add_argument('-s', '--weight_file', type=str, default='MLP.8.pth', help='Path to save the model weights')
    parser.add_argument('-p', '--loss_plot', type=str, default='loss_plot.png', help='Path to save the loss plot')

    argsUsed = parser.parse_args()
    return argsUsed


# Train the model
def train(n_epochs, optimizer, model, loss_fn, train_loader, scheduler, device):
    print('training...')
    model.train()
    losses_train = []

    # Iterate through each image in train loader for every epoch
    for epoch in range(1, n_epochs + 1):
        print('epoch', epoch)
        loss_train = 0.0
        for img, _ in train_loader:
            img = img.view(img.size(0), -1).to(device=device)  # Move batch to device
            outputs = model(img)
            loss = loss_fn(outputs, img)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_train += loss.item()

        scheduler.step(loss_train)

        losses_train += [loss_train / len(train_loader)]

        print('{} Epoch {}. Training Loss {}'.format(
            datetime.datetime.now(), epoch, loss_train / len(train_loader)))

        # Save, plot, and display the figure
        plt.savefig(args.loss_plot)
        plt.plot(losses_train)
        if epoch == 50:
            plt.show()


# Initializations and function calls
args = parse_arguments()

train_transform = transforms.Compose([transforms.ToTensor()])
train_set = MNIST('./data/MNIST', train=True, download=True, transform=train_transform)

modelUsed = autoencoderMLP4Layer()
print(torch.cuda.is_available())
torch.zeros(1).cuda()
modelUsed = modelUsed.to('cuda:0')

# Create a summary of the model
torchsummary.summary(modelUsed, (1, 28 * 28))

# Initialize some parameters for the train function
lr = 0.001
opt = optim.Adam(params=modelUsed.parameters(), lr=0.001, weight_decay=0.00001)
train(args.epochs, opt, modelUsed, nn.MSELoss(), DataLoader(train_set, batch_size=args.batch_size, shuffle=True),
      optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.9), 'cuda:0')

# Save the weights
torch.save(modelUsed.state_dict(), args.weight_file)

if __name__ == "__main__":
    main()
