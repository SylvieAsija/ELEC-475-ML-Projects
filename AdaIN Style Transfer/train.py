import os.path
import argparse
import torch
from torch.utils.data import DataLoader
from torch.optim.adam import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from datetime import datetime
from AdaIN_net import AdaIN_net, encoder_decoder
from torchvision import transforms
import matplotlib.pyplot as plt
from custom_dataset import custom_dataset
import time

# Setting up argparse for commandline arguments
parser = argparse.ArgumentParser()
parser.add_argument("-z", "--bottleneck", type=int)
parser.add_argument("-e", "--epochs", type=int)
parser.add_argument("-b", "--batch_size")
parser.add_argument("-content_dir", "--content_dir", type=str)
parser.add_argument("-style_dir", "--style_dir", type=str)
parser.add_argument("-gamma", "--gamma", type=str)
parser.add_argument("-l", "--encoder_filepath", type=str)
parser.add_argument("-s", "--decoder_filepath", type=str)
parser.add_argument("-p", "--decoder_image", type=str)
parser.add_argument("-cuda", "--use_cuda", type=str)

opt = parser.parse_args()
num_epochs = opt.epochs
decoder_image = opt.decoder_image
decoder_file = opt.decoder_filepath
encoder_file = "../encoder.pth"
if opt.encoder_filepath:
    encoder_file = opt.encoder_filepath
gamma = opt.gamma
use_cuda = False
if opt.use_cuda == 'y' or opt.use_cuda == 'Y':
    use_cuda = True
out_dir = '../output/'
os.makedirs(out_dir, exist_ok=True)
content_dir = opt.content_dir
style_dir = opt.style_dir
batch_size = int(opt.batch_size)

filepath = os.path.join(os.getcwd(), decoder_file)
intermediate_filepath = os.path.join(os.getcwd(), "intermediates", decoder_file)


def train(num_epochs: int, optimizer: torch.optim.Optimizer, model: AdaIN_net,
          content_loader, style_loader, scheduler, gamma, device):
    model.train()
    losses_train = []
    content_losses_train = []
    style_losses_train = []
    num_batches = len(content_loader)

    for epoch in range(1, num_epochs + 1):
        print(f"Epoch #{epoch}")
        loss_train = 0.0
        content_loss_train = 0.0
        style_loss_train = 0.0

        for batch in range(num_batches):
            content_images = next(iter(content_loader)).to(device)
            style_images = next(iter(style_loader)).to(device)

            content_loss, style_loss = model(content_images, style_images)

            optimizer.zero_grad()

            style_loss = style_loss * float(gamma)
            loss = content_loss + style_loss
            loss.backward()
            optimizer.step()
            loss_train += loss.item()
            content_loss_train += content_loss.item()
            style_loss_train += style_loss.item()

        training_loss = loss_train / num_batches
        content_training_loss = content_loss_train / num_batches
        style_training_loss = style_loss_train / num_batches

        scheduler.step(loss_train)
        losses_train += [training_loss]
        content_losses_train += [content_training_loss]
        style_losses_train += [style_training_loss]

        print(
            f"{datetime.now()} Epoch {epoch}, Training Loss {training_loss}, Content Loss {content_training_loss}, Style Loss {style_training_loss}")

    plt.plot(losses_train, label='Total Loss')
    plt.plot(content_losses_train, label='Content Loss')
    plt.plot(style_losses_train, label='Style Loss')
    plt.legend()
    plt.savefig(decoder_image)


torch.cuda.empty_cache()
encoder = encoder_decoder.encoder
encoder.load_state_dict(torch.load(encoder_file, map_location='cpu'))
model = AdaIN_net(encoder)
if use_cuda:
    model.to('cuda')

learning_rate = 1e-4
adam_optimizer = Adam(params=model.decoder.parameters(), lr=learning_rate)
reduce_on_plateau_scheduler = ReduceLROnPlateau(optimizer=adam_optimizer, factor=0.1, patience=5, threshold=1e-2,
                                                verbose=True)

train_transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((224, 224), antialias=True)])
content_train = custom_dataset(content_dir, transform=train_transform)
style_train = custom_dataset(style_dir, transform=train_transform)
content_loader = DataLoader(content_train, batch_size=batch_size, shuffle=True)
style_loader = DataLoader(style_train, batch_size=batch_size, shuffle=True)

start_time = time.time()
train(num_epochs=num_epochs, optimizer=adam_optimizer, model=model, content_loader=content_loader,
      style_loader=style_loader,
      scheduler=reduce_on_plateau_scheduler, gamma=gamma, device='cuda')  # call train function

torch.save(model.decoder.state_dict(), filepath)  # save model
end_time = time.time()

elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time} seconds")