import matplotlib.pyplot as plt
import torch
import argparse
from torchvision import transforms
from torchvision.datasets import MNIST
from model import autoencoderMLP4Layer
from torch.utils.data import DataLoader


# Get inputs from command line
def parse_arguments():
    parser = argparse.ArgumentParser(description='Arguments to pass to the lab1.py module')
    parser.add_argument('-l', '--weight_file', type=str, default='MLP.8.pth', help='Path to load the model weights')

    argsUsed = parser.parse_args()
    return argsUsed


args = parse_arguments()

# Initialize the training set
train_transform = transforms.Compose([transforms.ToTensor()])
train_set = MNIST('./data/MNIST', train=True, download=True, transform=train_transform)

print("Input an integer value between 0 and 59999")
datasetImageToGrab = input()
print(train_set.targets[int(datasetImageToGrab)])
plt.imshow(train_set.data[int(datasetImageToGrab)], cmap='gray')
plt.show()

# Instantiate the model
modelUsed = autoencoderMLP4Layer()
print(torch.cuda.is_available())
torch.zeros(1).cuda()
modelUsed = modelUsed.to('cuda:0')

# Load the trained model
modelUsed.load_state_dict(torch.load(args.weight_file))

# Set the model to evaluation mode
modelUsed.eval()

# Define the transform for loading and preprocessing images
transform = transforms.Compose([transforms.ToTensor()])

# Load the MNIST dataset
test_set = MNIST('./data/MNIST', train=False, download=True, transform=transform)

# Define a DataLoader for the test set
test_loader = DataLoader(test_set, batch_size=1, shuffle=True)


# Define a function to interpolate the tensors
def interpolate_tensors(start, end, steps):
    with torch.no_grad():
        coefficients = torch.linspace(0, 1, steps, device='cuda:0').view(-1, 1)
        interpolated_tensors = torch.lerp(start, end, coefficients)
        return interpolated_tensors


# Generate 2 different random numbers to use for interpolation
random_index1 = int(torch.rand(1) * len(test_set))
random_index2 = random_index1
while random_index2 == random_index1:
    random_index2 = int(torch.rand(1) * len(test_set))
random_image1 = test_set[random_index1][0].view(1, -1).to('cuda:0')
random_image2 = test_set[random_index2][0].view(1, -1).to('cuda:0')
with torch.no_grad():
    start_bottleneck = modelUsed.encode(random_image1).to('cuda:0')
    end_bottleneck = modelUsed.encode(random_image2).to('cuda:0')

    # Call the interpolate function to determine bottleneck sizes
    interpolated_bottlenecks = interpolate_tensors(start_bottleneck, end_bottleneck, 10)

    # Plot the figures
    plt.figure(figsize=(15, 5))
    for i, bottleneck in enumerate(interpolated_bottlenecks):
        reconstructed_image = modelUsed.decode(bottleneck).view(28, 28)
        plt.subplots_adjust(wspace=0.2)
        plt.subplot(1, 10, i + 1)
        plt.imshow(reconstructed_image.cpu(), cmap='gray')
        if i == 0:
            plt.title('Start Image')
        elif i == len(interpolated_bottlenecks) - 1:
            plt.title('End Image')
        else:
            plt.title(f'Step {i}')
        plt.axis('off')
    plt.show()


# Define a function to add noise to an image
def add_noise(image):
    noise = torch.randn_like(image) * noise_factor
    noisy_image = image + noise
    return torch.clamp(noisy_image, 0., 1.)


# Set a noise factor for a certain noise strength
noise_factor = 0.3

# Plot the figures, first the original image and its reconstruction, then the noisy image and its reconstruction
for imgs, labels in test_loader:
    f = plt.figure()
    f.subplots_adjust(hspace=0.5)
    original = f.add_subplot(2, 2, 1)
    original.title.set_text('Original Image')
    plt.imshow(imgs[0][0].cpu(), cmap='gray')  # Convert to numpy array
    reconstructed = f.add_subplot(2, 2, 2)
    with torch.no_grad():
        output = modelUsed(imgs.view(1, -1).to('cuda:0')).view(1, 1, 28, 28)  # Perform inference on GPU
    reconstructed.title.set_text('Reconstructed Image')
    plt.imshow(output[0][0].cpu().detach().numpy(), cmap='gray')  # Convert to numpy array
    # plt.show()

    noisy_img = add_noise(imgs.to('cuda:0'))
    noised = f.add_subplot(2, 2, 3)
    noised.title.set_text("Original Image with Noise")
    plt.imshow(noisy_img[0][0].cpu(), cmap='gray')
    denoised = f.add_subplot(2, 2, 4)
    with torch.no_grad():
        denoised_img = modelUsed(noisy_img.view(1, -1)).to('cuda:0').view(1, 1, 28, 28)
    denoised.title.set_text("Denoised Image")
    plt.imshow(denoised_img[0][0].cpu().detach().numpy(), cmap='gray')
    plt.show()
