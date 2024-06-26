import argparse
import torch
import network as net
from torchvision.datasets import CIFAR10, CIFAR100
from torchvision import transforms
from torch.utils.data import DataLoader


def test_transform():
    transform_list = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ]
    return transforms.Compose(transform_list)


def parse_arguments():
    parser = argparse.ArgumentParser(description='Arguments to pass to the train module')
    parser.add_argument('-cuda', type=str, default='cuda:0', help='device')
    parser.add_argument('-e', type=str, default='encoder.pth', help='encoder weight path')
    parser.add_argument('-ch', type=str, default='classification_head_10.pth', help='classification head weight path')
    parser.add_argument('-c', type=int, default=10, help='CIFAR 10 or 100 dataset')
    parser.add_argument('-b', type=int, default=512, help='batch size')
    parser.add_argument('-mod', type=int, default=0, help='Modded architecture (1) or vanilla (0)')

    argsUsed = parser.parse_args()
    return argsUsed


if __name__ == "__main__":

    args = parse_arguments()

    if args.c == 100:
        test_set = CIFAR100('./data/CIFAR100_test', train=False, download=True, transform=test_transform())
    else:
        test_set = CIFAR10('./data/CIFAR10_test', train=False, download=True, transform=test_transform())

    test_loader = DataLoader(test_set, batch_size=args.b, shuffle=True)

    encoder = net.encoder_classification.encoder
    encoder.load_state_dict(torch.load(args.e))

    if args.mod == 1:
        classification_head = net.encoder_classification.modded_classification_head
    else:
        classification_head = net.encoder_classification.classification_head

    classification_head.load_state_dict(torch.load(args.ch))

    model = net.network(encoder, classification_head)

    correct_top1 = 0
    correct_top5 = 0
    total = 0

    with torch.no_grad():
        for img, labels in test_loader:
            outputs = model(img)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct_top1 += (predicted == labels).sum().item()

            _, predicted_top5 = torch.topk(outputs.data, 5, dim=1)
            correct_top5 += sum([label in predicted_top5[i] for i, label in enumerate(labels)])

    top1_accuracy = 100 * correct_top1 / total
    top5_accuracy = 100 * correct_top5 / total

    print(f'Top-1 Accuracy: {top1_accuracy}%')
    print(f'Top-5 Accuracy: {top5_accuracy}%')
