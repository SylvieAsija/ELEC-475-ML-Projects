import argparse
import datetime
import os
import model as net
import cv2
import torch
from custom_dataset import KittiROITrainDataset, KittiROITestDataset
from torchvision import transforms
from torch.utils.data import DataLoader
from KittiAnchors import Anchors
from KittiDataset import KittiDataset

def parse_arguments():
    parser = argparse.ArgumentParser(description='Arguments to pass to the train module')
    parser.add_argument('-cuda', type=int, default='1', help='device')
    parser.add_argument('-c', type=str, default='weights.pth', help='classification head weight path')
    parser.add_argument('-a', type=int, default=1, help='test accuracy')
    parser.add_argument('-i', type=int, default=1, help='test iou')
    parser.add_argument('-rd', type=str, default='./data/Kitti8_ROIs', help='roi dataset location')
    parser.add_argument('-d', type=str, default='./data/Kitti8', help='kitti dataset location')
    parser.add_argument('-t', type=str, default='./temp_images', help='temporary ROI folder')
    parser.add_argument('-o', type=str, default='./test_outputs', help='test outputs folder')

    argsUsed = parser.parse_args()
    return argsUsed

def subdivide(image):
    anchors = Anchors()
    anchor_centers = anchors.calc_anchor_centers(image.shape, anchors.grid)
    ROIs, boxes = anchors.get_anchor_ROIs(image, anchor_centers, anchors.shapes)

    return ROIs, boxes


def test_transform():
    transform_list = [
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ]
    return transforms.Compose(transform_list)


def test_accuracy(model, device):
    transform = test_transform()
    test_dataset = KittiROITrainDataset(dir=args.rd, training=False, transform=transform)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    true_pos = 0
    false_pos = 0
    true_neg = 0
    false_neg = 0
    correct = 0
    total = 0

    model.to(device)

    print(datetime.datetime.now())
    with torch.no_grad():
        model.eval()
        for img, labels in test_dataloader:
            img = img.to(device)
            labels = labels.to(device)
            outputs = model(img)
            outputs = (outputs >= 0.5).float()

            total += labels.size(0)

            if outputs == 1:
                if labels == 1:
                    true_pos += 1
                else:
                    false_neg += 1
            else:
                if labels == 0:
                    true_neg += 1
                else:
                    false_pos += 1

            correct += (outputs[0] == labels).sum()

    accuracy = 100 * correct / total

    print(f'True Positives: {true_pos}')
    print(f'True Negatives: {true_neg}')
    print(f'False Positives: {false_pos}')
    print(f'False Negatives: {false_neg}')

    print(f'Accuracy: {accuracy}%')

def test_iou(model, device):
    dataset = KittiDataset(args.d, training=False)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    model.to(device)

    iou_total = 0
    iou_count = 0

    for item in enumerate(dataset):
        idx = item[0]
        image = item[1][0]
        label = item[1][1]
        target_ROIs = [((sublist[2], sublist[3]), (sublist[4], sublist[5])) for sublist in label]

        ROIs, boxes = subdivide(image.copy())
        for x, ROI in enumerate(ROIs):
            cv2.imwrite(os.path.join(args.t, f"image_{str(x)}.png"), ROI)
        test_dataset = KittiROITestDataset(args.t, transform=transform)
        test_loader = DataLoader(test_dataset, batch_size=48, shuffle=False)

        image2 = image.copy()
        model.eval()
        with torch.no_grad():
            for img, labels in test_loader:
                img = img.to(device)
                labels = labels.to(device)
                outputs = model(img)
                outputs = torch.where(outputs >= 0.5, torch.tensor(1), torch.tensor(0))
                idxs = torch.nonzero(outputs == 1, as_tuple=False)

        for index, _ in idxs:
            box = boxes[index]
            box_check = ((box[0][1], box[0][0]), (box[1][1], box[1][0]))
            max_iou = Anchors().calc_max_IoU(box_check, target_ROIs)

            iou_total += max_iou
            iou_count += 1

            pt1 = (box[0][1], box[0][0])
            pt2 = (box[1][1], box[1][0])
            cv2.rectangle(image2, pt1, pt2, color=(0, 255, 255))

        cv2.imwrite(os.path.join(args.o, f"image_{str(idx)}.png"), image2)

    average_iou = iou_total / iou_count
    print(f"Average IoU: {average_iou}")


if __name__ == "__main__":
    args = parse_arguments()

    encoder = net.encoder_classify.encoder
    classifier = net.encoder_classify.simple_classification
    classifier.load_state_dict(torch.load(args.c))
    model = net.CustomNetwork(encoder, classifier)

    device = 'cuda'
    if args.cuda == 0:
        device = 'cpu'

    if args.a:
        test_accuracy(model, device)

    if args.i:
        test_iou(model, device)
