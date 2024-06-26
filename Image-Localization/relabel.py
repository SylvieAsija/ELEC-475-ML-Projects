import cv2
import numpy as np
import csv
import os


def resize_image(image, image_name, target_size):
    # Resize image while maintaining aspect ratio
    h, w = image.shape[:2]
    aspect_ratio = target_size[0] / w if w > h else target_size[1] / h
    new_size = (int(w * aspect_ratio), int(h * aspect_ratio))
    resized_image = cv2.resize(image, new_size)
    # Padding to match the target size (if necessary)
    top_pad = (target_size[1] - new_size[1]) // 2
    bottom_pad = target_size[1] - new_size[1] - top_pad
    left_pad = (target_size[0] - new_size[0]) // 2
    right_pad = target_size[0] - new_size[0] - left_pad
    resized_image = cv2.copyMakeBorder(resized_image, top_pad, bottom_pad, left_pad, right_pad, cv2.BORDER_CONSTANT,
                                       value=0)

    # write to a directory you want your images to end up
    cv2.imwrite(os.path.join("./data/train/resized_images", image_name), resized_image)
    return resized_image, aspect_ratio, top_pad, left_pad


def resize_annotations(x, y, aspect_ratio, top_pad, left_pad):
    scaled_x = (x * aspect_ratio) + left_pad
    scaled_y = (y * aspect_ratio) + top_pad
    return scaled_x, scaled_y


if __name__ == '__main__':
    images = []
    noses = []

    # do this for directory and file for both train and test
    label_file = f"data/train/labels/labels.txt"
    with open(label_file) as csvfile:
        reader = csv.DictReader(csvfile)

        i = 0
        for row in reader:
            images.append(row['image'])
            noses.append(row['nose'])
            print(noses[i])
            i += 1

    # directory and file you want your new resized labels.txt to end up
    new_label_file = f"./data/resized_labels/labels.txt"

    # directory where your images currently are
    image_file = f"./data/train/images"
    with (open(new_label_file, "w") as f):
        for x in range(len(noses)):
            print(len(noses))

            image = cv2.imread(os.path.join(image_file, images[x]))

            coords_cleaned = noses[x].replace('(', '').replace(')', '').split(',')
            x_coord = float(coords_cleaned[0])
            y_coord = float(coords_cleaned[1])

            print("x coordinate:", x_coord, "y coordinate:", y_coord)

            if image is not None:
                resized_img, aspect_ratio, ypad, xpad = resize_image(image, images[x], target_size=(224, 224))

                resized_x, resized_y = resize_annotations(x_coord, y_coord, aspect_ratio, ypad, xpad)

                toWrite = images[x] + ',' + '"(' + str(int(resized_x)) + ',' + str(int(resized_y)) + ')"' + '\n'
                toWrite.replace(" ", "")
                f.write(toWrite)
                print(x)
            else:
                print(images[x])


