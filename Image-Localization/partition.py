import csv
import shutil

images = []
images2 = []
noses = []

# do this for directory and file for both train and test
label_file = f"./data/train/train_noses.3.txt"

# starting directory for where your images are
original_file_path = f"./data/"

# ending directory for where you want your images to end up
new_file_path = f"./data/train/images"

with open(label_file) as csvfile:
    reader = csv.DictReader(csvfile)

    for row in reader:
        images.append(row['image'])
        images2.append(row['image'])
        noses.append(row['nose'])

    i = 0
    for image in images:
        images[i] = original_file_path + image
        images2[i] = new_file_path + image

        # print(images[i], images2[i])
        shutil.move(images[i], images2[i])

        i += 1

