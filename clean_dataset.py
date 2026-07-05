from PIL import Image
import os

dataset_path = "dataset"

bad_images = []

for root, dirs, files in os.walk(dataset_path):

    for file in files:

        path = os.path.join(root, file)

        try:
            with Image.open(path) as img:
                img.verify()

        except Exception as e:

            print("Corrupted image:", path)

            bad_images.append(path)

            try:
                os.remove(path)
                print("Removed:", path)

            except:
                print("Could not remove, delete manually:", path)

print("\nDataset cleaning complete")
print("Total corrupted images:", len(bad_images))