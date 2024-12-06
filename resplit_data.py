import os
from glob import glob
from tqdm import tqdm

base_dir = "Fruits_Classification/"
train_dir = os.path.join(base_dir, "train")
valid_dir = os.path.join(base_dir, "valid")
test_dir = os.path.join(base_dir, "test")

class_names = ["Apple", "Banana", "Grape", "Mango", "Strawberry"]


def collect_images():
    all_images = {class_name: [] for class_name in class_names}
    for class_name in class_names:
        # Collect images from current train, valid, and test directories
        for split_dir in [train_dir, valid_dir, test_dir]:
            class_path = os.path.join(split_dir, class_name)
            if os.path.exists(class_path):
                all_images[class_name].extend(glob(os.path.join(class_path, "*")))
    return all_images


all_images = collect_images()

for class_name, images in all_images.items():

    total_images = len(images)
    train_end = int(total_images * 0.8)
    valid_end = train_end + int(total_images * 0.1)

    train_images = images[:train_end]
    valid_images = images[train_end:valid_end]
    test_images = images[valid_end:]

    print(
        f"{class_name} - Total: {total_images}, Train: {len(train_images)}, Valid: {len(valid_images)}, Test: {len(test_images)}"
    )

    for dataset, dataset_dir in [
        (train_images, train_dir),
        (valid_images, valid_dir),
        (test_images, test_dir),
    ]:
        class_dir = os.path.join(dataset_dir, class_name)
        if not os.path.exists(class_dir):
            os.makedirs(class_dir)

        for image_path in tqdm(
            dataset, desc=f"Organizing {class_name} -> {dataset_dir}"
        ):
            new_path = os.path.join(class_dir, os.path.basename(image_path))
            os.rename(image_path, new_path)

for split_dir in [train_dir, valid_dir, test_dir]:
    for class_name in class_names:
        class_path = os.path.join(split_dir, class_name)
        if os.path.exists(class_path) and not os.listdir(class_path):
            os.rmdir(class_path)

print("\nDataset successfully reorganized into 80-10-10 split!")
