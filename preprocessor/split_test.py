import os
import shutil
import random

dataset_dir = r'E:\Thesis\datasets\BM_cytomorphology_data'  # Replace with the path to your dataset directory
train_val_dir = r'E:\Thesis\datasets\BM_cytomorphology_data\train_val'  # Replace with the desired directory for the training set
test_dir = r'E:\Thesis\datasets\BM_cytomorphology_data\test'  # Replace with the desired directory for the test set

# Create directories for the training and test sets
os.makedirs(train_val_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# Get the list of folders for each class in the dataset
class_folders = os.listdir(dataset_dir)

for class_folder in class_folders:
    class_path = os.path.join(dataset_dir, class_folder)
    images = os.listdir(class_path)
    random.shuffle(images)  # Shuffle the order of images for the current class

    # Ensure at least one image is in the test set
    split_index = max(1, int(0.9 * len(images)))

    train_images = images[:split_index]
    test_images = images[split_index:]

    # Move images to the corresponding training and test set directories
    for img in train_images:
        src = os.path.join(class_path, img)
        dst = os.path.join(train_val_dir, class_folder, img)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy(src, dst)

    for img in test_images:
        src = os.path.join(class_path, img)
        dst = os.path.join(test_dir, class_folder, img)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy(src, dst)
