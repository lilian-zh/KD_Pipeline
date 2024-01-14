import torch
from torch.utils.data import DataLoader
from torchvision import transforms

import os
import yaml
import pickle

from datasets import DATASET_CLASSES

import os
import pickle

def create_data_loaders(dataset, train_path=None, val_path=None, test_path=None, 
                        train_transform=None, test_transform=None, 
                        num_workers=2, batch_size=32, is_train=True):
    if is_train and train_path and val_path:

        train_set = DATASET_CLASSES[dataset](train_path, is_train=True, transform=train_transform)
        val_set = DATASET_CLASSES[dataset](val_path, is_train=False, transform=test_transform)
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        
        train_length = len(train_set)
        img_shape = train_set[0][0].shape
        
        return train_loader, val_loader, train_length, img_shape
    elif not is_train and test_path:
        test_set = DATASET_CLASSES[dataset](test_path, is_train=False, transform=test_transform)
        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        return test_loader
    else:
        raise ValueError("Invalid configuration: Unable to create data loaders. Please provide valid dataset paths.")


def calculate_class_weights(class_counts):
    total_samples = sum(class_counts)
    class_frequencies = [count / total_samples for count in class_counts]

    class_weights = [1 / freq for freq in class_frequencies]
    # print(class_weights)
    class_weights = torch.tensor(class_weights)
    class_weights /= class_weights.sum() 

    return class_weights





if __name__ == '__main__':
    train_path = r'C:\Users\lili_\Documents\Thesis\hpc_kd4\data\02_intermediate\internal_data\train_set.pkl'
    val_path = r'C:\Users\lili_\Documents\Thesis\hpc_kd4\data\02_intermediate\internal_data\fold_1\val_set.pkl'
    test_path = r'C:\Users\lili_\Documents\Thesis\hpc_kd4\data\02_intermediate\internal_data\test_set.pkl'
    train_transform = transforms.Compose([
        transforms.Resize((250, 250)),
        #transforms.RandomCrop((128, 128)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    test_transform = transforms.Compose([
        transforms.Resize((250, 250)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    num_workers = 0
    batch_size = 32

    train_loader, val_loader, train_length, img_shape = create_data_loaders(
        dataset = 'Internal_BM',
        train_path=train_path,
        val_path=val_path,
        test_path=test_path,
        train_transform=train_transform,
        test_transform=test_transform,
        is_train=True
    )

    print(f"Training data length: {train_length}")
    print(f"Image shape: {img_shape}")

    test_loader = create_data_loaders(
        dataset = 'Internal_BM',
        test_path=test_path,
        test_transform=test_transform,
        is_train=False
    )
    print("Testing data loaded successfully")


    # class_counts = [166,40,409,511,11,535,128,865,450,475,1658,3242,3021]
    # weights = calculate_class_weights(class_counts)
    # print(weights)






