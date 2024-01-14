from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch
from PIL import Image
import pickle
# import traceback

class BaseDataset(Dataset):
    def __init__(self, data_path, is_train=True, transform=None):
        self.is_train = is_train
        if transform is None:
            self.transform = self._default_transform()
        else:
            self.transform = transform

        with open(data_path, 'rb') as f:
            self.data = pickle.load(f)

    def _default_transform(self):
        if self.is_train:
            return transforms.Compose([
                transforms.Resize((150, 150)),
                transforms.RandomCrop((128, 128)),
                transforms.RandomHorizontalFlip(),
                # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                # other required data transformations
            ])
        else:
            return transforms.Compose([
                transforms.Resize((128, 128)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                # other required data transformations
            ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path, label = self.data[idx]
        try:
            image = Image.open(image_path).convert('RGB')
            image = self.transform(image)
            label_tensor = torch.tensor(label)  # convert the lable to tensor
            return image, label_tensor, idx
        except Exception as e:
            print(f"Error loading image at index {idx}, path {image_path}: {e}")
            return None, None, idx



class Internal_BM(BaseDataset):
    def __init__(self, data_path, is_train=True, transform=None):
        super().__init__(data_path=data_path, is_train=is_train, transform=transform)


class LMU(BaseDataset):
    def __init__(self, data_path, is_train=True, transform=None):
        super().__init__(data_path=data_path, is_train=is_train, transform=transform)
    def _default_transform(self):
        if self.is_train:
            return transforms.Compose([
                transforms.Resize((400, 400)),
                #transforms.RandomCrop((128, 128)),
                transforms.RandomHorizontalFlip(),
                # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                # 其他需要的数据转换
            ])
        else:
            return transforms.Compose([
                transforms.Resize((400, 400)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                # 其他需要的数据转换
            ])

class BM_Cytomorphology(BaseDataset):
    def __init__(self, data_path, is_train=True, transform=None):
        super().__init__(data_path=data_path, is_train=is_train, transform=transform)
    def _default_transform(self):
        if self.is_train:
            return transforms.Compose([
                transforms.Resize((250, 250)),
                #transforms.RandomCrop((128, 128)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                # 其他需要的数据转换
            ])
        else:
            return transforms.Compose([
                transforms.Resize((250, 250)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                # 其他需要的数据转换
            ])


if __name__ == '__main__':
    # 加载数据集的 pickle 文件
    train_file = r'C:\Users\lili_\Documents\Thesis\hpc_kd4\data\02_intermediate\internal_data\train_set.pkl' 
    # test_file = r'C:\Users\lili_\Documents\Thesis\hpc_kd3\data\01_raw\lmu_png\test_set.pkl' 

    # 创建自定义的 Dataset 对象并使用其中的数据
    train_dataset = Internal_BM(train_file, is_train=True)
    # test_dataset = CustomDataset(test_file, is_train=False)

    # 使用 DataLoader 加载数据集
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    # test_loader = DataLoader(test_dataset, batch_size=batch_size // 2, shuffle=False)

    # 打印第一个训练样本的形状
    print(train_dataset[0][0].shape)  # torch.Size([3, 224, 224])

