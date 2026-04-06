import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import timm
from timm import resolve_data_config
from timm.data.transforms_factory import create_transform
from load_data import CropData
import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

model = timm.create_model('diet3_small_patch16_224', pretrained=True, num_classes = 10)
model.to(device)

img_dir = "A3_Dataset"
train_csv = "A3_Dataset/train.csv"
val_csv = "A3_Dataset/validation.csv"
test_csv = "A3_Dataset/test.csv"
EPOCHS = 10
batch_size=32
train_dataset = CropData(img_dir_path=img_dir, csv_file_path=train_csv)
val_dataset = CropData(img_dir_path=img_dir, csv_file_path=val_csv)
test_dataset = CropData(img_dir_path=img_dir, csv_file_path=test_csv)

num_workers = 4
pin_memory = device.type == "cuda"
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

 

