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


