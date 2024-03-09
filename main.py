from config import FLAGS
import numpy as np
import torch
from train import train, validate

from data import MyEEGDataset, split_dataset

# Load data
train_val_dataset = MyEEGDataset(split='trainval', subject=0)
test_dataset = MyEEGDataset(split='test', subject=0)

train_dataset, val_dataset = split_dataset(train_val_dataset, split=0.8)


