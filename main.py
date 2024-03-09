from config import FLAGS
import numpy as np
import torch
from train import train, evaluate

from data import MyEEGDataset, split_dataset
from torch.utils.data import DataLoader
from model import ResNet18
# Load data

torch.manual_seed(FLAGS.seed)
np.random.seed(FLAGS.seed)

train_val_dataset = MyEEGDataset(split='trainval', subject=-1)
test_dataset = MyEEGDataset(split='test', subject=0)

train_dataset, val_dataset = split_dataset(train_val_dataset, split=0.8)

train_loader = DataLoader(train_dataset, batch_size=FLAGS.batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=FLAGS.batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=FLAGS.batch_size, shuffle=False)

model = ResNet18(num_classes=4)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=FLAGS.lr)

train(model, train_loader, val_loader, criterion, optimizer, FLAGS.epochs, FLAGS.device)
avg_loss, accuracy = evaluate(model, test_loader, criterion, FLAGS.device)
print(f'Test set: Average loss = {avg_loss:.4f}, Accuracy = {accuracy:.4f}')
