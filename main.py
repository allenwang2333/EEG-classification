from config import FLAGS
import numpy as np
import torch
from train import train, evaluate

from data import MyEEGDataset, split_dataset
from torch.utils.data import DataLoader
from model import ResNet, HybridCNNLSTMModel
import matplotlib.pyplot as plt
# Load data

torch.manual_seed(FLAGS.seed)
np.random.seed(FLAGS.seed)

train_val_dataset = MyEEGDataset(split='trainval', subject=-1)
test_dataset = MyEEGDataset(split='test', subject=0)

train_dataset, val_dataset = split_dataset(train_val_dataset, split=0.8)

train_loader = DataLoader(train_dataset, batch_size=FLAGS.batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=FLAGS.batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=FLAGS.batch_size, shuffle=False)

# model = ResNet(num_classes=4)
model = HybridCNNLSTMModel()
print(model)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=FLAGS.lr)

train_loss_history, val_loss_history = train(model, train_loader, val_loader, criterion, optimizer, FLAGS.epochs, FLAGS.device)
x = np.arange(FLAGS.epochs)
plt.plot(x, train_loss_history, label='train')
plt.plot(x, val_loss_history, label='val')
plt.legend()
plt.show()
avg_loss, accuracy = evaluate(model, test_loader, criterion, FLAGS.device)
print(f'Test set: Average loss = {avg_loss:.4f}, Accuracy = {accuracy:.4f}')
