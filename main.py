from config import FLAGS
import numpy as np
import torch
from train import train, evaluate, evaluate_ensemble
from torch import mps
from data import MyEEGDataset, split_dataset
from torch.utils.data import DataLoader
from model import *
import matplotlib.pyplot as plt
# Load data

torch.manual_seed(FLAGS.seed)
np.random.seed(FLAGS.seed)

train_val_dataset = MyEEGDataset(split='trainval', subject=-1)
test_dataset = MyEEGDataset(split='test', subject=-1)

train_dataset, val_dataset = split_dataset(train_val_dataset, split=0.8)

train_loader = DataLoader(train_dataset, batch_size=FLAGS.batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=FLAGS.batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=FLAGS.batch_size, shuffle=False)

# model = ResNet(num_classes=4)
model = EEGAttentionNet()
print(model)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
#optimizer = torch.optim.SGD(model.parameters(), lr=FLAGS.lr, momentum=FLAGS.momentum)



_ , val_loss_history, train_acc_history, val_acc_history = train(model, train_loader, val_loader, criterion, optimizer, FLAGS.epochs, FLAGS.device)
x = np.arange(FLAGS.epochs)
plt.plot(x, train_acc_history, label='train')
plt.plot(x, val_acc_history, label='val')
plt.legend()
plt.show()
avg_loss, accuracy = evaluate(model, test_loader, criterion, FLAGS.device)
#avg_loss, accuracy = evaluate_ensemble(model_list, test_loader, criterion, FLAGS.device)
print(f'Test set: Average loss = {avg_loss:.4f}, Accuracy = {accuracy:.4f}')
