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

subjectIndex = range(0, 9)
all_val_acc = []
all_test_acc = []
all_val_loss = []
all_test_loss = []
for subject in subjectIndex:
    print(f"train on {subject}, test on {subject}")
    train_val_dataset = MyEEGDataset(split='trainval', subject=subject)
    test_dataset = MyEEGDataset(split='test', subject=subject)

    train_dataset, val_dataset = split_dataset(train_val_dataset, split=0.8)

    train_loader = DataLoader(train_dataset, batch_size=FLAGS.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=FLAGS.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=FLAGS.batch_size, shuffle=False)

    # model = ResNet(num_classes=4)
    # model_list = [EEGNet() for _ in range(5)]
    # for model in model_list:
    model = EEGAttentionNet()

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
    _ , val_loss_history, train_acc_history, val_acc_history = train(model, train_loader, val_loader, criterion, optimizer, scheduler, FLAGS.epochs, FLAGS.device)

    if FLAGS.model is not None:
        model.load_state_dict(torch.load(f'{FLAGS.model}.pth', map_location='cpu'))
        model = model.to(FLAGS.device)
    # avg_loss, accuracy = evaluate_ensemble(model, test_loader, criterion, FLAGS.device)
    avg_loss, accuracy = evaluate(model, test_loader, criterion, FLAGS.device)
    all_test_acc.append(accuracy)
    all_test_loss.append(avg_loss)
    all_val_acc.append(val_acc_history[-1])
    all_val_loss.append(val_loss_history[-1])
    print(f'Test set: Average loss = {avg_loss:.4f}, Accuracy = {accuracy:.4f}')

for i in range(0, 9):
    print(f"subject {i} val acc: {all_val_acc[i]}, test acc: {all_test_acc[i]}, val loss: {all_val_loss[i]}, test loss: {all_test_loss[i]}")
# x = np.arange(FLAGS.epochs)
# plt.plot(x, train_acc_history, label='train')
# plt.plot(x, val_acc_history, label='val')
# plt.legend()
# plt.show()

# python main.py --device cuda --epoch 60 --batch_size 16
