import argparse

parser = argparse.ArgumentParser(description='Please specify train details.')
parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='Momentum')
parser.add_argument('--seed', type=int, default=3407, help='Random seed')
parser.add_argument('--device', type=str, default='mps', help='Device to train on')
parser.add_argument('--data_aug', type=bool, default=False, help='Whether to use data augmentation')

FLAGS = parser.parse_args()