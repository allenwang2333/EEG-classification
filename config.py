import argparse

parser = argparse.ArgumentParser(description='Please specify train details.')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='Momentum')
parser.add_argument('--seed', type=int, default=42, help='Random seed')

FLAGS = parser.parse_args()