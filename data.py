from torch.utils.data import Dataset, random_split
import numpy as np
import torch

class MyEEGDataset(Dataset):
    def __init__(self, root="data", split='trainval', subject=0, augment=None):
        """
        set subject to -1 to load all data
        """

        X_train_valid = np.load(f'{root}/X_train_valid.npy')
        y_train_valid = np.load(f'{root}/y_train_valid.npy')
        X_test = np.load(f'{root}/X_test.npy')
        y_test = np.load(f'{root}/y_test.npy')
        person_train_valid = np.load(f'{root}/person_train_valid.npy').reshape(-1)
        person_test = np.load(f'{root}/person_test.npy').reshape(-1)

        # subtract 769 to make labels start from 0
        y_train_valid -= 769
        y_test -= 769

        if subject != -1:
            if split == 'trainval':
                self.X = X_train_valid[subject == person_train_valid]
                self.y = y_train_valid[subject == person_train_valid]
            else:
                self.X = X_test[subject == person_test]
                self.y = y_test[subject == person_test]
        else:
            if split == 'trainval':
                self.X = X_train_valid
                self.y = y_train_valid
            else:
                self.X = X_test
                self.y = y_test

        self.X = torch.tensor(self.X, dtype=torch.float32).unsqueeze(1)
        self.y = torch.tensor(self.y, dtype=torch.long)
        if augment is not None:
            if 'gaussian' in augment:
                self.X += torch.randn_like(self.X) * 0.01
            if 'channel_drop' in augment:
                random_index = torch.randint(0, 22, (1,)).item()
                self.X[:, :, random_index, :] = 0

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
def split_dataset(dataset, split=0.8):
    train_dataset, val_dataset = random_split(dataset, [split, 1-split])
    return train_dataset, val_dataset

# def custom_collate_fn(batch):
#     processed_batch = []

#     for data, label in batch:
#         # Noise Injection
#         noise_level = np.random.uniform(0.001, 0.005)
#         data += noise_level * np.random.normal(size=data.shape).astype(np.float32)
                
#         processed_batch.append((data, label))
    
#     # After processing, use the default collate function to handle the usual batching operations
#     return torch.utils.data.dataloader.default_collate(processed_batch)
