import torch
from tqdm import tqdm

def train(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
    model = model.to(device)
    train_loss_history = []
    val_loss_history = []
    for epoch in range(num_epochs):
        model.train()
        with tqdm(total=len(train_loader), desc=f'Epoch {epoch + 1}/{num_epochs}', position=0, leave=True) as pbar:
            train_loss = 0.0
            counter = 0
            for inputs, labels in train_loader:
                
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()

                logits = model(inputs)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()

                pbar.update(1)
                pbar.set_postfix(loss=loss.item())
                train_loss += loss.item()
                counter += 1
            avg_train_loss = train_loss / counter
        train_loss_history.append(avg_train_loss)

        avg_loss, accuracy = evaluate(model, val_loader, criterion, device)
        val_loss_history.append(avg_loss)
        print(f'Validation set: Average loss = {avg_loss:.4f}, Accuracy = {accuracy:.4f}')
    return train_loss_history, val_loss_history

def evaluate(model, test_loader, criterion, device):
    model.eval()

    with torch.no_grad():
        total_loss = 0.0
        num_correct = 0
        num_samples = 0

        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            logits = model(inputs)
            loss = criterion(logits, labels)
            total_loss += loss.item()

            _, predictions = torch.max(logits, dim=1)
            num_correct += (predictions == labels).sum().item()
            num_samples += len(inputs)
    avg_loss = total_loss / len(test_loader)
    accuracy = num_correct / num_samples

    return avg_loss, accuracy