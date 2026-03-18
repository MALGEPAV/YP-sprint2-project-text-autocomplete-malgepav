from tqdm import tqdm
import torch

def train_epoch(model, train_loader, loss_fn, optimizer, epoch, device):
    epoch_loss = 0.0
    model.train()
    for X, y, lengths in tqdm(train_loader, desc=f"Epoch {epoch}:"):
        X = X.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        y_pred = model.forward(X, lengths)
        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.detach().cpu().item()
    epoch_loss /= len(train_loader)
    return epoch_loss

