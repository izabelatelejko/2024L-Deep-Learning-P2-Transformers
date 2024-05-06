"""Module for training related functions."""

import random

import numpy as np
import torch
from tqdm import tqdm

from src.const import SEED


def train_model(
    model,
    criterion,
    optimizer,
    train_loader,
    val_loader,
    model_type,
    epoch_count,
    using_pretrained=False,
    early_stopping=False,
    patience=3,
    device="cpu",
):

    train_losses = []
    val_losses = []
    best_val_loss = float("inf")
    no_improve_count = 0
    is_stopped_early = False
    best_model_dict = None

    torch.manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)

    # Training
    for epoch in range(epoch_count):
        model.train()
        train_loss = 0
        correct_train = 0
        total_train = 0

        for batch_X, batch_y in tqdm(train_loader, f"Epoch {epoch+1}"):
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            model.zero_grad()
            outputs = model(batch_X)

            if using_pretrained:
                outputs = outputs.logits

            if model_type == "bin":
                loss = criterion(outputs, batch_y.unsqueeze(1))
            else:
                loss = criterion(outputs, batch_y.long())

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # Loss
            train_loss += loss.item() * batch_X.size(0)

            # Accuracy
            if model_type == "bin":
                predicted = (outputs > 0.5).float()
                correct_train += (predicted == batch_y.unsqueeze(1)).sum().item()
            else:
                predicted = torch.argmax(outputs, dim=1)
                correct_train += (predicted == batch_y).sum().item()

            total_train += batch_y.size(0)

        train_loss /= len(train_loader.dataset)
        train_acc = correct_train / total_train
        train_losses.append(train_loss)

        model.eval()
        val_loss = 0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)

                if using_pretrained:
                    outputs = outputs.logits

                if model_type == "bin":
                    loss = criterion(outputs, batch_y.unsqueeze(1))
                else:
                    loss = criterion(outputs, batch_y.long())

                # Loss
                val_loss += loss.item() * batch_X.size(0)

                # Accuracy
                if model_type == "bin":
                    predicted = (outputs > 0.5).float()
                    correct_val += (predicted == batch_y.unsqueeze(1)).sum().item()
                else:
                    predicted = torch.argmax(outputs, dim=1)
                    correct_val += (predicted == batch_y).sum().item()
                total_val += batch_y.size(0)

            val_loss /= len(val_loader.dataset)
            val_acc = correct_val / total_val
            val_losses.append(val_loss)

        print(
            f"Epoch {epoch+1}/{epoch_count}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}"
        )

        if not early_stopping:
            continue

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve_count = 0
            best_model_dict = model.state_dict()
        else:
            no_improve_count += 1

        if no_improve_count >= patience:
            print("Early stopping")
            is_stopped_early = True
            break

    if is_stopped_early:
        model.load_state_dict(best_model_dict)
    return train_losses, val_losses
