"""
train.py
========
Phase 3: The Training Loop

A single reusable `train_model` function drives BOTH experiments:
    - Experiment 1 (Baseline): criterion = nn.CrossEntropyLoss()
    - Experiment 2 (Imbalance-Aware): criterion = nn.CrossEntropyLoss(weight=class_weights)

Everything else (architecture, optimizer type, data, augmentation, number of
epochs) is held constant between the two runs -- this is essential for a
fair ablation. The ONLY variable that changes is the loss function. That is
what lets you correctly attribute any change in Recall/Accuracy/F1 to the
class-weighting strategy specifically, rather than to some other
confounding difference between runs.
"""

import copy
import time

import torch


def train_model(model, dataloaders, criterion, optimizer, device,
                 num_epochs=15, save_path="best_model.pth", scheduler=None):
    """
    Train `model` and track train/val loss + accuracy each epoch, saving the
    weights that achieve the BEST validation accuracy (not just final-epoch
    weights -- final-epoch weights can be worse if the model started
    overfitting in later epochs).

    Args:
        model (nn.Module): model to train (already moved to `device`).
        dataloaders (dict): {'train': DataLoader, 'val': DataLoader}.
        criterion: loss function. This is the single argument that
            distinguishes the Baseline experiment from the Imbalance-Aware
            experiment (weighted vs. unweighted CrossEntropyLoss).
        optimizer (torch.optim.Optimizer): e.g. Adam.
        device (torch.device): CUDA / MPS / CPU.
        num_epochs (int): number of training epochs.
        save_path (str): where to persist the best-performing weights
            (state_dict only, not the full model object -- this is the
            recommended, portable PyTorch practice).
        scheduler (optional): learning-rate scheduler, `.step()` called
            after each epoch's validation phase.

    Returns:
        model (nn.Module): the model loaded with its BEST validation weights.
        history (dict): per-epoch train/val loss and accuracy, useful for
            plotting learning curves in your report.
    """
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_val_acc = 0.0

    history = {
        "train_loss": [], "train_acc": [],
        "val_loss": [], "val_acc": [],
    }

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print("-" * 30)

        # Each epoch has a training phase and a validation phase.
        for phase in ["train", "val"]:
            if phase == "train":
                model.train()  # enables dropout/batchnorm update behavior
            else:
                model.eval()   # disables dropout, freezes batchnorm running stats

            running_loss = 0.0
            running_corrects = 0
            total_samples = 0

            # torch.set_grad_enabled toggles autograd tracking. We only need
            # gradients during the training phase -- disabling them during
            # validation saves memory and compute.
            with torch.set_grad_enabled(phase == "train"):
                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    optimizer.zero_grad()

                    outputs = model(inputs)               # raw logits, shape [B, 2]
                    _, preds = torch.max(outputs, 1)       # predicted class index
                    loss = criterion(outputs, labels)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                    total_samples += inputs.size(0)

            epoch_loss = running_loss / total_samples
            epoch_acc = running_corrects.double() / total_samples

            history[f"{phase}_loss"].append(epoch_loss)
            history[f"{phase}_acc"].append(epoch_acc.item())

            print(f"{phase:5s} Loss: {epoch_loss:.4f}  Acc: {epoch_acc:.4f}")

            # Track and persist the best VALIDATION accuracy checkpoint.
            # Using val (not train) accuracy to select the "best" model is
            # what protects you from picking an overfit checkpoint.
            if phase == "val" and epoch_acc > best_val_acc:
                best_val_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(best_model_wts, save_path)
                print(f"  -> New best val accuracy ({best_val_acc:.4f}). Saved to {save_path}")

        if scheduler is not None:
            scheduler.step()

    time_elapsed = time.time() - since
    print(f"\nTraining complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
    print(f"Best val Accuracy: {best_val_acc:.4f}")

    # Load best weights back into the model before returning, so the caller
    # always receives the best checkpoint, not whatever the last epoch left
    # the in-memory weights at.
    model.load_state_dict(best_model_wts)
    return model, history
