import copy
import time

import torch
import torch.nn as nn
import torch.nn.functional as F


class MedicalImageClassifier(nn.Module):
    """
    This class defines a convolutional neural network for medical image classification.
    It includes layers like convolutional layers, batch normalization, max pooling, and fully connected layers.
    """

    def __init__(self, num_classes):
        super(MedicalImageClassifier, self).__init__()
        # First convolutional layer
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)

        # Second convolutional layer
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)

        # Third convolutional layer
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)

        # Fourth convolutional layer
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.pool4 = nn.MaxPool2d(2, 2)

        # First fully connected layer
        self.fc1 = nn.Linear(128 * 14 * 14, 512)
        self.dropout1 = nn.Dropout(0.5)

        # Second fully connected layer
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        # Forward pass through the network, returning the output
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.pool4(F.relu(self.bn4(self.conv4(x))))
        x = x.view(x.size(0), -1)
        x = self.dropout1(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x


def train_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs=25, patience=5, device='cuda'):
    """
    Trains a given model with the possibility of early stopping to prevent overfitting.

    Args:
        model (nn.Module): The model to be trained.
        dataloaders (dict): Contains 'train' and 'validation' DataLoaders for loading data.
        criterion (callable): The loss function.
        optimizer (Optimizer): The optimizer.
        scheduler (lr_scheduler): Learning rate scheduler.
        num_epochs (int): The maximum number of epochs.
        patience (int): The patience for early stopping.
        device (str): The device type.

    Returns:
        nn.Module: The trained model.
    """
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        for phase in ['train', 'validation']:
            model.train() if phase == 'train' else model.eval()
            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'validation':
                scheduler.step(epoch_loss)
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    best_model_wts = copy.deepcopy(model.state_dict())
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1

        if epochs_no_improve == patience:
            print('Early stopping initiated')
            break

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Loss: {best_loss:.4f}')

    model.load_state_dict(best_model_wts)
    return model
