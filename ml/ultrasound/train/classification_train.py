import json
import os
from collections import Counter

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from keras.callbacks import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import RandomHorizontalFlip, RandomRotation, ColorJitter

from ultrasound.models.classification import MedicalImageClassifier, train_model


def prepare_config():
    """
    Loads the configuration from a JSON file located in the same directory as this script.

    Returns:
        dict: A dictionary containing configuration settings.
    """
    dir_path = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(dir_path, 'classification_config.json')
    print('\nUsing as config file: ' + '\033[1m' + f'{config_path}' + '\033[0m')
    with open(config_path, 'r') as fp:
        config = json.load(fp)
    return config


def load_dataset(config):
    data_dir = config['train_val_test_dir']

    class_names = ['malignant', 'normal', 'benign']
    minority_classes = ['malignant', 'normal']

    minority_class_transforms = transforms.Compose([
        RandomHorizontalFlip(p=0.9),
        RandomRotation(15, expand=False, center=None),
        ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    ])
    normalize = transforms.Normalize(mean=[0.485], std=[0.229])

    data_transforms = {
        'train': transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.RandomApply([minority_class_transforms], p=0.5) if any(
                cls in minority_classes for cls in class_names) else transforms.RandomApply([], p=0.0),
            transforms.ToTensor(),
            normalize
        ]),
        'validation': transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize
        ]),
        'test': transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize
        ]),
    }

    image_datasets = {
        x: ImageFolder(
            root=os.path.join(data_dir, x),
            transform=data_transforms[x]
        )
        for x in ['train', 'validation', 'test']
    }

    return image_datasets


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    config = prepare_config()
    datasets = load_dataset(config)
    batch_size = config['batch_size']
    dataloaders = {x: DataLoader(datasets[x], batch_size=batch_size, shuffle=True, num_workers=4)
                   for x in ['train', 'validation', 'test']}
    class_names = datasets['train'].classes
    class_counts = Counter(datasets['train'].targets)
    total_samples = sum(class_counts.values())
    class_weights = [total_samples / class_counts[i] for i in range(len(class_names))]

    min_weight = min(class_weights)
    class_weights = [x / min_weight for x in class_weights]

    weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)

    loss_function = nn.CrossEntropyLoss(weight=weights_tensor)

    num_classes = len(class_names)
    model = MedicalImageClassifier(num_classes=num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

    trained = train_model(model, dataloaders, loss_function, optimizer, scheduler, num_epochs=config['num_epochs'], patience=config['early_stopping_patience'],
                          device=device)
    save_path = os.path.join(config['save_path'], config['model_name'])
    torch.save(trained, str(save_path))


if __name__ == '__main__':
    main()
