import os

import pandas as pd
import seaborn as sns
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader

from ultrasound.models.classification import MedicalImageClassifier
from ultrasound.train.classification_train import load_dataset, prepare_config


def visualize_confusion_matrix_and_report(config, model, dataset, device):
    """
    Visualizes the confusion matrix and classification report for the model's performance on the test set.

    Args:
        config (dict): Configuration settings.
        model (torch.nn.Module): Trained model.
        dataset (dict): Datasets indexed by 'train', 'validation', and 'test'.
        device (torch.device): Device to perform computations on.
    """
    # Prepare dataloaders for testing
    batch_size = config['batch_size']
    dataloaders = {x: DataLoader(dataset[x], batch_size=batch_size, shuffle=True, num_workers=4)
                   for x in ['test']}
    class_names = dataset['train'].classes
    label_names = [str(class_names[i]) for i in range(len(class_names))]

    y_true = []
    y_pred = []

    model.eval()

    with torch.no_grad():
        for inputs, labels in dataloaders['test']:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    # Generate classification report and confusion matrix
    classification_rep = classification_report(y_true, y_pred, target_names=label_names, output_dict=True)
    confusion_mat = confusion_matrix(y_true, y_pred)

    # Visualize confusion matrix
    plt.figure(figsize=(5, 3))
    sns.heatmap(confusion_mat, annot=True, fmt='d', cmap='Blues', cbar=False, xticklabels=label_names,
                yticklabels=label_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

    # Visualize classification report
    plt.figure(figsize=(6, 4))
    sns.heatmap(pd.DataFrame(classification_rep).iloc[:-1, :].T, annot=True, cmap='Blues', fmt='.2f')
    plt.title('Classification Report Heatmap')
    plt.show()


def visualize_predictions(model, dataset, device):
    """
    Visualizes a batch of test images with their actual and predicted labels.

    Args:
        model (torch.nn.Module): Trained model.
        dataset (dict): Dataset containing test images.
        device (torch.device): Device to perform computations on.
    """
    num_images_to_display = 15

    # Load a batch of test images
    test_dataloader = DataLoader(dataset['test'], batch_size=num_images_to_display, shuffle=True, num_workers=4)
    inputs, labels = next(iter(test_dataloader))
    inputs = inputs.to(device)

    # Convert to grayscale by averaging channels
    grayscale_images = inputs.cpu().numpy().mean(axis=1)
    class_names = dataset['train'].classes

    with torch.no_grad():
        model.eval()
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)

    # Display images with actual and predicted labels
    plt.figure(figsize=(15, 20))
    for i in range(num_images_to_display):
        ax = plt.subplot(5, 3, i + 1)
        ax.axis('off')
        ax.set_title(f'Actual: {class_names[labels[i]]}\nPredicted: {class_names[preds[i]]}')
        plt.imshow(grayscale_images[i], cmap='gray')
    plt.show()


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    config = prepare_config()
    dataset = load_dataset(config)
    class_names = dataset['train'].classes
    model = MedicalImageClassifier(num_classes=len(class_names)).to(device)
    model.load_state_dict(torch.load(os.path.join(config['save_dir'], config['model_name'])))
    visualize_confusion_matrix_and_report(config, model, dataset, device)
    visualize_predictions(model, dataset, device)


if __name__ == '__main__':
    main()
