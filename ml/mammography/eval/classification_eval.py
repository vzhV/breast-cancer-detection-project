import os

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report
from tensorflow.keras.models import load_model

from mammography.train.classification_train import get_train_val_test_split, prepare_config


def visualize_classification_report(model, test_x, test_y):
    """
    Generates and visualizes the classification report as a heatmap.

    Args:
        model (tf.keras.Model): Trained model.
        test_x (np.array): Test dataset features.
        test_y (np.array): Actual labels of the test dataset.
    """

    test_predictions = model.predict(test_x).round()

    report_dict = classification_report(test_y, test_predictions, target_names=['BENIGN', 'MALIGNANT'],
                                        output_dict=True)
    print("\nClassification Report:\n", report_dict)
    report_df = pd.DataFrame(report_dict).transpose()
    report_df = report_df.drop(columns=['support'])

    plt.figure(figsize=(10, 6))
    sns.heatmap(report_df, annot=True, cmap="Blues", fmt=".2f")
    plt.title("Classification Report")
    plt.show()


def visualize_predictions(model, X_test, y_test, num_images=5):
    """
    Visualizes predictions for a subset of test images.

    Args:
        model (tf.keras.Model): Trained model.
        X_test (np.array): Test dataset features.
        y_test (np.array): Actual labels of the test dataset.
        num_images (int): Number of images to display.
    """
    class_labels = {0: 'BENIGN', 1: 'MALIGNANT'}
    # Randomly select indices to visualize
    random_indices = np.random.choice(X_test.shape[0], size=num_images, replace=False)

    predictions = model.predict(X_test[random_indices])
    predicted_classes = (predictions > 0.5).astype(int).flatten()

    plt.figure(figsize=(20, 5))
    for i, index in enumerate(random_indices):
        plt.subplot(1, num_images, i + 1)
        plt.imshow(X_test[index])
        plt.title(f'Actual: {class_labels[y_test[index]]}\nPredicted: {class_labels[predicted_classes[i]]}')
        plt.axis('off')
    plt.show()


def main():
    """
    Main function to execute the testing and visualization procedures.
    """
    # Load configuration and prepare data
    config = prepare_config()
    _, _, _, _, test_x, test_y = get_train_val_test_split(config)
    # Load the model from saved path
    model_path = os.path.join(config['files_dir'], config['model_name'])
    model = load_model(model_path, compile=False)
    # Execute visualization functions
    visualize_classification_report(model, test_x, test_y)
    visualize_predictions(model, test_x, test_y)


if __name__ == '__main__':
    main()
