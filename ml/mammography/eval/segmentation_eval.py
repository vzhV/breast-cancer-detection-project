import os

import numpy as np
from keras.metrics import Recall, Precision, MeanIoU
from matplotlib import pyplot as plt
from tensorflow.keras.models import load_model

from mammography.train.segmentation_train import get_train_val_test_split, prepare_config


def visualize_segmentation(model, test_data_x, test_data_y):
    """
    Visualizes a random sample of images, their true masks, and predicted masks from the test dataset.

    Args:
        model (tf.keras.Model): Loaded model.
        test_data_x (np.array): Test dataset images.
        test_data_y (np.array): Masks for the test images.
    """
    fig, ax = plt.subplots(5, 3, figsize=(10, 18))

    # Randomly select 5 indices from test data.
    j = np.random.randint(0, test_data_x.shape[0], 5)
    for i in range(5):
        # Display the test image.
        ax[i, 0].imshow(test_data_x[j[i]], cmap='gray')
        ax[i, 0].set_title('Image')

        # Display the true mask.
        ax[i, 1].imshow(test_data_y[j[i]], cmap='gray')
        ax[i, 1].set_title('Mask')

        # Display the predicted mask.
        ax[i, 2].imshow(model.predict(np.expand_dims(test_data_x[j[i]], 0), verbose=0)[0],
                        cmap='gray')
        ax[i, 2].set_title('Prediction')
    fig.suptitle('Results', fontsize=16)
    plt.show()


def print_metrics_evaluation(model, test_data_x, test_data_y):
    """
    Evaluates the model using the test dataset and prints the precision, recall, and F1-score.

    Args:
        model (tf.keras.Model): The trained model to use for prediction.
        test_data_x (np.array): The test dataset images.
        test_data_y (np.array): The actual segmentation masks for the test images.
    """
    print(f'\033[93m')
    y_pred = model.predict(test_data_x, verbose=0)
    # Apply a threshold to convert probabilities to binary mask.
    y_pred_thresholded = y_pred > 0.5

    IOU_keras = MeanIoU(num_classes=2)
    IOU_keras.update_state(y_pred_thresholded, test_data_y)
    print("Mean IoU =", IOU_keras.result().numpy())

    prec_score = Precision()
    prec_score.update_state(y_pred_thresholded, test_data_y)
    p = prec_score.result().numpy()
    print('Precision Score = %.3f' % p)

    recall_score = Recall()
    recall_score.update_state(y_pred_thresholded, test_data_y)
    r = recall_score.result().numpy()
    print('Recall Score = %.3f' % r)

    f1_score = 2 * (p * r) / (p + r) if (p + r) != 0 else 0
    print('F1 Score = %.3f' % f1_score)


def main():
    """
    Main function to set up the environment, load the test data, load the model, and perform evaluations and visualizations.
    """
    config = prepare_config()
    _, _, _, _, test_x, test_y = get_train_val_test_split(config)
    model_path = os.path.join(config['files_dir'], config['model_name'])
    model = load_model(model_path, compile=False)
    visualize_segmentation(model, test_x, test_y)
    print_metrics_evaluation(model, test_x, test_y)


if __name__ == '__main__':
    main()
