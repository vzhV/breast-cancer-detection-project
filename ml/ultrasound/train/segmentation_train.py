import json
import os
from glob import glob

import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, CSVLogger, EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from mammography.metrics import metrics
from mammography.models import segmentation

np.random.seed(42)
tf.random.set_seed(42)


def create_dir(path):
    """
    Creates a directory if it does not already exist.

    Args:
        path (str): Path to the directory to be created.
    """
    if not os.path.exists(path):
        os.makedirs(path)


def prepare_config():
    """
    Loads the configuration from a JSON file located in the same directory as this script.

    Returns:
        dict: A dictionary containing configuration settings.
    """
    dir_path = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(dir_path, 'segmentation_config.json')
    print('\nUsing as config file: ' + '\033[1m' + f'{config_path}' + '\033[0m')
    with open(config_path, 'r') as fp:
        config = json.load(fp)
    return config


def load_image(path, size):
    """
    Loads an image from a specified path and preprocesses it.

    Args:
        path (str): Path to the image file.
        size (int): Desired image size (square images).

    Returns:
        ndarray: Preprocessed image as a numpy array.
    """

    # Load image using OpenCV.
    image = cv2.imread(path)
    # Resize image to specified size.
    image = cv2.resize(image, (size, size))
    # Convert image to grayscale.
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # Normalize pixel values to be between 0 and 1.
    image = image / 255.
    return image


def load_data(root_path, size):
    """
    Load and preprocess images (ultrasound images + masks) from their directory

    Args:
        root_path (str): Path to the folder containing the images and masks
        size (int): Desired image size (square images).

    Returns:
        tuple: Two numpy arrays containing loaded and preprocessed images and masks.
    """
    images = []
    masks = []

    x = 0

    for path in sorted(glob(root_path)):
        img = load_image(path, size)

        if 'mask' in path:
            if x:
                masks[-1] += img
                masks[-1] = np.array(masks[-1] > 0.5, dtype='float64')
            else:
                masks.append(img)
                x = 1
        else:
            images.append(img)
            x = 0
    return np.array(images), np.array(masks)


def get_train_val_test_split(config):
    """
    Splits the dataset into training, validation, and test sets.

    Args:
        config (dict): Configuration dictionary containing dataset details.

    Returns:
        tuple: Training, validation, and test sets (images and masks).
    """
    root_path = config['dataset_path'] + '\\*\\*'
    X, y = load_data(root_path=root_path, size=config['size'])

    # Add channel dimensions.
    X = np.expand_dims(X, -1)
    y = np.expand_dims(y, -1)

    train_x, valid_x, train_y, valid_y = train_test_split(X, y, test_size=0.2, random_state=42)
    valid_x, test_x, valid_y, test_y = train_test_split(valid_x, valid_y, test_size=0.5, random_state=42)
    return train_x, train_y, valid_x, valid_y, test_x, test_y


def main():
    """
    Main function to execute the training process.

    Sets up the environment, prepares data, builds the model, and executes the training process.
    """

    # Load configuration settings.
    config = prepare_config()
    # Prepare data splits.
    train_x, train_y, valid_x, valid_y, test_x, test_y = get_train_val_test_split(config)
    # Create directory for storing files.
    create_dir(config['files_dir'])
    # Path to save the model.
    model_path = os.path.join(config['files_dir'], config['model_name'])
    # Path to save logs.
    csv_path = os.path.join(config['files_dir'], 'log.csv')
    # Input image dimensions.
    input_img = (config['size'], config['size'], 1)

    # Build and compile model.
    model = segmentation.build_unet(input_img)
    model.compile(loss=metrics.dice_loss, optimizer=config['optimizer'], metrics=[metrics.dice_coef])

    callbacks = [
        ModelCheckpoint(model_path, verbose=1, save_best_only=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=1e-7, verbose=1),
        CSVLogger(csv_path),
        EarlyStopping(monitor='val_loss', patience=config['early_stopping_patience'], restore_best_weights=False),
    ]

    data_gen_args = dict(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='nearest'
    )
    # Data augmentation
    image_datagen = ImageDataGenerator(**data_gen_args)
    mask_datagen = ImageDataGenerator(**data_gen_args)

    image_generator = image_datagen.flow(train_x, batch_size=config['batch_size'],
                                         seed=42)
    mask_generator = mask_datagen.flow(train_y, batch_size=config['batch_size'],
                                       seed=42)
    train_generator = zip(image_generator, mask_generator)

    train_samples = train_x.shape[0]
    steps_per_epoch = train_samples // config['batch_size']

    model.fit(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        epochs=config['num_epochs'],
        validation_data=(train_x, train_y),
        callbacks=callbacks
    )


if __name__ == '__main__':
    main()
