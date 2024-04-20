import json
import os

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, CSVLogger, EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from mammography.models import classification

np.random.seed(42)
tf.random.set_seed(42)


def create_dir(path):
    """
    Ensures a directory exists, and if not, creates it.

    Args:
        path (str): Path to the directory to be created.
    """
    if not os.path.exists(path):
        os.makedirs(path)


def prepare_config():
    """
    Loads the configuration from a JSON file located in the same directory as this script.

    Returns:
        dict: Configuration settings as a dictionary.
    """
    dir_path = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(dir_path, 'classification_config.json')
    print('\nUsing as config file: ' + '\033[1m' + f'{config_path}' + '\033[0m')
    with open(config_path, 'r') as fp:
        config = json.load(fp)
    return config


def image_processor(image_path, target_size):
    """
    Processes images by loading, resizing, applying histogram equalization, and normalizing.

    Args:
        image_path (str): Path to the image.
        target_size (tuple): Desired size of the output image.

    Returns:
        ndarray: The processed image as a numpy array.
    """
    # Load image with full path resolution
    absolute_image_path = os.path.abspath(image_path)
    image = cv2.imread(absolute_image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Resize image
    image = cv2.resize(image, (target_size[1], target_size[0]))

    # Apply histogram equalization
    image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(image)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l_channel = clahe.apply(l_channel)
    image = cv2.merge((l_channel, a_channel, b_channel))
    image = cv2.cvtColor(image, cv2.COLOR_LAB2RGB)

    # Apply negative transformation
    image = 255 - image

    # Normalize the image to range [0, 1]
    image = image.astype(np.float32) / 255.0

    return image


def get_train_val_test_split(config):
    """
    Loads and splits the data into training, validation, and test sets.

    Args:
        config (dict): Configuration data containing paths and parameters.

    Returns:
        tuple: Arrays containing train, validation, and test sets.
    """
    # Load dataset
    train = pd.read_csv(config['mass_train'])
    test = pd.read_csv(config['mass_test'])
    full_mass = pd.concat([train, test], axis=0)
    target_size = (config['size'], config['size'], 3)

    # Process images and map labels
    full_mass['processed_images'] = full_mass['image file path'].apply(lambda x: image_processor(x, target_size))
    class_mapper = {'MALIGNANT': 1, 'BENIGN': 0, 'BENIGN_WITHOUT_CALLBACK': 0}
    X_resized = np.array(full_mass['processed_images'].tolist())
    full_mass['labels'] = full_mass['pathology'].replace(class_mapper).infer_objects(copy=False)

    # Split dataset
    train_x, valid_x, train_y, valid_y = train_test_split(X_resized, full_mass['labels'].values, test_size=0.2,
                                                          random_state=42)
    valid_x, test_x, valid_y, test_y = train_test_split(valid_x, valid_y, test_size=0.5, random_state=42)
    return train_x, train_y, valid_x, valid_y, test_x, test_y


def main():
    """
    Main function to execute the training of the model.
    """
    config = prepare_config()
    train_x, train_y, valid_x, valid_y, test_x, test_y = get_train_val_test_split(config)

    create_dir(config['files_dir'])
    model_path = os.path.join(config['files_dir'], config['model_name'])
    csv_path = os.path.join(config['files_dir'], 'log.csv')

    data_gen_args = dict(
        rotation_range=40,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        vertical_flip=False,
        fill_mode='nearest'
    )

    train_datagen = ImageDataGenerator(**data_gen_args)
    valid_datagen = ImageDataGenerator()

    train_generator = train_datagen.flow(train_x, train_y, batch_size=config['batch_size'])
    valid_generator = valid_datagen.flow(valid_x, valid_y, batch_size=config['batch_size'])

    callbacks = [
        ModelCheckpoint(model_path, save_best_only=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=1e-7, verbose=1),
        CSVLogger(csv_path),
        EarlyStopping(monitor='val_loss', patience=config['early_stopping_patience'], verbose=1)
    ]

    input_shape = (config['size'], config['size'], 3)
    model = classification.build_model(input_shape=input_shape, num_classes=1)
    model.compile(optimizer=config['optimizer'], loss='binary_crossentropy', metrics=['accuracy'])

    model.fit(
        train_generator,
        steps_per_epoch=len(train_x) // config['batch_size'],
        epochs=config['num_epochs'],
        validation_data=valid_generator,
        validation_steps=len(valid_x) // config['batch_size'],
        callbacks=callbacks
    )




if __name__ == '__main__':
    main()
