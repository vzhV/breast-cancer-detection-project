import json
import os
import shutil
import warnings

import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split


def overlay_and_save(image_path, mask_path):
    """
    Overlays an image with its corresponding mask and saves the result in an output directory.

    Args:
        image_path (str): Path to the original image.
        mask_path (str): Path to the mask image.
    """
    try:
        # Check if both image and mask exist
        if os.path.exists(image_path) and os.path.exists(mask_path):
            image = Image.open(image_path)
            mask = Image.open(mask_path)

            # Ensure mask is in the same mode as the image
            if image.mode != mask.mode:
                mask = mask.convert(image.mode)

            # Resize image to match the mask size if they differ
            if image.size != mask.size:
                image = image.resize(mask.size)

            # Blend the image and the mask
            overlayed = Image.blend(image, mask, alpha=0.5)

            # Construct output path using the label and save the overlayed image
            label = os.path.basename(os.path.dirname(image_path))
            output_path = os.path.join(output_dir, label, os.path.basename(image_path))
            overlayed.save(output_path)
        else:
            # Pass if the paths do not exist
            pass
    except Exception as e:
        print(f"An error occurred for: {image_path} or {mask_path}. Error: {str(e)}")


def prepare_config():
    """
    Loads the configuration from a JSON file located in the same directory as this script.

    Returns:
        dict: A dictionary containing configuration settings.
    """
    dir_path = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(dir_path, 'config.json')
    print('\nUsing as config file: ' + '\033[1m' + f'{config_path}' + '\033[0m')
    with open(config_path, 'r') as fp:
        config = json.load(fp)
    return config


if __name__ == '__main__':
    # Ignore certain warnings from PIL
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=ResourceWarning)

    config = prepare_config()
    input_dir = config['data_dir']
    output_dir = config['save_dir']
    labels = ['benign', 'malignant', 'normal']

    if config['use_overlay']:
        # Ensure output directories exist for each label
        for label in labels:
            os.makedirs(os.path.join(output_dir, label), exist_ok=True)

        # Process each label directory
        for label in labels:
            label_dir = os.path.join(input_dir, label)
            if os.path.isdir(label_dir):
                for image_filename in os.listdir(label_dir):
                    if image_filename.endswith('.png'):
                        image_path = os.path.join(label_dir, image_filename)
                        mask_filename = image_filename.replace('.png', '_mask.png')
                        mask_path = os.path.join(label_dir, mask_filename)
                        overlay_and_save(image_path, mask_path)

        print(f"Overlayed images have been saved to {output_dir}")

    data_dir = config['save_dir'] if config['use_overlay'] else config['data_dir']

    file_paths = []
    labels = []

    # Loop through the subdirectories (benign, malignant, normal)
    for label in os.listdir(data_dir):
        label_dir = os.path.join(data_dir, label)
        if os.path.isdir(label_dir):
            for image_file in os.listdir(label_dir):
                if image_file.endswith('.png') and not (image_file.endswith('_mask.png') or
                                                        image_file.endswith('_mask_1.png') or
                                                        image_file.endswith('_mask_2.png')):
                    image_path = os.path.join(label_dir, image_file)
                    labels.append(label)
                    file_paths.append(image_path)

    # Create a DataFrame to store the file paths and labels
    data = pd.DataFrame({'img': file_paths, 'label': labels})

    # Split the dataset into train, validation, and test sets
    train_data, test_data = train_test_split(data, test_size=0.15, random_state=42, stratify=data['Label'])
    train_data, val_data = train_test_split(train_data, test_size=0.15, random_state=42, stratify=train_data['Label'])

    # Define the paths for the train, validation, and test directories
    train_dir = os.path.join(config['train_val_test_dir'], 'train')
    val_dir = os.path.join(config['train_val_test_dir'], 'validation')
    test_dir = os.path.join(config['train_val_test_dir'], 'test')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # Create the train, validation, and test directories and subdirectories
    for label in labels:
        os.makedirs(os.path.join(train_dir, label), exist_ok=True)
        os.makedirs(os.path.join(val_dir, label), exist_ok=True)
        os.makedirs(os.path.join(test_dir, label), exist_ok=True)

    # Copy the images to the corresponding directories
    for _, row in train_data.iterrows():
        image_path = row['img']
        label = row['label']
        shutil.copy(image_path, os.path.join(train_dir, label))

    for _, row in val_data.iterrows():
        image_path = row['img']
        label = row['label']
        shutil.copy(image_path, os.path.join(val_dir, label))

    for _, row in test_data.iterrows():
        image_path = row['img']
        label = row['label']
        shutil.copy(image_path, os.path.join(test_dir, label))

    print('Completed')
