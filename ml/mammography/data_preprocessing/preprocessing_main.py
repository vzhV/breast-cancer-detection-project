import json
import os
import shutil
from argparse import ArgumentParser, Namespace
from typing import Union

import data_preprocessing


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


def initial_preprocessing(keep_files: bool):
    """
        Main function for initial preprocessing of cbis-ddsm data
    """
    config = prepare_config()

    name_of_df = 'cbis_ddsm'

    # Preprocess .csv files to set up new paths to images
    processed_csv_path = data_preprocessing.preprocess_csv_files(config, name_of_df)
    config["cbis_ddsm_csv_proc"] = processed_csv_path

    # Remove artifacts and generate new .csv files
    artifacts_removed_csv_path, artifacts_removed_jpg_path = data_preprocessing.init_preprocessing_images(
        config, name_of_df
    )
    config["cbis_ddsm_csv_artifacts_removed"] = artifacts_removed_csv_path
    config["cbis_ddsm_jpeg_artifacts_removed"] = artifacts_removed_jpg_path

    # Crop images and generate new .csv files
    cropped_csv_path, cropped_jpg_path = data_preprocessing.crop_artifact_removed_images(config, name_of_df)
    config["cbis_ddsm_csv_artifacts_removed_cropped"] = cropped_csv_path
    config["cbis_ddsm_jpeg_artifacts_removed_cropped"] = cropped_jpg_path

    # Save the dictionary to JSON with indentation for readability
    with open(os.path.join(os.getcwd(), 'config_after_preprocessing.json'), "w") as json_file:
        json.dump(config, json_file, indent=2)

    if not keep_files:
        shutil.rmtree(config['temp_data_folder_path'])
    else:
        pass


def start_preprocessing():
    """
        Starts the preprocessing
    :return:
    """
    parser: ArgumentParser = ArgumentParser()
    parser.add_argument("-c", "--config_path", action="store", dest="config_path",
                        default=None, type=str)
    parser.add_argument("-f", "--keep_files", action="store", dest="keep_files",
                        default=True, type=bool,
                        help="Boolean on keeping downloaded unprocessed files")
    parser.add_argument("-a", "--action", action="store", dest="action",
                        default='all_scores_calc', type=str)
    args: Namespace = parser.parse_args()

    # python preprocessing_main.py -a "init_preprocessing"
    if args.action == "init_preprocessing":
        initial_preprocessing(True)


if "__main__" == __name__:
    start_preprocessing()
