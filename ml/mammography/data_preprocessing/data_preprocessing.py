import glob
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, Tuple

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

import process_img

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..'))


def replace_path_to_main_image(_path: str, _data_path_jpeg: str) -> str:
    """
        Function to replace path to the main image
    """
    updated_path: str = glob.glob(os.path.join(_data_path_jpeg, f'{_path.split("/")[-2]}\\*'))[0]
    return updated_path


def get_shape_of_image(_path: str) -> tuple:
    """
        Function which returns image dimensions
    """
    _img = cv2.imread(_path)
    img_shape = _img.shape[:2]

    return img_shape


def replace_path_cropped(_data, _data_path_jpeg: str) -> str:
    """
        Function to replace path to the cropped image
    """
    _main_img_path = _data['image file path']
    _init_crop_path = _data['cropped image file path']
    _main_img_shape = get_shape_of_image(_main_img_path)

    _cropped_img_paths = glob.glob(os.path.join(_data_path_jpeg, f'{_init_crop_path.split("/")[-2]}\\*'))
    _new_cropped_img_path = ''
    for _crop_image_path in _cropped_img_paths:
        _crop_image_shape = get_shape_of_image(_crop_image_path)
        if _crop_image_shape != _main_img_shape:
            _new_cropped_img_path = _crop_image_path
            break

    return _new_cropped_img_path


def replace_path_roi_mask(_data, _data_path_jpeg: str) -> str:
    """
        Function to replace path to the ground truth image
    """
    _crop_path = _data['cropped image file path']
    _init_roi_mask_path = _data['ROI mask file path']

    _roi_mask_paths = glob.glob(os.path.join(_data_path_jpeg, f'{_init_roi_mask_path.split("/")[-2]}\\*'))
    _new_roi_mask_path = ''
    for _roi_mask_path in _roi_mask_paths:
        if _roi_mask_path != _crop_path:
            _new_roi_mask_path = _roi_mask_path
            break

    return _new_roi_mask_path


def preprocess_csv_files(config, dataset_name) -> str:
    if type(config) != dict:
        raise 'Problem with initialization of config'
    print('\n' + 'Processing ' + '\033[1m' + f'{dataset_name} .csv ' + '\033[0m' + 'files...')

    processed_csv_path = os.path.join(os.path.dirname(config["cbis_ddsm_csv_dp"]), 'processed_csv')
    Path(processed_csv_path).mkdir(parents=True, exist_ok=True)

    csv_files_to_proc = [el for el in glob.glob(os.path.join(config["cbis_ddsm_csv_dp"], '*.csv')) if
                         "_train_set" in el or "_test_set" in el]

    for _csv_path in csv_files_to_proc:
        print(' Processing ' + '\033[1m' + f'{os.path.basename(_csv_path)} ' + '\033[0m' + 'file')

        if os.path.isfile(os.path.join(processed_csv_path, f'{os.path.basename(_csv_path)}')):
            print(f'     Processed {os.path.basename(_csv_path)} file already saved into: ' + '\033[1m' +
                  os.path.join(processed_csv_path, f'{os.path.basename(_csv_path)}') + '\033[0m' + 'file')
        else:
            start_time = time.time()
            df = pd.read_csv(_csv_path)

            df['image file path'] = df['image file path'].apply(
                replace_path_to_main_image,
                _data_path_jpeg=config["cbis_ddsm_jpeg_dp"]
            )

            df['cropped image file path'] = df[['image file path', 'cropped image file path']].apply(
                replace_path_cropped,
                _data_path_jpeg=config["cbis_ddsm_jpeg_dp"],
                axis=1
            )

            df['ROI mask file path'] = df[['cropped image file path', 'ROI mask file path']].apply(
                replace_path_roi_mask,
                _data_path_jpeg=config["cbis_ddsm_jpeg_dp"],
                axis=1
            )

            df.to_csv(
                os.path.join(processed_csv_path, f'{os.path.basename(_csv_path)}'),
                index=False
            )
            print('\033[1m' + f'        {os.path.basename(_csv_path)} ' + '\033[0m' + 'file processed in ' +
                  f'{time.time() - start_time} seconds')
            print(f'     Processed {os.path.basename(_csv_path)} file saved into: ' + '\033[1m' +
                  os.path.join(processed_csv_path, f'{os.path.basename(_csv_path)}') + '\033[0m' + 'file')

    print('\n' + 'All ' + '\033[1m' + f'{dataset_name} .csv ' + '\033[0m' + 'files were processed sucessfully')

    return processed_csv_path


def preprocess_three_images_in_row(_row: pd.Series, _config: Dict, _image_processor) -> tuple:
    """
        Function to process rows of dataframe with images
    """
    _img_path = _row['image file path']
    _cropped_area_path = _row['cropped image file path']
    _roi_mask_path = _row['ROI mask file path']

    input_path = _config["cbis_ddsm_jpeg_dp"]
    output_path = _config["cbis_ddsm_jpeg_artifacts_removed"]

    _new_img_path = _img_path.replace(
        os.path.basename(input_path), os.path.basename(output_path)
    )
    Path(os.path.dirname(_new_img_path)).mkdir(parents=True, exist_ok=True)

    if type(_cropped_area_path) == float:
        _cropped_area_path = None
        _new_cropped_area_path = np.NaN
    else:
        _new_cropped_area_path = _cropped_area_path.replace(
            os.path.basename(input_path), os.path.basename(output_path)
        )
        Path(os.path.dirname(_new_cropped_area_path)).mkdir(parents=True, exist_ok=True)

    _new_roi_mask_path = _roi_mask_path.replace(
        os.path.basename(input_path), os.path.basename(output_path)
    )
    Path(os.path.dirname(_new_roi_mask_path)).mkdir(parents=True, exist_ok=True)

    _image_processor.remove_artifacts_from_img(
        _img_path=_img_path,
        _cropped_area_path=_cropped_area_path,
        _roi_mask_path=_roi_mask_path,
        _input_path=input_path,
        _output_path=output_path
    )

    return _new_img_path, _new_cropped_area_path, _new_roi_mask_path


def init_preprocessing_images(config, dataset_name) -> Tuple[str, str]:
    """
        Function for initial preprocessing of images
    """
    if type(config) != dict:
        raise 'Problem with initialization of config'
    print('\n' + 'Initial processing of images annotated in ' + '\033[1m' +
          f'{dataset_name} .csv ' + '\033[0m' + 'files...')

    csv_files_to_proc = [el for el in glob.glob(os.path.join(config["cbis_ddsm_csv_proc"], '*.csv')) if
                         "_train_set" in el or "_test_set" in el]

    config["cbis_ddsm_csv_artifacts_removed"] = os.path.join(
        os.path.dirname(config["cbis_ddsm_csv_dp"]), 'processed_csv_artifacts_removed'
    )
    Path(config["cbis_ddsm_csv_artifacts_removed"]).mkdir(parents=True, exist_ok=True)

    config["cbis_ddsm_jpeg_artifacts_removed"] = os.path.join(
        os.path.dirname(config["cbis_ddsm_jpeg_dp"]), 'jpeg_artifacts_removed'
    )
    if os.path.isdir(config["cbis_ddsm_jpeg_artifacts_removed"]):
        shutil.rmtree(config['cbis_ddsm_jpeg_artifacts_removed'])
    Path(config["cbis_ddsm_jpeg_artifacts_removed"]).mkdir(parents=True, exist_ok=True)

    image_processor = process_img.MammogramInitImageProcessor()

    for _csv_path in csv_files_to_proc:
        print(' Processing images annotated in ' + '\033[1m' + f'{os.path.basename(_csv_path)} ' + '\033[0m' + 'file')

        if os.path.isfile(
                os.path.join(config["cbis_ddsm_csv_artifacts_removed"], f'{os.path.basename(_csv_path)}')):
            print(f'     Processed {os.path.basename(_csv_path)} .csv file already saved into: ' + '\033[1m' +
                  os.path.join(config["cbis_ddsm_csv_artifacts_removed"], f'{os.path.basename(_csv_path)}') +
                  '\033[0m' + ' file')
            print(f'     Images annotated in {os.path.basename(_csv_path)} .csv file already saved into: ' +
                  '\033[1m' + os.path.join(config["cbis_ddsm_jpeg_artifacts_removed"], '') + '\033[0m' + ' file')
        else:
            start_time = time.time()
            df = pd.read_csv(_csv_path)
            img_cols_to_process = ['image file path', 'cropped image file path', 'ROI mask file path']

            df = df.dropna(subset=['image file path', 'ROI mask file path'])

            tqdm.pandas()
            df[img_cols_to_process] = df.progress_apply(
                preprocess_three_images_in_row,
                _config=config, _image_processor=image_processor,
                axis=1, result_type='expand'
            )

            df.to_csv(
                os.path.join(config["cbis_ddsm_csv_artifacts_removed"], f'{os.path.basename(_csv_path)}'),
                index=False
            )
            print('\033[1m' + '     Images annotated in ' + f'{os.path.basename(_csv_path)} ' + '\033[0m' +
                  'file processed in ' + f'{time.time() - start_time} seconds')

            print(f'     Processed {os.path.basename(_csv_path)} .csv file saved into: ' + '\033[1m' +
                  os.path.join(config["cbis_ddsm_csv_artifacts_removed"], f'{os.path.basename(_csv_path)}') +
                  '\033[0m' + ' file')

            print(f'     Images annotated in {os.path.basename(_csv_path)} .csv file saved into: ' + '\033[1m' +
                  os.path.join(config["cbis_ddsm_jpeg_artifacts_removed"], '') + '\033[0m' + ' file')

    print('\n' + 'All ' + '\033[1m' + f'{dataset_name} .csv ' + '\033[0m' + 'files were processed sucessfully')

    return config["cbis_ddsm_csv_artifacts_removed"], config["cbis_ddsm_jpeg_artifacts_removed"]


def preprocess_two_images_in_row_crop(_row: pd.Series, _config: Dict, _image_processor) -> tuple:
    """
        Function to process rows of dataframe with images
    """
    _img_path = _row['image file path']
    _roi_mask_path = _row['ROI mask file path']

    input_path = _config["cbis_ddsm_jpeg_artifacts_removed"]
    output_path = _config["cbis_ddsm_jpeg_artifacts_removed_cropped"]

    _new_img_path = _img_path.replace(os.path.basename(input_path), os.path.basename(output_path))
    Path(os.path.dirname(_new_img_path)).mkdir(parents=True, exist_ok=True)

    _new_roi_mask_path = _roi_mask_path.replace(os.path.basename(input_path), os.path.basename(output_path))
    Path(os.path.dirname(_new_roi_mask_path)).mkdir(parents=True, exist_ok=True)

    crop_img_shapes = _image_processor.crop_image_by_mask(
        _img_path=_img_path,
        _roi_mask_path=_roi_mask_path,
        _input_path=input_path,
        _output_path=output_path
    )

    return _new_img_path, _new_roi_mask_path, crop_img_shapes


def get_biggest_dimensions(_series: pd.Series, _biggest_img_dimensions: tuple) -> tuple:
    """
        Function which returns the biggest dimensions of the series
    """

    for dimensions in _series:
        _biggest_img_dimensions = tuple(max(a, b) for a, b in zip(_biggest_img_dimensions, dimensions))

    return _biggest_img_dimensions


def crop_artifact_removed_images(config, dataset_name) -> Tuple[str, str]:
    """
        Function for cropping the images after removing the artifacts
    """

    if type(config) != dict:
        raise 'Problem with initialization of config'
    print('\n' + 'Cropping of initially preprocessed images annotated in ' + '\033[1m' +
          f'{dataset_name} .csv ' + '\033[0m' + 'files...')

    csv_files_to_proc = [el for el in glob.glob(os.path.join(config["cbis_ddsm_csv_artifacts_removed"], '*.csv')) if
                         "_train_set" in el or "_test_set" in el]

    config["cbis_ddsm_csv_artifacts_removed_cropped"] = os.path.join(
        os.path.dirname(config["cbis_ddsm_csv_dp"]), 'processed_csv_artifacts_removed_cropped'
    )
    Path(config["cbis_ddsm_csv_artifacts_removed_cropped"]).mkdir(parents=True, exist_ok=True)

    config["cbis_ddsm_jpeg_artifacts_removed_cropped"] = os.path.join(
        os.path.dirname(config["cbis_ddsm_jpeg_dp"]), 'jpeg_artifacts_removed_cropped'
    )
    if os.path.isdir(config["cbis_ddsm_jpeg_artifacts_removed_cropped"]):
        shutil.rmtree(config['cbis_ddsm_jpeg_artifacts_removed_cropped'])
    Path(config["cbis_ddsm_jpeg_artifacts_removed_cropped"]).mkdir(parents=True, exist_ok=True)

    biggest_img_dimensions = (-np.inf, -np.inf)
    image_processor = process_img.ImageShapeProcessing()

    for _csv_path in csv_files_to_proc:
        print(' Processing images annotated in ' + '\033[1m' + f'{os.path.basename(_csv_path)} ' + '\033[0m' + 'file')

        if os.path.isfile(
                os.path.join(config["cbis_ddsm_csv_artifacts_removed_cropped"], f'{os.path.basename(_csv_path)}')):
            print(f'     Processed {os.path.basename(_csv_path)} .csv file already saved into: ' + '\033[1m' +
                  os.path.join(config["cbis_ddsm_csv_artifacts_removed_cropped"], f'{os.path.basename(_csv_path)}') +
                  '\033[0m' + ' file')
            print(f'     Images annotated in {os.path.basename(_csv_path)} .csv file already saved into: ' +
                  '\033[1m' + os.path.join(config["cbis_ddsm_jpeg_artifacts_removed_cropped"], '') +
                  '\033[0m' + ' file')
        else:
            start_time = time.time()
            df = pd.read_csv(_csv_path)
            img_cols_to_process = ['image file path', 'ROI mask file path']

            df = df.dropna(subset=img_cols_to_process)

            tqdm.pandas()
            df[['image file path', 'ROI mask file path', 'biggest_img_dimensions']] = df.progress_apply(
                preprocess_two_images_in_row_crop,
                _config=config, _image_processor=image_processor,
                axis=1, result_type='expand'
            )

            biggest_img_dimensions = get_biggest_dimensions(df['biggest_img_dimensions'], biggest_img_dimensions)

            df = df.drop('biggest_img_dimensions', axis=1)

            df.to_csv(
                os.path.join(config["cbis_ddsm_csv_artifacts_removed_cropped"], f'{os.path.basename(_csv_path)}'),
                index=False
            )
            print('\033[1m' + '     Images annotated in ' + f'{os.path.basename(_csv_path)} ' + '\033[0m' +
                  'file processed in ' + f'{time.time() - start_time} seconds')

            print(f'     Processed {os.path.basename(_csv_path)} .csv file saved into: ' + '\033[1m' +
                  os.path.join(config["cbis_ddsm_csv_artifacts_removed_cropped"], f'{os.path.basename(_csv_path)}') +
                  '\033[0m' + ' file')

            print(f'     Images annotated in {os.path.basename(_csv_path)} .csv file saved into: ' + '\033[1m' +
                  os.path.join(config["cbis_ddsm_jpeg_artifacts_removed_cropped"], '') + '\033[0m' + ' file')

    config["biggest_img_dimensions"] = biggest_img_dimensions

    print('\n' + 'All ' + '\033[1m' + f'{dataset_name} .csv ' + '\033[0m' + 'files were processed sucessfully')

    return config["cbis_ddsm_csv_artifacts_removed_cropped"], config["cbis_ddsm_jpeg_artifacts_removed_cropped"]
