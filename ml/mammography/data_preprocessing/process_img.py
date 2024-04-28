import os
from typing import Union, Tuple

import cv2
import numpy as np


class MammogramInitImageProcessor:
    """
    Class for preprocessing mammogram images.
    """

    def __init__(self):
        self.t_crop = self.b_crop = 0.035
        self.r_crop = self.l_crop = 0.0125

        self.blur_size = 3
        self.threshold = 18
        self.kernel_size = 15
        self.dilation_size = 35

    def remove_artifacts_from_img(self, _img_path: str,
                                  _cropped_area_path: Union[str, None],
                                  _roi_mask_path: Union[str, None],
                                  _input_path: str, _output_path: str):
        """
        Process an image to remove artifacts, possibly apply horizontal flip and save the processed image.

        Arguments:
        _img_path -- path to the image to be processed
        _cropped_area_path -- path to the cropped area image
        _roi_mask_path -- path to the region of interest (ROI) mask image
        _input_path -- directory path of the input image
        _output_path -- directory path for the output processed image

        """
        img = cv2.imread(
            _img_path,
            cv2.IMREAD_GRAYSCALE
        )

        cropped_img = self.crop_borders(_img=img)

        mask = self.create_breast_mask(
            _img=cropped_img, fill_holes_bool=False,
            dilate_mask_bool=True, smooth_boundary_bool=True
        )

        # Apply the mask to the image to remove artifacts
        img_artifacts_removed = self.apply_mask(_img=cropped_img, _mask=mask)

        # Apply horizontal flip if needed
        left_right_flip_bool = self.check_left_right_flip(_mask=mask)
        if left_right_flip_bool:
            img_processed = self.make_left_right_flip(_img=img_artifacts_removed)
        else:
            img_processed = img_artifacts_removed

        _new_img_path = _img_path.replace(os.path.basename(_input_path), os.path.basename(_output_path))
        cv2.imwrite(_new_img_path, img_processed)

        if _cropped_area_path is None:
            pass
        else:
            cropped_area_img = cv2.imread(
                _cropped_area_path,
                cv2.IMREAD_GRAYSCALE
            )

            # Process the cropped area image (possibly flip)
            cropped_area_img_processed = self.cropped_area_processor(
                _img=cropped_area_img,
                _left_right_flip_bool=left_right_flip_bool
            )

            _new_cropped_area_path = _cropped_area_path.replace(os.path.basename(_input_path),
                                                                os.path.basename(_output_path))
            cv2.imwrite(_new_cropped_area_path, cropped_area_img_processed)

        if _roi_mask_path is None:
            pass
        else:
            roi_mask_img = cv2.imread(
                _roi_mask_path,
                cv2.IMREAD_GRAYSCALE
            )

            if img.shape != roi_mask_img.shape:
                pass
            roi_mask_img = self.resize_mask(
                _img=img,
                _mask=roi_mask_img
            )

            # Process the ROI mask image (crop and possibly flip)
            roi_mask_img_processed = self.roi_mask_processor(
                _mask=roi_mask_img,
                _left_right_flip_bool=left_right_flip_bool
            )

            _new_roi_mask_path = _roi_mask_path.replace(os.path.basename(_input_path), os.path.basename(_output_path))
            cv2.imwrite(_new_roi_mask_path, roi_mask_img_processed)

    def crop_borders(self, _img: np.ndarray):
        """
        Crop an image from the specified percentages of the top, bottom, right, and left sides.

        Arguments:
        _img -- image to crop (np.ndarray)

        Shadow arguments:
        self.t_crop -- percentage of cropping from the top side
        self.b_crop -- percentage of cropping from the bottom side
        self.r_crop -- percentage of cropping from the right side
        self.l_crop -- percentage of cropping from the left side

        Returns:
        cropped_img -- the cropped image (np.ndarray)

        """
        height, width = _img.shape[:2]

        top_pixels = int(height * self.t_crop)
        bottom_pixels = int(height * self.b_crop)
        right_pixels = int(width * self.r_crop)
        left_pixels = int(width * self.l_crop)

        _cropped_img = _img[top_pixels:height - bottom_pixels, left_pixels:width - right_pixels]

        return _cropped_img

    def create_breast_mask(self, _img: np.ndarray,
                           fill_holes_bool: bool = False,
                           dilate_mask_bool: bool = False,
                           smooth_boundary_bool: bool = False):
        """
        Create mask over breast area on the mammogram (remove other artifacts from the image).

        Arguments:
        _img (np.ndarray) -- image to build mask on
        fill_holes (boolean) -- whether fill the holes inside the largest object or not
        smooth_boundary (boolean) -- whether smooth the boundary of the largest
            object using morphological opening or not

        Returns:
        dilated_mask (np.ndarray) -- mask over breast area

        """

        # Blur the image to smooth out small artifacts
        blurred = cv2.medianBlur(_img, self.blur_size)

        largest_mask = self.select_largest_object(blurred)

        # If fill_holes flag is True, the function fills any holes inside the largest object
        if fill_holes_bool:
            largest_mask = self.fill_holes_in_mask(largest_mask)
        else:
            pass

        # If dilate_mask_bool flag is True, the function applies dilation to the mask
        if dilate_mask_bool:
            largest_mask = self.dilate_mask(largest_mask)
        else:
            pass

        # If smooth_boundary flag is True, the function applies a morphological operation (opening)
        # to smooth the boundary of the largest object
        if smooth_boundary_bool:
            largest_mask = self.smooth_boundary(largest_mask)
        else:
            pass

        return largest_mask

    def select_largest_object(self, _img: np.ndarray) -> np.ndarray:
        """
        Selects the largest object in the binary image.

        This function converts the image to a binary image, identifies the largest
        connected component (assumed to be the breast), and creates a mask with only
        the largest connected component.

        Arguments:
        _img -- the input image as a numpy array

        Returns:
        _largest_mask -- a binary mask with only the largest connected component

        """
        # Create a binary image by thresholding
        # cv2.THRESH_BINARY will set pixels with intensity greater than self.threshold to 255 (max value)
        _, binarised_img = cv2.threshold(_img, self.threshold, 255, cv2.THRESH_BINARY)

        # Identify all connected components in the binary image using connectedComponentsWithStats
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binarised_img, connectivity=8, ltype=cv2.CV_32S)

        # The background is also considered a component and usually has the largest area
        # So, find the component with the second-largest area, which is assumed to be the breast
        _largest_component = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1

        # Create a binary mask with only the largest connected component (the breast)
        _largest_mask = np.zeros_like(binarised_img, dtype=np.uint8)
        _largest_mask[labels == _largest_component] = 255

        return _largest_mask

    @staticmethod
    def fill_holes_in_mask(_mask: np.ndarray) -> np.ndarray:
        """
        Fills holes in a binary mask.

        This function uses flood fill to fill holes, and combines the filled and original
        mask to create the final mask.

        Arguments:
        _mask -- the input binary mask as a numpy array

        Returns:
        mask_filled -- the binary mask with filled holes

        """
        # Add a border of 0 (black) pixels around the mask
        mask_border = cv2.copyMakeBorder(_mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)

        # Prepare for flood filling by creating an image that's 2 pixels bigger in both dimensions than the mask
        h, w = mask_border.shape[:2]
        mask_floodfill = np.zeros((h + 2, w + 2), np.uint8)

        # Use flood fill to change connected background (black) pixels to a temporary value (128)
        cv2.floodFill(mask_floodfill, None, (0, 0), 128)

        # Invert the flood-filled image, changing the temporary value pixels back to background (black) and vice versa
        mask_floodfill_inv = cv2.bitwise_not(mask_floodfill)

        # Combine the flood-filled image with the original mask to get the final mask
        # This will leave black pixels only where there were "holes" in the original mask
        mask_filled = mask_border | mask_floodfill_inv[1:-1, 1:-1]

        # Remove the border
        mask_filled = mask_filled[1:-1, 1:-1]

        return mask_filled

    def dilate_mask(self, _mask: np.ndarray) -> np.ndarray:
        """
        Dilate a binary image mask.

        Arguments:
        _mask -- a 2D binary mask (np.ndarray) to dilate

        Shadow arguments:
        self.dilation_size -- size of dilation operation. It determines the size of structuring element
            used in the dilation operation

        Returns:
        _dilated_mask -- the dilated binary mask (np.ndarray)

        """
        # Create a structuring element for dilation
        se = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * self.dilation_size + 1, 2 * self.dilation_size + 1))

        # Dilate the mask
        # Dilation adds pixels to the boundaries of objects in an image.
        _dilated_mask = cv2.dilate(_mask, se)

        return _dilated_mask

    def smooth_boundary(self, _mask: np.ndarray) -> np.ndarray:
        """
        Smooth the boundary of objects in a binary image mask using morphological opening operation.

        Arguments:
        _mask -- a 2D binary mask (np.ndarray) whose boundaries are to be smoothed

        Shadow arguments:
        self.kernel_size -- the size of the kernel used in the morphological operation.
                            A larger kernel size results in more smoothing

        Returns:
        _mask -- the binary mask with smoothed boundaries (np.ndarray)

        """
        # Create a structuring element for the morphological operation
        kernel_ = np.ones((self.kernel_size, self.kernel_size), dtype=np.uint8)

        # Apply the morphological operation (opening)
        _mask = cv2.morphologyEx(_mask, cv2.MORPH_OPEN, kernel_)

        return _mask

    @staticmethod
    def apply_mask(_img: np.ndarray, _mask: np.ndarray) -> np.ndarray:
        """
        Apply a binary mask to an image.

        Arguments:
        _img -- the original image (np.ndarray) to which the mask is to be applied
        _mask -- a binary mask (np.ndarray)

        Returns:
        _masked_img -- the image after applying the mask (np.ndarray)

        """
        # Apply the mask to the image
        _masked_img = cv2.bitwise_and(_img, _img, mask=_mask)

        return _masked_img

    @staticmethod
    def check_left_right_flip(_mask: np.ndarray) -> bool:
        """
        Check if the mask has more white pixels on the right side than the left side.

        This can be used to determine if an image should be flipped horizontally,
        to ensure a consistent orientation for further processing.

        Arguments:
        _mask -- a binary mask (np.ndarray) to be checked

        Returns:
        _left_right_flip_bool -- True if there are more white pixels on the right side, False otherwise

        """
        # Get number of rows and columns in the image
        n_rows, n_cols = _mask.shape
        x_center = n_cols // 2

        # Sum down each column. This gives the total number of white pixels in each column
        col_sum = _mask.sum(axis=0)

        # Calculate the sum of the white pixels on the left and right sides of the image
        left_sum = sum(col_sum[0:x_center])
        right_sum = sum(col_sum[x_center:-1])

        # Determine if there are more white pixels on the right side than the left side
        if left_sum < right_sum:
            _left_right_flip_bool = True
        else:
            _left_right_flip_bool = False

        return _left_right_flip_bool

    @staticmethod
    def make_left_right_flip(_img: np.ndarray) -> np.ndarray:
        """
        Flip an image horizontally (left-right).

        Arguments:
        _img -- the original image (np.ndarray) to be flipped

        Returns:
        flipped_img -- the horizontally flipped image (np.ndarray)

        """
        # Perform the left-right flip using numpy's flip function
        # This essentially creates a mirror image of the original one
        flipped_img = np.fliplr(_img)

        return flipped_img

    @staticmethod
    def resize_mask(_img: np.ndarray, _mask: np.ndarray) -> np.ndarray:
        """
        Resize a mask to match the dimensions of a given image.

        Arguments:
        _img -- the image (np.ndarray) whose dimensions the mask should be resized to
        _mask -- the mask (np.ndarray) to be resized

        Returns:
        _resized_mask -- the resized mask (np.ndarray)

        """
        # Check if the mask and the image have the same dimensions
        if _img.shape == _mask.shape:
            # If they have the same dimensions, there's no need to resize
            _resized_mask = _mask
        else:
            # Get the dimensions of the image
            dimensions = _img.shape

            # Resize the mask to match the image's dimensions
            _resized_mask = cv2.resize(_mask, (dimensions[1], dimensions[0]),
                                       interpolation=cv2.INTER_AREA)

        return _resized_mask

    def cropped_area_processor(self, _img: np.ndarray, _left_right_flip_bool: bool) -> np.ndarray:
        """
        Process a cropped area of an image. If _left_right_flip_bool is True, the image is flipped horizontally.

        Arguments:
        _img -- the input image as a numpy array.
        _left_right_flip_bool -- boolean indicating whether a left-right flip is required.

        Returns:
        _flipped_img -- the processed image, which may have been flipped horizontally.

        """
        # Apply horizontal flip if needed
        if _left_right_flip_bool:
            _flipped_img = self.make_left_right_flip(_img=_img)
        else:
            _flipped_img = _img

        return _flipped_img

    def roi_mask_processor(self, _mask: np.ndarray, _left_right_flip_bool: bool) -> np.ndarray:
        """
        Process a region of interest (ROI) mask. The mask is first cropped, and then possibly flipped horizontally.

        Arguments:
        _mask -- the input mask as a numpy array.
        _left_right_flip_bool -- boolean indicating whether a left-right flip is required.

        Returns:
        _flipped_mask -- the processed mask, which may have been cropped and flipped.

        """
        # Initial cropping
        _cropped_mask = self.crop_borders(_img=_mask)

        # Apply horizontal flip if needed
        if _left_right_flip_bool:
            _flipped_mask = self.make_left_right_flip(_img=_cropped_mask)
        else:
            _flipped_mask = _cropped_mask

        return _flipped_mask


class ImageShapeProcessing(MammogramInitImageProcessor):
    """
    Class for preprocessing shapes of mammogram images.

    This class inherits from MammogramInitImageProcessor and adds functionality for
    cropping and resizing mammograms images to match specific aspect ratios.

    Attributes:
    crop_limit_value -- The limit to the amount that images can be cropped
    padding_chunk -- The chunk size for padding images

    """

    def __init__(self):
        # Call the constructor of the parent class
        super(ImageShapeProcessing, self).__init__()

        # Initialize variables
        self.crop_limit_value = 50
        self.padding_chunk = 128

    def crop_image_by_mask(self, _img_path: str,
                           _roi_mask_path: Union[str, None],
                           _input_path: str, _output_path: str) -> tuple:
        """
        Crops image and mask by a specified mask.

        This function reads the image and ROI mask, creates a mask over the breast area on the mammogram,
        determines the cropping borders, crops the image and mask by these borders, and saves the processed images.

        Arguments:
        _img_path -- the path to the image file
        _roi_mask_path -- the path to the ROI mask file
        _input_path -- the path to the input directory
        _output_path -- the path to the output directory

        Returns:
        img_processed.shape -- a tuple of the shape of the processed image (height, width)

        """
        # Read the mammogram image in grayscale
        img = cv2.imread(
            _img_path,
            cv2.IMREAD_GRAYSCALE
        )

        # Create a mask over the breast area on the mammogram
        mask = self.create_breast_mask(
            _img=img, fill_holes_bool=False,
            dilate_mask_bool=True, smooth_boundary_bool=True
        )

        # Determine the cropping borders from the mask
        r_min, r_max, c_min, c_max = self.determine_cropping_borders(_mask=mask)

        # Crop the image by the determined borders
        img_processed = self.crop_image_by_borders(
            _image=img, _r_min=r_min, _r_max=r_max, _c_min=c_min, _c_max=c_max
        )

        # Replace the original image path with new path and save the processed image
        _new_img_path = _img_path.replace(os.path.basename(_input_path), os.path.basename(_output_path))
        cv2.imwrite(_new_img_path, img_processed)

        if _roi_mask_path is None:
            pass
        else:
            # Read the ROI mask image in grayscale
            roi_mask_img = cv2.imread(
                _roi_mask_path,
                cv2.IMREAD_GRAYSCALE
            )

            # Crop the mask by the determined borders
            roi_mask_img_processed = self.crop_image_by_borders(
                _image=roi_mask_img, _r_min=r_min, _r_max=r_max, _c_min=c_min, _c_max=c_max
            )

            # Replace the ROI mask path with new path and save the processed image
            _new_roi_mask_path = _roi_mask_path.replace(os.path.basename(_input_path), os.path.basename(_output_path))
            cv2.imwrite(_new_roi_mask_path, roi_mask_img_processed)

        # Return the shape of the processed image
        return img_processed.shape

    @staticmethod
    def determine_cropping_borders(_mask: np.ndarray) -> tuple:
        """
        Determine the cropping borders based on the given mask.

        This function finds the indices of the rows and columns of the mask
        where there is breast tissue (mask value is True or 1), and determines
        the range (min to max) of rows and columns that contain breast tissue.

        Arguments:
        _mask -- a 2D numpy array representing the mask

        Returns:
        _r_min, _r_max, _c_min, _c_max -- the minimum and maximum row and column indices
        that contain breast tissue

        """
        # Find the rows and columns where there is breast tissue (the mask is 1 or True)
        rows = np.any(_mask, axis=1)
        cols = np.any(_mask, axis=0)

        # Find the range of rows and columns that contain breast tissue
        _r_min, _r_max = np.where(rows)[0][[0, -1]]
        _c_min, _c_max = np.where(cols)[0][[0, -1]]

        return _r_min, _r_max, _c_min, _c_max

    def crop_image_by_borders(self, _image: np.ndarray, _r_min: int,
                              _r_max: int, _c_min: int, _c_max: int) -> np.ndarray:
        """
        Crop the given image based on specified borders.

        The image is cropped based on the given minimum and maximum rows
        and columns. However, a margin is added to both sides of the rows,
        and to the right side of the columns. The size of the margin is
        determined by `self.crop_limit_value`.

        Arguments:
        _image -- a 2D numpy array representing the image
        _r_min, _r_max, _c_min, _c_max -- the minimum and maximum row and column indices
        for cropping.

        Returns:
        _cropped_img -- the cropped image as a 2D numpy array

        """
        # Crop image by specified borders
        _cropped_img = _image[
                       max(_r_min - int(self.crop_limit_value / 2), 0):_r_max + int(self.crop_limit_value / 2),
                       _c_min:_c_max + self.crop_limit_value]

        return _cropped_img
