import io

import cv2
import numpy as np
import tensorflow as tf

from models.dto import PredictionDTO, MaskDTO, PointDTO
from models.input_models import MammogramAction


class MammogramService:
    """
        Service class for mammogram image processing which includes classification
        and segmentation of mammogram images using pre-trained deep learning models.

        Attributes:
            classification_model (tf.keras.Model): Model for classifying mammogram images.
            segmentation_model (tf.keras.Model): Model for segmenting mammogram images.
    """

    def __init__(self):
        """
            Initializes the MammogramService by loading the required machine learning models
            for classification and segmentation.
        """
        self.classification_model = self._load_model('weights/mg_class.h5')
        self.segmentation_model = self._load_model('weights/mg_segm.h5')

    def _load_model(self, model_path):
        """
            Loads a TensorFlow model from a given file path.

            Args:
                model_path (str): Path to the h5 file containing the TensorFlow model.

            Returns:
                tf.keras.Model or None: The loaded model, or None if loading fails.
        """
        try:
            model = tf.keras.models.load_model(model_path, compile=False)
            print(f"Model loaded successfully from {model_path}")
            return model
        except Exception as e:
            print(f"Failed to load model from {model_path}. Error: {e}")
            return None

    def predict(self, file, action):
        """
            Processes a mammogram image file using the specified action: classification, segmentation,
                or both, and returns the respective results in a structured format.

            Args:
                file: A file-like object containing the mammogram image data.
                action (MammogramAction): The action to perform (classification, segmentation, or both).

            Returns:
                PredictionDTO: Data transfer object containing the action performed, the classification
                                   result (if applicable), and the segmentation mask (if applicable).

            Raises:
                ValueError: If an invalid action is specified.
        """
        image = self._read_image(file)
        prediction = None
        mask = None

        if action == MammogramAction.CLASSIFICATION:
            prediction = self._predict_classification(image)
        elif action == MammogramAction.SEGMENTATION:
            mask = self._predict_segmentation(image)
        elif action == MammogramAction.ALL:
            prediction = self._predict_classification(image)
            mask = self._predict_segmentation(image)
        else:
            raise ValueError("Invalid action specified. Choose 'classification' or 'segmentation'.")
        return PredictionDTO(
            action=action,
            mask=mask,
            severity=self._get_class_name(prediction)
        )

    def _read_image(self, file):
        """
            Reads an image from a file-like object and converts it to a format suitable for model prediction.

            Args:
                file: A file-like object containing the image data.

            Returns:
                ndarray: The image in an array format suitable for processing.
        """
        image_bytes = io.BytesIO(file.file.read())
        image = cv2.imdecode(np.frombuffer(image_bytes.getvalue(), np.uint8), -1)
        return image

    def _get_class_name(self, prediction=None):
        """
            Determines the class name based on the model's prediction value.

            Args:
                prediction (float, optional): The prediction value from the classification model.

            Returns:
                str or None: The class name ('BENIGN' or 'MALIGNANT') or None if no prediction was made.
        """
        if not prediction:
            return None
        return 'BENIGN' if prediction <= 0.5 else 'MALIGNANT'

    def _predict_classification(self, image):
        """
            Predicts the class of a mammogram image using the classification model.

            Args:
                image (ndarray): The image to classify.

            Returns:
                float: The prediction probability of the image being malignant.
        """
        image = cv2.resize(image, (128, 128))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = np.expand_dims(image, axis=0) / 255.0
        prediction = self.classification_model.predict(image)
        return prediction[0][0]

    def _predict_segmentation(self, image):
        """
               Performs segmentation on a mammogram image and identifies lesions using the segmentation model.

               Args:
                   image (ndarray): The image to segment.

               Returns:
                   MaskDTO: Data transfer object containing the number of lesions and their coordinates.
               """

        # Image preparation for segmentation model
        if image.ndim == 2 or (image.ndim == 3 and image.shape[2] == 1):
            resized_image = cv2.resize(image, (128, 128))
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
            resized_image = cv2.resize(image, (128, 128))
            resized_image = cv2.cvtColor(resized_image, cv2.COLOR_RGB2GRAY)
        else:
            resized_image = cv2.resize(image, (128, 128))
            resized_image = cv2.cvtColor(resized_image, cv2.COLOR_RGB2GRAY)

        # Model prediction and post-processing
        original_shape = image.shape
        image_normalized = resized_image / 255.0
        image_expanded = np.expand_dims(np.expand_dims(image_normalized, axis=-1), axis=0)

        prediction = self.segmentation_model.predict(image_expanded)[0]

        prediction_resized = cv2.resize(prediction, (original_shape[1], original_shape[0]))

        _, binary_mask = cv2.threshold(prediction_resized, 0.5, 1, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours((binary_mask * 255).astype(np.uint8), cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)

        lesions = []

        for contour in contours:
            simplified_contour = cv2.approxPolyDP(contour, epsilon=1.0, closed=True)

            contour_points = simplified_contour[:, 0, :].tolist()
            lesions.append(contour_points)

        num_lesions = len(lesions)
        borders = []
        for lesion in lesions:
            curr_lesion = []
            for point in lesion:
                curr_lesion.append(PointDTO(x=point[0], y=point[1]))
            borders.append(curr_lesion)

        return MaskDTO(
            number_of_lesions=num_lesions,
            mask=borders
        )
