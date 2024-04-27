from io import BytesIO

import cv2
import numpy as np
import tensorflow as tf
import torch
import torchvision.transforms as transforms
from PIL import Image

from models.dto import PredictionDTO, PointDTO, MaskDTO
from models.input_models import UltrasoundAction
from models.ml_model_torch import MedicalImageClassifier


class UltrasoundService:
    """
        Service class for ultrasound image processing, offering classification and
        segmentation capabilities using TensorFlow for segmentation and PyTorch for
        classification.

        Attributes:
            segmentation_model (tf.keras.Model): TensorFlow model for image segmentation.
            classification_overlaid_model (torch.nn.Module): PyTorch model for classification of overlaid images.
            classification_model (torch.nn.Module): PyTorch model for standard image classification.
            class_names (list of str): Names of the classes used in classification.
    """
    def __init__(self):
        """
            Initializes the UltrasoundService by loading the required machine learning models
            for classification and segmentation.
        """
        self.segmentation_model = self._load_model_segmentation('weights/us_segm.h5')
        self.classification_overlaid_model = self._load_model_classification('weights/us_class_overlaid.pth')
        self.classification_model = self._load_model_classification('weights/us_class.pth')
        self.class_nems = ['BENIGN', 'MALIGNANT', 'NORMAL']

    def _load_model_segmentation(self, model_path):
        """
            Loads a TensorFlow model for segmentation from the specified file path.

            Args:
                model_path (str): Path to the model file.

            Returns:
                tf.keras.Model or None: Loaded model or None if an error occurs.
        """
        try:
            model = tf.keras.models.load_model(model_path, compile=False)
            print(f"Model loaded successfully from {model_path}")
            return model
        except Exception as e:
            print(f"Failed to load model from {model_path}. Error: {e}")
            return None

    def _load_model_classification(self, model_path):
        """
            Loads a PyTorch model for classification from the specified file path.

            Args:
                model_path (str): Path to the model file.

            Returns:
                torch.nn.Module: Loaded PyTorch model.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = MedicalImageClassifier(3)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        return model

    def predict(self, file, action):
        """
            Processes an ultrasound image file based on the specified action and returns the results.

            Args:
                file: A file-like object containing the ultrasound image data.
                action (UltrasoundAction): The action to perform on the image.

            Returns:
                PredictionDTO: Object containing the results of the specified action.

             Raises:
                ValueError: If an unrecognized action is specified.
        """
        prediction = None
        severity_mask = None
        mask = None

        if action == UltrasoundAction.CLASSIFICATION:
            prediction = self._predict_classification(file)
        elif action == UltrasoundAction.SEGMENTATION:
            file_bytes = file.file.read()
            mask = self._predict_segmentation(file_bytes)
        elif action == UltrasoundAction.CLASSIFICATION_OVERLAYED:
            severity_mask, mask = self._predict_classification_overlaid(file)
        else:
            raise ValueError("Invalid action specified. Choose 'CLASSIFICATION' or 'SEGMENTATION'.")
        return PredictionDTO(
            action=action,
            mask=mask,
            severity=prediction,
            severity_mask=severity_mask
        )

    def _read_image(self, image_bytes):
        """
            Reads an image from bytes and converts it into a format suitable for processing.

            Args:
                image_bytes (bytes): Image data in bytes.

            Returns:
                ndarray: The image as an ndarray.
        """
        image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), -1)
        return image

    def load_and_transform_image_from_upload(self, file, image_bytes=False):
        """
            Loads an image from an uploaded file and applies necessary transformations for model prediction.

            Args:
                file: File-like object or image in bytes format.
                image_bytes (bool): If True, treat 'file' as image bytes; otherwise, read from file-like object.

            Returns:
                torch.Tensor: Transformed image tensor.
        """
        normalize = transforms.Normalize(mean=[0.485], std=[0.229])
        data_transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize
        ])
        if image_bytes:
            image = file
        else:
            image_data = file.file.read()
            image = Image.open(BytesIO(image_data)).convert('RGB')
        image = data_transform(image)
        return image

    def _predict_classification(self, file, use_bytes=False):
        """
            Predicts the classification of an ultrasound image.

            Args:
                file: File-like object or image in bytes format.
                use_bytes (bool): If True, treat 'file' as image bytes; otherwise, read from file-like object.

            Returns:
                str: Predicted class name.
        """
        image_tensor = self.load_and_transform_image_from_upload(file, image_bytes=use_bytes)
        image_tensor = image_tensor.unsqueeze(0)
        with torch.no_grad():
            outputs = self.classification_model(image_tensor)
            _, predicted = torch.max(outputs.data, 1)
            return self.class_nems[predicted.item()]

    def _predict_segmentation(self, image_bytes):
        """
            Performs segmentation on an ultrasound image to identify lesions.

            Args:
                image_bytes (bytes): Image data in bytes.

            Returns:
                MaskDTO: Object containing the segmentation results.
        """
        image = self._read_image(image_bytes)
        if image.ndim == 2 or (image.ndim == 3 and image.shape[2] == 1):
            resized_image = cv2.resize(image, (128, 128))
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
            resized_image = cv2.resize(image, (128, 128))
            resized_image = cv2.cvtColor(resized_image, cv2.COLOR_RGB2GRAY)
        else:
            resized_image = cv2.resize(image, (128, 128))
            resized_image = cv2.cvtColor(resized_image, cv2.COLOR_RGB2GRAY)

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

    def _predict_classification_overlaid(self, file, use_bytes=False):
        """
            Performs classification on an overlaid ultrasound image created by blending the original image
            and its segmentation mask.

            Args:
                file: File-like object or image in bytes format.
                use_bytes (bool): If True, treat 'file' as image bytes; otherwise, read from file-like object.

            Returns:
                tuple: Predicted class name and the segmentation mask DTO.
        """
        file_bytes = file.file.read() if not use_bytes else file
        image = self._read_image(file_bytes)
        mask_dto = self._predict_segmentation(file_bytes)
        image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        mask_image = np.zeros_like(image, dtype=np.uint8)
        for lesion in mask_dto.mask:
            lesion_points = np.array([[point.x, point.y] for point in lesion], dtype=np.int32)
            cv2.fillPoly(mask_image, [np.array(lesion_points)], (255, 255, 255))

        mask_pil = Image.fromarray(mask_image[:, :, 0])
        mask_pil = mask_pil.convert("L")

        overlayed_image = Image.blend(image_pil, mask_pil.convert('RGB'), alpha=0.5)

        overlayed_image = overlayed_image.convert('RGB')
        overlayed_image_tensor = self.load_and_transform_image_from_upload(overlayed_image, image_bytes=True)
        overlayed_image_tensor = overlayed_image_tensor.unsqueeze(0)

        with torch.no_grad():
            outputs = self.classification_overlaid_model(overlayed_image_tensor)
            _, predicted = torch.max(outputs.data, 1)
            predicted_class = self.class_nems[predicted.item()]

        return predicted_class, mask_dto
