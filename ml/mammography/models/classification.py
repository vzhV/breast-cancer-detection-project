from tensorflow.keras.applications import DenseNet201
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, Input
from tensorflow.keras.models import Model


def build_model(input_shape, num_classes, pretrained=True):
    """
    Builds a DenseNet-201 model according to specified parameters.

    Args:
        input_shape (tuple): Shape of the input images.
        num_classes (int): Number of classes for the output layer (used in the final dense layer with activation).
        pretrained (bool): Whether to load weights pre-trained on ImageNet.

    Returns:
        Model: A TensorFlow Keras model instance compiled with the DenseNet-201 architecture.
    """
    if pretrained:
        base_model = DenseNet201(weights='imagenet', include_top=False, input_shape=input_shape)
    else:
        base_model = DenseNet201(weights=None, include_top=False, input_shape=input_shape)

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)  # Optional Dropout for regularization
    outputs = Dense(num_classes, activation='sigmoid')(x)

    model = Model(inputs=base_model.input, outputs=outputs)
    return model