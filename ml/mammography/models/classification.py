from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, Concatenate, AveragePooling2D, \
    GlobalAveragePooling2D, Dense, Input
from tensorflow.keras.layers import Dropout
from tensorflow.keras.models import Model


def conv_block(x, growth_rate):
    """
    Creates a convolutional block used within a Dense Block, incorporating batch normalization, ReLU activation,
    convolutional layers, and dropout for regularization.

    Args:
        x (tensor): Input tensor to the convolutional block.
        growth_rate (int): The number of filters to add per convolutional block.

    Returns:
        tensor: Output tensor of the convolutional block.
    """
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # Bottleneck layer, reducing dimension.
    x = Conv2D(4 * growth_rate, (1, 1), use_bias=False)(x)

    # Dropout for regularization.
    x = Dropout(0.2)(x)

    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # Convolution preserving spatial dimensions.
    x = Conv2D(growth_rate, (3, 3), padding='same', use_bias=False)(x)
    return x


def dense_block(x, num_convs, growth_rate):
    """
    Constructs a Dense Block where each conv_block is concatenated with the feature-maps of all preceding layers.

    Args:
        x (tensor): Input tensor to the dense block.
        num_convs (int): Number of convolutional blocks within the dense block.
        growth_rate (int): The number of filters to add per convolutional block.

    Returns:
        tensor: Output tensor after passing through the dense block.
    """
    for _ in range(num_convs):
        cb = conv_block(x, growth_rate)
        # Concatenate output of conv_block with input feature maps.
        x = Concatenate()([x, cb])
    return x


def transition_layer(x):
    """
    Creates a transition layer between two dense blocks, which does batch normalization, ReLU activation,
    convolution and average pooling to reduce dimensionality.

    Args:
        x (tensor): Input tensor to the transition layer.

    Returns:
        tensor: Output tensor after dimensionality reduction.
    """
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(x.shape[-1] // 2, (1, 1), use_bias=False)(x)  # Convolution reducing the number of filters.
    x = AveragePooling2D((2, 2), strides=2)(x)  # Pooling to reduce spatial dimensions.
    return x


def build_model(input_shape, num_classes):
    """
    Builds a model according to specified parameters.

    Args:
        input_shape (tuple): Shape of the input images.
        num_classes (int): Number of classes for the output layer (used in the final dense layer with activation).

    Returns:
        Model: A TensorFlow Keras model instance compiled with the DenseNet architecture.
    """
    inputs = Input(shape=input_shape)
    x = Conv2D(64, (7, 7), strides=2, padding='same', use_bias=False)(inputs)  # Initial convolution layer.
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = AveragePooling2D((3, 3), strides=2, padding='same')(x)  # Initial pooling to reduce dimension.

    # Stack of dense blocks and transition layers
    x = dense_block(x, num_convs=8, growth_rate=48)
    x = transition_layer(x)
    x = dense_block(x, num_convs=16, growth_rate=48)
    x = transition_layer(x)
    x = dense_block(x, num_convs=32, growth_rate=48)
    x = transition_layer(x)
    x = dense_block(x, num_convs=32, growth_rate=48)

    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = GlobalAveragePooling2D()(x)
    outputs = Dense(num_classes, activation='sigmoid')(x)

    model = Model(inputs=inputs, outputs=outputs)
    return model
