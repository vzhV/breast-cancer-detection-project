from tensorflow.keras.layers import Conv2D, BatchNormalization, MaxPool2D, Conv2DTranspose, Concatenate, Input, Dropout, \
    Add, Multiply, Activation, UpSampling2D, ReLU
from tensorflow.keras.models import Model


def conv_block(inputs, num_filters, dropout_rate=0.3):
    """
    Constructs a convolutional block using batch normalization, ReLU activations, and dropout.

    Args:
        inputs (tensor): Input tensor to the convolutional block.
        num_filters (int): Number of filters for the convolution layers.
        dropout_rate (float): Fraction of the input units to drop.

    Returns:
        tensor: Output tensor after applying convolutions, batch normalization, ReLU activations, and dropout.
    """
    x = inputs
    for _ in range(3):
        x = Conv2D(num_filters, 3, padding="same")(x)  # Convolution layer with 'same' padding.
        x = BatchNormalization()(x)  # Apply batch normalization.
        x = Activation("relu")(x)  # Apply ReLU activation function.

    x = Dropout(dropout_rate)(x)  # Apply dropout.

    if inputs.shape[-1] == num_filters:
        shortcut = inputs  # Use input as shortcut if number of filters unchanged.
    else:
        shortcut = Conv2D(num_filters, 1, padding="same")(inputs)  # 1x1 Convolution for dimension matching.
    x = Add()([x, shortcut])  # Add the shortcut to the output of the block.
    x = ReLU()(x)  # Apply ReLU activation function after adding the shortcut.
    return x


def encoder_block(inputs, num_filters):
    """
    Constructs an encoder block used in U-Net which applies a convolutional block followed by max pooling.

    Args:
        inputs (tensor): Input tensor to the encoder block.
        num_filters (int): Number of filters for the convolution layers.

    Returns:
        tuple: A tuple containing the output tensor before and after applying max pooling.
    """
    x = conv_block(inputs, num_filters)  # Apply a convolution block.
    p = MaxPool2D((2, 2))(x)  # Apply max pooling with a (2, 2) window.
    return x, p


def attention_gate(inputs, skip_features, filters, bn=True):
    """
    Constructs an attention gate for focusing on specific features, used in the U-Net architecture.

    Args:
        inputs (tensor): Input tensor to the attention gate.
        skip_features (tensor): Skip connection features from the encoder to be weighted.
        filters (int): Number of filters for the convolution operations.
        bn (bool): If True, applies batch normalization.

    Returns:
        tensor: Weighted skip connection features after applying the attention mechanism.
    """
    gating_signal = Conv2D(filters, kernel_size=3, padding='same', activation='relu', kernel_initializer='he_normal')(
        inputs)
    skip_signal = Conv2D(filters, kernel_size=3, strides=2, padding='same', activation='relu',
                         kernel_initializer='he_normal')(skip_features)
    combined = Add()([gating_signal, skip_signal])  # Combine gating and skip signal.
    attention_weights = Conv2D(1, kernel_size=1, padding='same', activation='sigmoid')(
        combined)  # Compute attention weights.
    attention_weights = UpSampling2D(size=(2, 2))(attention_weights)  # Upsample to match dimensions.
    weighted_skip = Multiply()([attention_weights, skip_features])  # Apply attention weights to the skip features.
    if bn:
        weighted_skip = BatchNormalization()(weighted_skip)  # Apply batch normalization.
    return weighted_skip


def decoder_block(inputs, skip_features, num_filters, dropout_rate=0.5):
    """
    Constructs a decoder block for U-Net, which upsamples the input and concatenates with skip features.

    Args:
        inputs (tensor): Input tensor to the decoder block.
        skip_features (tensor): Skip connection features from the corresponding encoder block.
        num_filters (int): Number of filters for the transposed convolution.
        dropout_rate (float): Fraction of the input units to drop.

    Returns:
        tensor: Output tensor after applying transposed convolution, concatenation, and dropout.
    """
    skip_features = attention_gate(inputs, skip_features, num_filters)  # Apply attention gate.
    x = Conv2DTranspose(num_filters, (2, 2), strides=(2, 2), padding='same')(inputs)  # Apply transposed convolution.
    x = Concatenate()([x, skip_features])  # Concatenate skip features with upsampled input.
    x = Dropout(dropout_rate)(x)  # Apply dropout.
    return x


def build_unet(input_shape):
    """
    Builds a U-Net model with an attention mechanism. The U-Net consists of an encoder (downsampling path),
    bottleneck, and a decoder (upsampling path) with skip connections enhanced by attention gates.

    Args:
        input_shape (tuple): Shape of the input image.

    Returns:
        Model: Compiled U-Net model with attention gates.
    """
    inputs = Input(input_shape)
    s1, p1 = encoder_block(inputs, 64)
    s2, p2 = encoder_block(p1, 128)
    s3, p3 = encoder_block(p2, 256)
    s4, p4 = encoder_block(p3, 512)

    b1 = conv_block(p4, 1024)  # Bottleneck part of U-Net.

    d1 = decoder_block(b1, s4, 512, 0.5)
    d2 = decoder_block(d1, s3, 256, 0.5)
    d3 = decoder_block(d2, s2, 128, 0.5)
    d4 = decoder_block(d3, s1, 64, 0.5)

    outputs = Conv2D(1, 1, padding="same", activation="sigmoid")(d4)  # Output layer with sigmoid activation.
    model = Model(inputs, outputs, name="ATTENTION_UNET")
    return model
