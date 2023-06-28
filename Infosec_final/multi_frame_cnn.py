import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.regularizers import l2


def build_multiframe_model(input_shape: tuple, internal_model_name: str, model: str = "default"):
    """

    :param input_shape: shape of the input
    :param internal_model_name: informal name for the model; one of: "left", "right", "center", "thermal"
    :param model: identifies the model you want to return
    :return:
    """
    # print("input sequence shape ", input_shape)
    video_inputs = keras.Input(shape=input_shape, name="frame_sequence")
    timestamp_inputs = keras.Input(shape=(2,), name="timestamps")
    key_pred = None

    if model == "default":

        # TIME_DISTRIBUTED[CONV + POOLING]* -> TIME_DISTRIBUTED[FLATTEN] -> [LSTM] -> [CONCAT] -> [FC]* -> [SOFTMAX]

        # CONVOLUTION + POOLING
        x = layers.TimeDistributed(
            layers.Conv2D(filters=32, kernel_size=(3, 3), activation="relu")
        )(video_inputs)

        # removing this causes OOM
        x = layers.TimeDistributed(
            layers.MaxPooling2D(pool_size=(2, 2))
        )(x)
        # x = layers.Dropout(rate=0.3)(x)

        # kernel size is 9x9 because 3x3 would cause OOM error
        # CONVOLUTION + POOLING
        x = layers.TimeDistributed(
            layers.Conv2D(filters=64, kernel_size=(9, 9), activation="relu")
        )(x)

        x = layers.TimeDistributed(
            layers.MaxPooling2D(pool_size=(2, 2))
        )(x)

        # CONVOLUTION + POOLING
        x = layers.TimeDistributed(
            layers.Conv2D(filters=128, kernel_size=(3, 3), activation="relu")
        )(x)

        x = layers.TimeDistributed(
            layers.MaxPooling2D(pool_size=(2, 2))
        )(x)

        # CONVOLUTION + POOLING
        x = layers.TimeDistributed(

            layers.Conv2D(filters=256, kernel_size=(3, 3), activation="relu")
        )(x)

        x = layers.TimeDistributed(
            layers.MaxPooling2D(pool_size=(2, 2))
        )(x)

        x = layers.TimeDistributed(layers.Dropout(rate=0.1)
                                   )(x)

        print("Shape before any flatten:", x.shape)

        x = layers.TimeDistributed(
            layers.Flatten()
        )(x)

        # GRU or LSTM
        x = layers.LSTM(units=128, activation="tanh")(x)

        # The timestamp is actually deactivated. We tested the performance of the model also with the timestamps, but
        # the results are similar. The following concatenation line of code does not affect the model an can be ignored.
        x = layers.concatenate([x, timestamp_inputs])

        x = layers.Dense(units=64, activation="tanh")(x)
        x = layers.Dense(units=64, activation="tanh")(x)
        x = layers.Dense(units=64, activation="tanh")(x)
        x = layers.Dense(units=64, activation="tanh")(x)

        key_pred = layers.Dense(10, activation="softmax")(x)

    elif model == "conv3d":  # obsolete model

        # [CONV + POOLING]* -> [FLATTEN] -> [CONCAT] -> [FC]* -> [SOFTMAX]

        # the real model is deeper, see pg. 5 -> https://arxiv.org/pdf/1412.0767.pdf

        # Conv1a
        x = layers.Conv3D(filters=16, kernel_size=(9, 9, 9), activation="relu")(video_inputs)
        x = layers.MaxPooling3D(pool_size=(2, 2, 2))(x)

        x = layers.MaxPooling3D(pool_size=(2, 2, 2))(x)

        x = layers.Flatten()(x)

        x = layers.concatenate([x, timestamp_inputs])

        x = layers.Dense(units=32, activation="relu")(x)
        x = layers.Dropout(rate=0.2)(x)

        key_pred = layers.Dense(10, activation="softmax")(x)

    else:
        assert False, f"[ASSERTION ERROR] -> The model you specified ({model}) does not exist."

    # return keras.Model(inputs=[video_inputs, timestamp_inputs], outputs=key_pred, name=internal_model_name)
    return keras.Model(inputs=[video_inputs, timestamp_inputs], outputs=key_pred, name=internal_model_name)
