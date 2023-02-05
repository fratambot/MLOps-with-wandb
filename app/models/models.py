import os
import sys
import tensorflow as tf
import tensorflow.keras.layers as tfl

# we use /app folder as base_path
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# and we add it at the beginning of PYTHONPATH otherwise we
# cannot import from modules at the same level
sys.path.insert(0, base_path)


def CNN_model(lr):
    model = tf.keras.Sequential()
    model.add(tfl.Input(shape=(28, 28, 1), name="input_layer"))
    model.add(
        tfl.Conv2D(32, kernel_size=5, padding="same", activation="relu", name="conv_1")
    )
    model.add(tfl.MaxPooling2D())
    model.add(tfl.Dropout(0.4, name="dropout_1"))
    model.add(
        tfl.Conv2D(64, kernel_size=5, padding="same", activation="relu", name="conv_2")
    )
    model.add(tfl.MaxPooling2D())
    model.add(tfl.Dropout(0.4, name="dropout_2"))
    model.add(tfl.Flatten())
    model.add(tfl.Dense(128, activation="relu", name="dense"))
    model.add(tfl.Dropout(0.4, name="dropout_3"))
    model.add(tfl.Dense(10, activation="softmax", name="softmax"))

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="categorical_crossentropy",
        metrics=[tf.keras.metrics.CategoricalAccuracy()],
    )

    print(model.summary())

    return model
