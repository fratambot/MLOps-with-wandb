import tensorflow as tf
import tensorflow.keras.layers as tfl


def CNN_model(config):
    model = tf.keras.Sequential()
    model.add(tfl.Input(shape=(28, 28, 1), name="input_layer"))
    model.add(
        tfl.Conv2D(32, kernel_size=5, padding="same", activation="relu", name="conv_1")
    )
    model.add(tfl.MaxPooling2D())
    model.add(tfl.Dropout(config.dropout_1, name="dropout_1"))
    model.add(
        tfl.Conv2D(64, kernel_size=5, padding="same", activation="relu", name="conv_2")
    )
    model.add(tfl.MaxPooling2D())
    model.add(tfl.Dropout(config.dropout_2, name="dropout_2"))
    model.add(tfl.Flatten())
    model.add(tfl.Dense(config.dense, activation="relu", name="dense"))
    model.add(tfl.Dropout(config.dropout_3, name="dropout_3"))
    model.add(tfl.Dense(10, activation="softmax", name="softmax"))

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.lr),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
    )

    print(model.summary())

    return model
