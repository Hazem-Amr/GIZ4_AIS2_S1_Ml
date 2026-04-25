import tensorflow as tf

class NeuralNetwork:

    def build(self, config):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(500, activation=config.ACTIVATION_FUNCTION, input_shape=(config.INPUT_SIZE,)),
            tf.keras.layers.Dense(100, activation=config.ACTIVATION_FUNCTION),
            tf.keras.layers.Dense(config.NUM_CLASSES, activation="softmax")
        ])

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=config.LEARNING_RATE),
            loss="categorical_crossentropy",
            metrics=["accuracy"]
        )

        return model