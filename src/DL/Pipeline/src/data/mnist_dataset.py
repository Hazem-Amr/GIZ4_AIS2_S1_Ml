import tensorflow as tf

class MNISTDataset:

    def load(self):
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

        # normalize
        x_train = x_train.astype("float32") / 255.0
        x_test = x_test.astype("float32") / 255.0

        # flatten
        x_train = x_train.reshape(-1, 784)
        x_test = x_test.reshape(-1, 784)

        # one-hot
        y_train = tf.keras.utils.to_categorical(y_train, 10)
        y_test = tf.keras.utils.to_categorical(y_test, 10)

        return x_train, y_train, x_test, y_test