import os
import tensorflow as tf

class Trainer:

    def train(self, model, x_train, y_train, config):

        log_dir = os.path.join("logs", config.EXPERIMENT_NAME)

        tensorboard = tf.keras.callbacks.TensorBoard(
            log_dir=log_dir,
            histogram_freq=1
        )

        history = model.fit(
            x_train,
            y_train,
            epochs=config.EPOCHS,
            batch_size=config.BATCH_SIZE,
            validation_split=0.2,
            callbacks=[tensorboard]
        )

        return history