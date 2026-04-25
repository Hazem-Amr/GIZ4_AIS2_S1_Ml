import numpy as np
from keras.datasets import mnist
import tensorflow as tf

class MNISTDataset:
    def __init__(self, config):
        self.config = config
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None
    
    def load(self):
        """Load and preprocess MNIST data"""
        # Load data
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        
        # Scale pixel values to [0, 1]
        x_train = x_train.astype('float32') / 255.0
        x_test = x_test.astype('float32') / 255.0
        
        # Reshape to flatten (N, 784)
        x_train = x_train.reshape(-1, self.config.INPUT_SIZE)
        x_test = x_test.reshape(-1, self.config.INPUT_SIZE)
        
        # One-hot encode labels
        y_train_oh = tf.keras.utils.to_categorical(y_train, self.config.NUM_CLASSES)
        y_test_oh = tf.keras.utils.to_categorical(y_test, self.config.NUM_CLASSES)
        
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train_oh
        self.y_test = y_test_oh
        
        return self
