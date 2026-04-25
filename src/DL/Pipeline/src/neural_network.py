import tensorflow as tf

class NeuralNetwork:
    def __init__(self, config):
        self.config = config
        self.model = None
    
    def build(self):
        """Build the neural network model"""
        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(500, activation='sigmoid', input_shape=(self.config.INPUT_SIZE,)),
            tf.keras.layers.Dense(100, activation='sigmoid'),
            tf.keras.layers.Dense(self.config.NUM_CLASSES, activation='softmax')
        ])
        
        return self.model
