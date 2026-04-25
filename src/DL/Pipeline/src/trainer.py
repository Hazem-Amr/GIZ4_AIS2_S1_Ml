class Trainer:
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.history = None
    
    def train(self, x_train, y_train):
        """Train the model"""
        self.model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.history = self.model.fit(
            x_train, y_train,
            epochs=self.config.EPOCHS,
            batch_size=self.config.BATCH_SIZE,
            validation_split=self.config.VALIDATION_SPLIT,
            verbose=1
        )
        
        return self.history
