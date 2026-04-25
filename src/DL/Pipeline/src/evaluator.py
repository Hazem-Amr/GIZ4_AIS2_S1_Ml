import numpy as np

class Evaluator:
    def __init__(self, model):
        self.model = model
    
    def evaluate(self, x_test, y_test):
        """Evaluate model on test set"""
        test_loss, test_acc = self.model.evaluate(x_test, y_test, verbose=0)
        
        print(f'Test Loss: {test_loss:.4f}')
        print(f'Test Accuracy: {test_acc:.4f}')
        
        return test_loss, test_acc
    
    def predict(self, x_test):
        """Make predictions on test set"""
        predictions = self.model.predict(x_test)
        prediction_classes = np.argmax(predictions, axis=1)
        
        return predictions, prediction_classes
