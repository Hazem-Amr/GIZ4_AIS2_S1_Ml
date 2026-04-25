from .config import Config
from .mnist_dataset import MNISTDataset
from .neural_network import NeuralNetwork
from .trainer import Trainer
from .evaluator import Evaluator

class TrainingPipeline:
    def __init__(self):
        self.config = Config()
        self.dataset = None
        self.model = None
        self.trainer = None
        self.evaluator = None
    
    def run(self):
        """Execute the full training pipeline"""
        print("Starting Training Pipeline...")
        
        # Step 1: Load data
        print("\n[1/5] Loading MNIST Dataset...")
        self.dataset = MNISTDataset(self.config).load()
        
        # Step 2: Build model
        print("[2/5] Building Neural Network...")
        nn = NeuralNetwork(self.config)
        self.model = nn.build()
        self.model.summary()
        
        # Step 3: Train model
        print("[3/5] Training Model...")
        self.trainer = Trainer(self.model, self.config)
        self.trainer.train(self.dataset.x_train, self.dataset.y_train)
        
        # Step 4: Evaluate model
        print("\n[4/5] Evaluating Model on Test Set...")
        self.evaluator = Evaluator(self.model)
        test_loss, test_acc = self.evaluator.evaluate(self.dataset.x_test, self.dataset.y_test)
        
        # Step 5: Make predictions
        print("[5/5] Making Predictions...")
        predictions, prediction_classes = self.evaluator.predict(self.dataset.x_test)
        
        print("\nTraining Pipeline Complete!")
        
        return {
            'model': self.model,
            'history': self.trainer.history,
            'test_loss': test_loss,
            'test_acc': test_acc,
            'predictions': predictions,
            'prediction_classes': prediction_classes
        }
