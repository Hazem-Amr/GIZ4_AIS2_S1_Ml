from data.mnist_dataset import MNISTDataset
from model.neural_network import NeuralNetwork
from training.trainer import Trainer
from evaluation.evaluator import Evaluator
from utils.saver import save_results


class TrainingPipeline:

    def __init__(self, config):
        self.config = config

    def run(self):


        dataset = MNISTDataset()
        x_train, y_train, x_test, y_test = dataset.load()


        model_builder = NeuralNetwork()
        model = model_builder.build(self.config)


        trainer = Trainer()
        trainer.train(model, x_train, y_train, self.config)


        evaluator = Evaluator()
        metrics = evaluator.evaluate(model, x_test, y_test)


        save_results(self.config, metrics)

        print("Final Metrics:", metrics)