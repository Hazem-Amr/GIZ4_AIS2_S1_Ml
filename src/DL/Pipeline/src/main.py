from config.config import Config
from pipeline.training_pipeline import TrainingPipeline


def main():
    config = Config()
    pipeline = TrainingPipeline(config)
    pipeline.run()


if __name__ == "__main__":
    main()