# Deep Learning Pipeline

This directory contains the source code for a modular Deep Learning pipeline designed for training and evaluating models. It is structured to separate concerns, making the codebase maintainable and scalable.

## Directory Structure

The source code is organized into the following modules:

- **`config/`**: Contains configuration classes and files (e.g., hyperparameter settings, paths).
- **`data/`**: Responsible for loading, preprocessing, and managing datasets (e.g., `MNISTDataset`).
- **`model/`**: Defines the neural network architectures.
- **`training/`**: Contains the training loop logic and the `Trainer` class.
- **`evaluation/`**: Handles evaluating the trained model and computing metrics via the `Evaluator` class.
- **`pipeline/`**: Orchestrates the entire process. The `TrainingPipeline` integrates data, model, training, and evaluation.
- **`utils/`**: Utility functions such as saving results, logging, and other helper methods.
- **`logs/`**: Directory where training logs and output metrics are stored.

## Entry Point

The entry point for executing the pipeline is `main.py`.

### How to Run

To start the training pipeline, simply execute `main.py`:

```bash
python main.py
```

This script will:
1. Initialize the `Config`.
2. Instantiate the `TrainingPipeline` with the configuration.
3. Run the complete pipeline (data loading -> model building -> training -> evaluation -> saving results).
