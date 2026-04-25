# MNIST Neural Network Pipeline

A complete deep learning pipeline for training and evaluating a neural network on the MNIST dataset. This project includes both command-line and interactive GUI modes with digit recognition capabilities.

## Project Overview

This pipeline implements a full machine learning workflow for MNIST digit classification:
- **Data Loading**: Automatically downloads and preprocesses MNIST dataset
- **Model Training**: Trains a customizable neural network with adjustable hyperparameters
- **Model Evaluation**: Evaluates performance on test set
- **Predictions**: Makes predictions on new data
- **Interactive GUI**: Draw digits and get real-time predictions from the trained model

## Pipeline Architecture

```
TrainingPipeline (main.py)
    ├── Step 1: Load MNIST Dataset (MNISTDataset)
    ├── Step 2: Build Neural Network (NeuralNetwork)
    ├── Step 3: Train Model (Trainer)
    ├── Step 4: Evaluate on Test Set (Evaluator)
    └── Step 5: Make Predictions (Evaluator)
```

## File Structure

```
Pipeline/
├── main.py                  # Entry point (CLI & GUI modes)
├── gui.py                   # Interactive GUI for drawing predictions
├── mnist_model.h5          # Pre-trained model (generated on first run)
├── README.md               # This file
└── src/
    ├── config.py           # Configuration parameters
    ├── mnist_dataset.py    # MNIST data loading and preprocessing
    ├── neural_network.py   # Model architecture
    ├── trainer.py          # Training logic
    ├── evaluator.py        # Evaluation and prediction
    └── training_pipeline.py # Main pipeline orchestration
```

##  Configuration

Edit `src/config.py` to customize training parameters:

```python
class Config:
    EPOCHS = 10              # Number of training epochs
    BATCH_SIZE = 64          # Batch size for training
    LR = 0.001               # Learning rate (adam optimizer)
    INPUT_SIZE = 784         # Flattened image size (28x28)
    NUM_CLASSES = 10         # Number of digit classes (0-9)
    VALIDATION_SPLIT = 0.2   # Validation/training split ratio
```

## Requirements

```
tensorflow>=2.10.0
keras>=2.10.0
numpy
pillow
```

Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Option 1: CLI Mode (Training Only)
Run the training pipeline from command line:

```bash
python -m main
# or
python main.py
```

**Output:**
```
Starting Training Pipeline...

[1/5] Loading MNIST Dataset...
[2/5] Building Neural Network...
[3/5] Training Model...
[4/5] Evaluating Model on Test Set...
[5/5] Making Predictions...

Training Pipeline Complete!
```

### Option 2: Interactive GUI (Drawing Predictions)
Launch the interactive drawing GUI:

```bash
python gui.py
# or
python -m main --gui
# or
python -m main -g
```

**Features:**
- Draw digits on canvas with mouse
- Real-time prediction
- Confidence score display
- Auto-trains model if not available
- Saves trained model for reuse

## Neural Network Architecture

```
Input Layer: 784 neurons (28x28 flattened image)
    ↓
Dense Layer 1: 500 neurons (sigmoid activation)
    ↓
Dense Layer 2: 100 neurons (sigmoid activation)
    ↓
Output Layer: 10 neurons (softmax activation)
```

**Model Specs:**
- **Optimizer**: Adam
- **Loss Function**: Categorical Cross-entropy
- **Metrics**: Accuracy
- **Parameters**: ~465,000

## Data Processing

The pipeline preprocesses MNIST data as follows:

1. **Normalization**: Pixel values scaled to [0, 1] range (0-255 → 0-1)
2. **Flattening**: 28×28 images flattened to 784-dimensional vectors
3. **One-hot Encoding**: Labels converted to one-hot vectors (10 classes)
4. **Validation Split**: 80% training, 20% validation

## GUI Usage Guide

### Drawing Prediction Interface

1. **Launch GUI**: `python gui.py`
2. **Draw**: Click and drag on the white canvas to draw a digit
3. **Predict**: Click the "Predict" button to classify your drawing
4. **Results**: 
   - Predicted digit displayed in large font
   - Confidence score shown as percentage and progress bar
5. **Clear**: Click "Clear" button to reset canvas

### Model Handling

- **First Run**: If `mnist_model.h5` doesn't exist, the GUI will automatically train a new model (may take a few minutes)
- **Subsequent Runs**: Pre-trained model loads instantly
- **Model Persistence**: Trained model saved as `mnist_model.h5` for future use

## Expected Performance

With default configuration (10 epochs):
- **Test Accuracy**: ~95-97%
- **Test Loss**: ~0.12-0.15
- **Training Time**: ~2-5 minutes (depends on hardware)

## Class Overview

### `TrainingPipeline`
Orchestrates the complete ML workflow:
- Loads dataset
- Builds model
- Trains model
- Evaluates performance
- Makes predictions

### `MNISTDataset`
Handles data loading and preprocessing:
- Downloads MNIST dataset
- Normalizes pixel values
- Flattens images
- One-hot encodes labels

### `NeuralNetwork`
Defines model architecture:
- 3-layer dense network
- Sigmoid and softmax activations

### `Trainer`
Manages model training:
- Compiles model
- Trains with specified hyperparameters
- Tracks training history

### `Evaluator`
Evaluates and makes predictions:
- Computes test loss and accuracy
- Makes predictions on new data
- Extracts predicted classes

### `DrawingPredictionGUI`
Provides interactive interface:
- Canvas for drawing digits
- Real-time predictions
- Confidence visualization
- Automatic model handling

## Development

### Modifying Architecture
Edit `src/neural_network.py` to change the model:

```python
def build(self):
    self.model = tf.keras.Sequential([
        tf.keras.layers.Dense(500, activation='relu', input_shape=(784,)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(100, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    return self.model
```

### Adjusting Training
Modify `src/config.py` for different training parameters:

```python
EPOCHS = 20          # Longer training
BATCH_SIZE = 32      # Smaller batches
LR = 0.0005          # Lower learning rate
```

## Output Files

- **`mnist_model.h5`**: Trained model (created after first training)
- **Training logs**: Printed to console during execution
- **Predictions**: Returned from `pipeline.run()` method

