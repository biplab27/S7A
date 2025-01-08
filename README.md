# MNIST Digit Classification with PyTorch

This project implements a Convolutional Neural Network (CNN) to classify handwritten digits from the MNIST dataset using PyTorch.

## Project Structure 
```
project/
├── data/ # MNIST dataset storage
├── model_checkpoints/ # Saved model checkpoints
├── download_mnist.py # Script to download MNIST dataset
├── train_test_mnist.py # Main training script
├── requirements.txt # Project dependencies
└── README.md # This file
```

## Model Architecture
The CNN architecture consists of:
- Input Block: 1 → 32 channels
- Convolution Block 1: 32 → 64 → 128 channels
- Transition Block: 128 → 32 channels with MaxPooling
- Convolution Block 2: 32 → 64 → 128 channels
- Output Block: 128 → 10 channels with Global Average Pooling

Key features:
- Uses only 3x3 and 1x1 convolutions
- No bias terms in convolution layers
- ReLU activation throughout
- MaxPooling for dimensionality reduction

## Requirements 
```
bash
torch>=2.0.0
torchvision>=0.15.0
tqdm>=4.65.0
numpy>=1.24.0
pillow>=9.0.0
```

## Installation
1. Clone the repository:
```bash
git clone <repository-url>
cd mnist-classification
```

2. Create and activate a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Download the MNIST dataset:
```bash
python download_mnist.py
```

2. Train the model:
```bash
python train_test_mnist.py
```

The training script will:
- Train for 20 epochs
- Show progress with tqdm progress bars
- Display training loss and accuracy
- Show test accuracy after each epoch
- Save model checkpoints in `model_checkpoints/` directory

## Model Checkpoints
The training process saves two types of checkpoints:
- `mnist_model_epoch_X.pth`: Saved after each epoch
- `best_mnist_model.pth`: Updated when test accuracy improves

Each checkpoint contains:
- Model state
- Optimizer state
- Current epoch
- Training accuracy
- Test accuracy

## Training Details
- Optimizer: SGD with momentum (0.9)
- Learning rate: 0.01
- Batch size: 128 (GPU) / 64 (CPU)
- Data normalization: mean=0.1307, std=0.3081
- Random seed: 1 (for reproducibility)

## Results
The model achieves competitive accuracy on the MNIST test set. Progress can be monitored during training with:
- Per-batch training loss
- Per-epoch test accuracy
- Best model checkpoint saves

## License
[Your chosen license]

## Acknowledgments
- MNIST dataset from LeCun et al.
- PyTorch framework