# DCGAN on MNIST

This repository contains the implementation of a Generative Adversarial Network (GAN) trained on the MNIST dataset to generate handwritten digits. The implementation leverages PyTorch and includes scripts for both training the GAN and visualizing results using TensorBoard.

## Project Overview

This GAN consists of two components:

1. **Discriminator**: A convolutional neural network (CNN) that classifies whether an input image is real or fake.
2. **Generator**: A CNN that generates images resembling the real MNIST digits.

The training process involves a min-max optimization where the generator tries to fool the discriminator, and the discriminator aims to distinguish between real and fake images.

---

## Repository Contents

### Files
- **`model.py`**: Defines the architecture for the Generator and Discriminator models along with a utility function for initializing weights.
- **`training.py`**: Contains the training loop for the GAN, along with data preprocessing and logging using TensorBoard.
- **`logs`**: A folder containing pre-trained results logged during training. These can be visualized in TensorBoard.

---

## Prerequisites

### Install Required Libraries
Make sure to install the required Python packages:
```bash
pip install torch torchvision tensorboard
```

### Dataset
The code automatically downloads the MNIST dataset using PyTorch's `datasets.MNIST` module.

---

## How to Use

### Training the GAN
1. Run the `training.py` script to train the GAN.
   ```bash
   python training.py
   ```
   - **Default Parameters:**
     - `num_epochs`: 35
     - `batch_size`: 128
     - `z_dim`: 100 (size of the random noise input to the generator)
     - `image_channels`: 1
     - `image_size`: 64x64
     - `LEARNING_RATE`: 2e-4

2. Training progress, including the discriminator and generator losses, will be printed in the console.

### Visualizing Results
1. **TensorBoard Visualization:**
   - Use the pre-trained results stored in the `logs` folder to visualize images in TensorBoard.
   - Start TensorBoard in the project directory:
     ```bash
     %reload_ext tensorboard
     LOG_DIR = "logs"
     %tensorboard --logdir {LOG_DIR}
     ```
   - Open the provided URL in your browser to view real and generated images.

### Pre-Trained Logs
The training diverged after 35 epochs, so the code has been updated to limit training to 35 epochs. The pre-trained logs available in the `logs` folder demonstrate the results up to 50 epochs for analysis.

---

## Results
- The generator successfully produces high-quality MNIST-like digits for the first 35 epochs.
- After 35 epochs, the training diverges, indicating the necessity of early stopping to maintain stability.

---

## Key Features
- **Weight Initialization**: All convolutional layers use a normal initialization with a mean of 0 and a standard deviation of 0.02.
- **Logging with TensorBoard**: Real and fake images, along with loss metrics, are logged during training for easy monitoring.
- **Pre-trained Logs**: Includes logs for analyzing results without retraining.

---

## Citation
If you use this repository, please cite or mention this project.

---

## Acknowledgements
- [PyTorch Documentation](https://pytorch.org/docs/)
- [MNIST Dataset](http://yann.lecun.com/exdb/mnist/)

Feel free to explore and modify the code as needed. Contributions are welcome!

