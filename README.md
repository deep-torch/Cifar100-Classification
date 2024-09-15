# CIFAR-100 Image Classification using EfficientNetV2

This project implements a custom image classifier for the CIFAR-100 dataset using an EfficientNetV2 backbone. It includes training and evaluation scripts, as well as utilities for handling checkpoints and category mappings. The model is designed to classify 100 unique classes with additional hierarchical general classification into broader categories.


## Overview

The project uses PyTorch and the CIFAR-100 dataset to classify images into 100 classes with an additional general classification task. The EfficientNetV2-S backbone is used for feature extraction, and the model has two linear heads:
- A general classifier that maps to 20 super-categories.
- A specialized classifier that maps to 100 fine-grained categories.

This dual-head structure allows the model to make both high-level and detailed predictions.


## Dependencies

This project requires the following Python libraries:

- `torch` (PyTorch)
- `torchvision`
- `numpy`
- `argparse`

To install the required packages, use the following command:

``` bash
pip install torch torchvision numpy argparse
```



## Usage

### Data Loading
The `get_dataloaders` function in `data.py` handles data preprocessing and augmentation. The CIFAR-100 dataset is automatically downloaded if not present and is augmented with transformations like random rotations, resizing, and color jittering for the training set.


### Model Architecture
The model architecture consists of:
- EfficientNet feature extractor: Pretrained on ImageNet, with the first half of the layers frozen during training.
- General classifier: A fully connected layer predicting one of the 20 superclasses.
- pecific classifier: A fully connected layer that combines features from EfficientNet and the general classifier to predict one of the 100 specific classes.

### Checkpointing
The `utils.py` file contains functions to save and load model checkpoints. This allows training to be resumed from any saved epoch.



### Catrgory Mapping
The dataset is divided into 20 super-categories, with each super-category containing 5 classes. The `get_category_mapping` function in `utils.py` creates a mapping from class indices to super-categories.

Example super-categories:
- Aquatic mammals: `['beaver', 'dolphin', 'otter', 'seal', 'whale']`
- Fish: `['aquarium_fish', 'flatfish', 'ray', 'shark', 'trout']`
- Flowers: `['orchid', 'poppy', 'rose', 'sunflower', 'tulip']`

This mapping is utilized during training and evaluation to compute the loss and accuracy for both general and fine-grained categories.



## File Structure

- `data.py`: Handles data preprocessing and loading using `torchvision.datasets.CIFAR100`.
- `model.py`: Contains the `ClassificationModel` class, which defines the architecture.
- `train.py`: Implements the training and evaluation loops. Includes checkpoint saving and loading, as well as accuracy tracking.
- `utils.py`: Includes utility functions for checkpointing and category mapping.
- `usage_example.ipynb`: A notebook that demonstrates how to set up, train, and evaluate the model.


## Training and Evaluation

To train the model, the script `train.py` accepts the following hyperparameters:

- `--epochs`: Number of training epochs (default: 10)
- `--batch_size`: Batch size for training (default: 128)
- `--lr_conv`: Learning rate for the convolutional layers (default: 5e-4)
- `--lr_fc`: Learning rate for the fully connected layers (default: 1e-3)
- `--weight_decay`: Weight decay for regularization (default: 1e-4)
- `--checkpoint_path`: Path to load or save model checkpoints



## Notes
A Jupyter Notebook (`usage_example.ipynb`) is provided to demonstrate how to use the project. It includes example code for:

- Loading and preprocessing the CIFAR-100 dataset.
- Training the model.
- Evaluating model performance.

Make sure to explore the notebook if you are unfamiliar with the training process or want a guided example.






