# image_classification

The goal is to build a custom Convolutional Neural Network for image classification tasks. Pytorch is used for building the model architecture. The train, test and predict processes shall also be streamlined for ease of usage.

## Dataset

Dataset is downloaded from https://www.kaggle.com/datasets/puneet6060/intel-image-classification.

The Intel Image Classification dataset has 6 classes (buildings, forest, glacier, mountain, sea, street) and are indexed accordingly from 0 to 5.

Train set contains 14034 images (buildings - 2191, forest - 2271, glacier - 2404, mountain - 2512, sea - 2274, street - 2382).

Validation set contains 3000 images (buildings - 437, forest - 474, glacier - 553, mountain - 525, sea - 510, street - 501).

All classes are pretty balanced.

Test set contains 7301 images. Unfortunately, the test set is not sorted. But we will use the machine algorithm that will be trained later on to sort the test set (and evaluate as well).

## Folder Structure

Dataset is to be stored stored in the following structure (Example shown below). Images are stored in their corresponding sub-folders (class).

    .
    └── train
        └── buildings
            └── image1.jpg
            └── image2.jpg
            └── image3.jpg
        └── forest
            └── image1.jpg
            └── image2.jpg
            └── image3.jpg
        └── glacier
            └── image1.jpg
            └── image2.jpg
            └── image3.jpg
    └── test
        └── buildings
            └── image1.jpg
            └── image2.jpg
            └── image3.jpg
        └── forest
            └── image1.jpg
            └── image2.jpg
            └── image3.jpg
        └── glacier
            └── image1.jpg
            └── image2.jpg
            └── image3.jpg
    └── valid
        └── buildings
            └── image1.jpg
            └── image2.jpg
            └── image3.jpg
        └── forest
            └── image1.jpg
            └── image2.jpg
            └── image3.jpg
        └── glacier
            └── image1.jpg
            └── image2.jpg
            └── image3.jpg

## Model Architecture

It will be a custom Convolutional Neural Network built for experimentational.

## Training

Loss function: Cross Entropy Loss
Optimizer: Adam
Epochs: 100
Learning rate: 0.001

The above parameters are used for training. Checkpoints are being saved every epoch, with the best checkpoint saved as best.pt. F1 macro average score (where all classes are treated equally) is used to determine the best checkpoint.

### Training Loss vs Validation Loss



### Training Accuracy vs Validation Accuracy



### Best checkpoint



Validation accuracy is slightly below 90%. Not too bad.