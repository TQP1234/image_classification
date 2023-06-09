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

It will be a custom Convolutional Neural Network built for experimentational. Model architeture is written at model.py where it can be edited and imported to train.py, test.py, and predict.py.

## Training

Loss function: Cross Entropy Loss</br>
Optimizer: Adam</br>
Epochs: 100</br>
Learning rate: 0.001</br>
Image size = 224 x 224</br>

The above parameters are used for training. Checkpoints are being saved every epoch, with the best checkpoint saved as best.pt. F1 macro average score (where all classes are treated equally) is used to determine the best checkpoint.

### Training Loss vs Validation Loss

![loss](https://github.com/TQP1234/image_classification/assets/75831732/b832906f-feaa-4bb2-b7f8-a720c099a6d2)

### Training Accuracy vs Validation Accuracy

![accuracy](https://github.com/TQP1234/image_classification/assets/75831732/cc3bda6d-ed8d-49b4-b31e-159ca967f063)

### Best checkpoint

<img width="386" alt="best_checkpoint" src="https://github.com/TQP1234/image_classification/assets/75831732/9c04ea2f-8c0e-43c2-8365-fd643b6b7740">

Validation accuracy is slightly below 90%. Not too bad.

## Usage

### Training

Use the following command. Loss and accuracy graph will be updated and stored at the root folder.

``` shell
python train.py --image_size 224 --batch_size 64 --num_workers 4 --image_path ./datasets/ --output_path ./saved_models/
```

Table of parameters:

| Parameter | Function | Required? | Example input | Default Value |
| :-- | :-: | :-: | :-: | :-: |
| image_size | Set the image input size | No | Integer value (eg. 224) | 224 |
| batch_size | Number of images in a batch | No | Integer value (eg. 64) | 64 |
| num_workers | Number of sub-processes to use for data loading | No | Integer value (eg. 4) | 4 |
| image_path | Path to the dataset | Yes | ./dataset | NIL |
| output_path | Set the path where the weights are saved | No | ./saved_models/ | ./saved_models/ |

### Testing

Use the following command. Confusion matrix will be stored at the root folder.

``` shell
python test.py --image_size 224 --batch_size 64 --num_workers 4 --weights ./saved_models/Intel_Image_Classification/weights/best.pt --image_path ./datasets/valid/
```

Table of parameters:

| Parameter | Function | Required? | Example input | Default Value |
| :-- | :-: | :-: | :-: | :-: |
| image_size | Set the image input size | No | Integer value (eg. 224) | 224 |
| batch_size | Number of images in a batch | No | Integer value (eg. 64) | 64 |
| num_workers | Number of sub-processes to use for data loading | No | Integer value (eg. 4) | 4 |
| image_path | Path to the dataset | Yes | ./dataset | NIL |
| weight | Path to the saved weight | Yes | ./saved_models/weights/ | NIL |