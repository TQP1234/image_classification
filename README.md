# Image Classification

The goal is to build a custom Convolutional Neural Network for image classification tasks. Pytorch is used for building the model architecture. The train, test, and predict processes shall also be streamlined for ease of usage.

## Dataset

Dataset is downloaded from https://www.kaggle.com/datasets/puneet6060/intel-image-classification.

The Intel Image Classification dataset has 6 classes <b>(buildings, forest, glacier, mountain, sea, street)</b> and are <b>indexed accordingly from 0 to 5</b>.

Train set contains 14034 images <b>(buildings - 2191, forest - 2271, glacier - 2404, mountain - 2512, sea - 2274, street - 2382)</b>.

Validation set contains 3000 images <b>(buildings - 437, forest - 474, glacier - 553, mountain - 525, sea - 510, street - 501)</b>.

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

It will be a custom Convolutional Neural Network built for experimentational. Model architeture is written at model.py. Inspiration taken from https://arxiv.org/abs/1709.01507 (Squeeze-and-Excitation Networks).

![SimpleConvNetv2](https://github.com/TQP1234/image_classification/assets/75831732/fc1c3354-5eee-4d49-ae4e-9811b2da954d)

## Training

Loss function: Cross Entropy Loss</br>
Optimizer: Adam</br>
Epochs: 100</br>
Learning rate: 0.001</br>
Image size = 224 x 224</br>

The above parameters are used for training. Checkpoints are being saved every epoch, with the best checkpoint saved as best.pt. F1 macro average score <b>(where all classes are treated equally)</b> is used to determine the best checkpoint.

### Training Loss vs Validation Loss

![loss](https://github.com/TQP1234/image_classification/assets/75831732/65a9d402-a36a-4e6f-ac5d-5941feebe74c)

### Training Accuracy vs Validation Accuracy

![accuracy](https://github.com/TQP1234/image_classification/assets/75831732/efb2169c-9162-4c37-b337-36ade53c21c7)

### Best checkpoint

<img width="380" alt="best_checkpoint" src="https://github.com/TQP1234/image_classification/assets/75831732/1025cee0-4477-4dd0-97b2-4ca905afe75b">

Validation accuracy is <b>91.53%</b>. Not too bad.

## Model Evaluation

In practical cases, we will always use unseen data as the test set to evaluate the model. Since the test set is not labeled, the validation set will be used as an example in this case. Run test.py <b>(refer to the usage section on how to use it)</b> and a confusion matrix will be generated.

### Confusion Matrix

![confusion_matrix](https://github.com/TQP1234/image_classification/assets/75831732/cb4f4e42-41b2-43b7-83ec-a2363acc8bf7)

As we can see, the model performs generally well. Most errors come from the buildings/street and glacier/mountain classes. It's because those pairs of classes are somewhat correlated. Glacier and mountain looked similar; street and buildings in a single image. To further improve this classification model, we may employ image augmentation technique.

## Usage

### Training

Use the following command to train the model. Loss and accuracy graph will be updated and stored at the root folder.

``` shell
python train.py --image_size 224 --batch_size 64 --num_workers 4 --image_path ./datasets/ --output_path ./saved_models/ --epochs 100 --learning_rate 0.001 --weight ./saved_models/Intel_Image_Classification/weights/best.pt
```

Table of parameters:

| Parameter | Function | Required? | Example input | Default Value |
| :-- | :-: | :-: | :-: | :-: |
| image_size | Set the image input size | No | Integer value (eg. 224) | 224 |
| batch_size | Number of images in a batch | No | Integer value (eg. 64) | 64 |
| num_workers | Number of sub-processes to use for data loading | No | Integer value (eg. 4) | 4 |
| image_path | Path to the dataset | Yes | ./dataset | NIL |
| output_path | Set the path where the weights are saved | No | ./saved_models/ | ./saved_models/ |
| epochs | Set the number of training epochs | Yes | Integer value (eg. 100) | NIL |
| learning_rate | Set the learning rate | Yes | Float value (eg. 0.001) | NIL |
| weight | Path to the saved weight | No | ./saved_models/weights/ | NIL |

### Testing

Use the following command for model evaluation. Confusion matrix will be stored at the root folder.

``` shell
python test.py --image_size 224 --batch_size 64 --num_workers 4 --weight ./saved_models/Intel_Image_Classification/weights/best.pt --image_path ./datasets/valid/
```

Table of parameters:

| Parameter | Function | Required? | Example input | Default Value |
| :-- | :-: | :-: | :-: | :-: |
| image_size | Set the image input size | No | Integer value (eg. 224) | 224 |
| batch_size | Number of images in a batch | No | Integer value (eg. 64) | 64 |
| num_workers | Number of sub-processes to use for data loading | No | Integer value (eg. 4) | 4 |
| image_path | Path to the dataset | Yes | ./datasets/ | NIL |
| weight | Path to the saved weight | Yes | ./saved_models/weights/ | NIL |


### Making Prediction

Use the following command to make prediction. It will output the class index.

``` shell
python predict.py --image_size 224 --weight ./saved_models/Intel_Image_Classification/weights/best.pt --image ./street.jpg
```

Table of parameters:

| Parameter | Function | Required? | Example input | Default Value |
| :-- | :-: | :-: | :-: | :-: |
| image_size | Set the image input size | No | Integer value (eg. 224) | 224 |
| image | Path to the dataset | Yes | ./datasets/ | NIL |
| weight | Path to the saved weight | Yes | ./saved_models/weights/ | NIL |
