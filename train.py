import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn import metrics
from model import Simple_ConvNet
import matplotlib.pyplot as plt
import os
import cv2
import random
import time


def get_dataset(path, train_ratio):
    label_details = {
        '0': 0, '1': 1, '2': 2, '3': 3, '4': 4,
        '5': 5, '6': 6, '7': 7, '8': 8, '9': 9,
        'A': 10, 'B': 11, 'C': 12, 'D': 13, 'E': 14,
        'F': 15, 'G': 16, 'H': 17, 'I': 18, 'J': 19,
        'K': 20, 'L': 21, 'M': 22, 'N': 23, 'O': 24,
        'P': 25, 'Q': 26, 'R': 27, 'S': 28, 'T': 29,
        'U': 30, 'V': 31, 'W': 32, 'X': 33, 'Y': 34,
        'Z': 35
    }

    labels = os.listdir(path)

    train_set = []
    valid_set = []

    random.seed(29)

    for label in labels:
        image_list = os.listdir(path + label)
        print(f'label: {label.upper()}, number_of_images: {len(image_list)}')

        random.shuffle(image_list)

        for n, i in enumerate(image_list):
            if n < int(len(image_list) * train_ratio):
                train_set.append(
                    (cv2.imread(path + label + '/' + i),
                     label_details[label.upper()])
                )
            else:
                valid_set.append(
                    (cv2.imread(path + label + '/' + i),
                     label_details[label.upper()])
                )

    return train_set, valid_set


def split_xy(data_list):
    x = []
    y = []

    for data in data_list:
        x.append(data[0])
        y.append(data[1])

    return x, y


def split_into_batches(X, y, batch_size):
    loader = []

    temp_index = 0
    for i in range(len(X) + 1):
        if i != 0:
            if i % batch_size == 0:
                loader.append((X[temp_index:i], y[temp_index:i]))
                temp_index = i

    return loader


def shuffle_data(data_list):
    random.seed(9)
    random.shuffle(data_list)

    return data_list


def main():
    image_size = 224
    batch_size = 64
    num_workers = 4
    image_path = './datasets/Intel_Image_Classification/'
    output_path = './saved_models/'

    train_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    valid_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])

    train_path = image_path + 'train/'
    valid_path = image_path + 'valid/'

    train_data = datasets.ImageFolder(
            os.path.join(train_path),
            transform=train_transform
        )

    valid_data = datasets.ImageFolder(
            os.path.join(valid_path),
            transform=valid_transform
        )

    num_batch = int(len(train_data) / batch_size)
    class_names = train_data.classes

    torch.manual_seed(29)
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        num_workers=num_workers,
        # prefetch_factor=8,
        # pin_memory=True,
        # persistent_workers=True,
        shuffle=True
    )

    torch.manual_seed(29)
    valid_loader = DataLoader(
        valid_data,
        batch_size=batch_size,
        num_workers=num_workers,
        # prefetch_factor=8,
        # pin_memory=True,
        # persistent_workers=True,
        shuffle=False
    )

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    torch.manual_seed(29)
    model = Simple_ConvNet(3, image_size, len(class_names)).to(device)

    print(f'{model}\n\n'
          f'number_of_training_images: {len(train_data)}\n'
          f'number_of_validation_images: {len(valid_data)}\n'
          f'number_of_classes: {len(class_names)}\n'
          f'classes: {class_names}')

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters()
                           if p.requires_grad)

    print(f'\nloss_function: {criterion.__class__.__name__}\n'
          f'optimizer: {optimizer.__class__.__name__}\n'
          f'learning_rate: {optimizer.param_groups[0]["lr"]}\n'
          f'total_params: {total_params}\n'
          f'trainable_params: {trainable_params}\n')

    start_time = time.time()

    epochs = 100
    train_losses = []
    valid_losses = []
    train_correct = []
    valid_correct = []
    y_valid_initialized = False
    best_f1_score = 0

    for i in range(epochs):
        trn_corr = 0
        val_corr = 0

        print('=====Training=====')

        for b, (X_train, y_train) in enumerate(train_loader):
            b += 1

            X_train = X_train.to(device)
            y_train = y_train.to(device)

            y_pred = model(X_train)
            loss = criterion(y_pred, y_train)

            predicted = torch.max(y_pred.data, 1)[1]
            batch_corr = (predicted == y_train).sum()
            trn_corr += batch_corr

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if b != num_batch + 1:
                progress = str(int(batch_size*(b)))
            else:
                progress = str(len(train_data))

            training_loss = f'{loss.item():.5f}'

            print('epoch: ' + str(i) + ', progress: ' + progress +
                  '/' + str(len(train_data)) + ', training_loss: ' +
                  training_loss, end='\r')

        print()

        train_losses.append(loss.item())
        train_correct.append(trn_corr)

        training_accuracy = 100 * trn_corr.item() / len(train_data)

        if y_valid_initialized is False:
            val_target = torch.tensor([]).to(device)

        val_pred = torch.tensor([]).to(device)

        # performing validation with the validation data
        model.eval()
        with torch.no_grad():
            for X_valid, y_valid in valid_loader:
                X_valid = X_valid.to(device)
                y_valid = y_valid.to(device)

                y_val = model(X_valid)

                predicted = torch.max(y_val.data, 1)[1]
                val_corr += (predicted == y_valid).sum()

                if y_valid_initialized is False:
                    val_target = torch.concat((val_target, y_valid))

                val_pred = torch.concat((val_pred, predicted))

        if y_valid_initialized is False:
            y_valid_initialized = True

        loss = criterion(y_val, y_valid)
        valid_losses.append(loss.item())
        valid_correct.append(val_corr)

        valid_loss = f'{loss.item():.5f}'

        # getting the metrics (accuracy, f1-score, etc.)
        valid_accuracy = 100 * val_corr.item() / len(valid_data)
        val_results = metrics.classification_report(
            val_target.cpu(), val_pred.cpu(), digits=3)
        val_results_dict = metrics.classification_report(
            val_target.cpu(), val_pred.cpu(), digits=3, output_dict=True)
        macro_avg = val_results_dict['macro avg']['f1-score']

        # store checkpoint details
        checkpoint = (f'epoch: {i}\n'
                      f'training_loss: {training_loss}\n'
                      f'training_accuracy: {training_accuracy:.2f} %\n'
                      f'validation_loss: {valid_loss}\n'
                      f'validation_accuracy: {valid_accuracy:.2f} %\n'
                      f'\n{val_results}\n')

        print(f'\n=====Validation=====\n{checkpoint}\n')

        torch.save(model.state_dict(),
                   output_path + str(i).zfill(len(str(epochs))) + '.pt')

        with open(output_path + str(i).zfill(len(str(epochs))) + '.txt',
                  'w') as f:
            f.write(checkpoint)

        # set f1 macro avg score as the criteria for the best score
        if macro_avg > best_f1_score:
            torch.save(model.state_dict(), output_path + 'best.pt')
            best_f1_score = macro_avg
            best_checkpoint = checkpoint

            with open(output_path + 'best.txt', 'w') as f:
                f.write(best_checkpoint)

        model.train()

        # plot training loss and validation loss
        plt.figure()
        plt.grid()
        plt.plot(train_losses, label='Training Loss')
        plt.plot(valid_losses, label='Validation Loss')
        plt.legend()
        plt.title('Loss')
        plt.savefig('./loss.png')
        plt.close()

        # plot training accuracy and validation accuracy
        train_acc = [t.item()/len(train_data) for t in train_correct]
        val_acc = [t.item()/len(valid_data) for t in valid_correct]
        plt.figure()
        plt.grid()
        plt.plot(train_acc, label='train_acc')
        plt.plot(val_acc, label='val_acc')
        plt.legend()
        plt.title('Accuracy')
        plt.savefig('./accuracy.png')
        plt.close()

    current_time = time.time()
    total = current_time - start_time
    print(f'\nTraining took {total/60} minutes\n'
            f'\n=====Best Checkpoint=====\n{best_checkpoint}\n')


if __name__ == '__main__':
    main()
