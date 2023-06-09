import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn import metrics
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from model import Simple_ConvNet
import matplotlib.pyplot as plt
import os


def main():
    model_path = './saved_models/Intel_image_Classification/weights/best.pt'
    output_path = './saved_models/'
    image_size = 224
    batch_size = 64
    num_workers = 4

    test_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    test_path = './datasets/Intel_Image_Classification/valid/'

    test_data = datasets.ImageFolder(
            os.path.join(test_path),
            transform=test_transform
        )

    class_names = test_data.classes

    test_loader = DataLoader(
        test_data,
        batch_size=batch_size,
        num_workers=num_workers,
        # prefetch_factor=8,
        # pin_memory=True,
        # persistent_workers=True,
        shuffle=True
    )

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = Simple_ConvNet(3, image_size, len(class_names)).to(device)

    # load model
    model.load_state_dict(torch.load(model_path))
    model.eval()

    test_correct = []
    tst_corr = 0
    test_target = torch.tensor([]).to(device)
    test_pred = torch.tensor([]).to(device)

    with torch.no_grad():
        for X_test, y_test in test_loader:
            X_test = X_test.to(device)
            y_test = y_test.to(device)

            y_pred = model(X_test)

            predicted = torch.max(y_pred.data, 1)[1]
            tst_corr += (predicted == y_test).sum()

            test_target = torch.concat((test_target, y_test))
            test_pred = torch.concat((test_pred, predicted))

    test_correct.append(tst_corr)

    # getting the metrics (accuracy, f1-score, etc.)
    test_results = metrics.classification_report(
        test_target.cpu(), test_pred.cpu(), digits=3)

    print(test_results)

    # plot confusion matrix
    con_mat = confusion_matrix(test_target.to('cpu'), test_pred.to('cpu'))
    cm_display = ConfusionMatrixDisplay(
        confusion_matrix=con_mat,
        display_labels=class_names
    )
    cm_display.plot()
    plt.savefig('confusion_matrix.png')


if __name__ == '__main__':
    main()
