import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn import metrics
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import argparse
import os


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--weight', type=str, required=True)
    parser.add_argument('--image_size', type=int, required=False)
    parser.add_argument('--batch_size', type=int, required=False)
    parser.add_argument('--num_workers', type=int, required=False)
    parser.add_argument('--image_path', type=str, required=True)

    args = parser.parse_args()

    if args.image_size is not None:
        image_size = args.image_size
    else:
        image_size = 224

    if args.batch_size is not None:
        batch_size = args.batch_size
    else:
        batch_size = 64

    if args.num_workers is not None:
        num_workers = args.num_workers
    else:
        num_workers = 4

    test_path = args.image_path
    saved_weight = args.weight

    test_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

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
        shuffle=False
    )

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # load model
    model = torch.jit.load(saved_weight).to(device)
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
