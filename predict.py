import torch
from torchvision import transforms
from PIL import Image
import argparse


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--weight', type=str, required=True)
    parser.add_argument('--image_size', type=int, required=False)
    parser.add_argument('--image', type=str, required=True)

    args = parser.parse_args()

    if args.image_size is not None:
        image_size = args.image_size
    else:
        image_size = 224

    image = args.image
    saved_weight = args.weight

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # load model
    model = torch.jit.load(saved_weight).to(device)

    # eval mode
    model.eval()

    image = Image.open(image)

    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    image = transform(image)
    image = image.unsqueeze(0)
    image = image.to(device)

    pred = model(image)
    pred = torch.exp(pred)
    pred = torch.argmax(pred)

    print(f'\nprediction (class index): {pred}\n')


if __name__ == '__main__':
    main()
