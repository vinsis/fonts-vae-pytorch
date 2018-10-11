import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

root = '/Users/sisovina/Documents/datasets/fonts_kaggle/images/latin_jpg/'

image_transform = transforms.Compose([
    transforms.Resize(32),
    transforms.ToTensor()
])

font_data = ImageFolder(root, transform=image_transform)
loader = DataLoader(font_data, batch_size=16, shuffle=True)

if __name__ == '__main__':
    for images, _ in loader:
        print(images.type())
        print(images.min(), images.max())
        print(images.size())
        break
