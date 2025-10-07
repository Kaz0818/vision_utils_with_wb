from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms

def make_transform(mean, std):
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(30),
        transforms.ToTensor(),
        transforms.Normalize((mean), (std),)
    ])
    val_transform = transforms.Compose([
        transforms.Resize(28),
        transforms.ToTensor(),
        transforms.Normalize((mean), (std))
    ])
    return train_transform, val_transform

def load_dataset(root, train_transform, val_transform, batch_size):
    train_dataset = datasets.MNIST(root=root, train=True, download=True, transform=train_transform)
    val_dataset = datasets.MNIST(root=root, train=False, download=True, transform=val_transform)
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size, shuffle=False)
    return train_loader, val_loader


tr_trf, val_trf = make_transform(0.5, 0.5)

print(tr_trf, val_trf)