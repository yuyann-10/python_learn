from torchvision import transforms

transform = transforms.ToTensor()
transform = transforms.Normalize(mean=[0.5], std=[0.5])  # 归一化到 [-1, 1]
transform = transforms.Resize((128, 128))  # 将图像调整为 128x128
transform = transforms.CenterCrop(128)  # 裁剪 128x128 的区域
transform = transforms.RandomCrop(128)
transform = transforms.RandomHorizontalFlip(p=0.5)  # 50% 概率翻转
transform = transforms.RandomRotation(degrees=30)  # 随机旋转 -30 到 +30 度
transform = transforms.ColorJitter(brightness=0.5, contrast=0.5)
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

class CustomTransform:
    def __call__(self, x):
        # 这里可以自定义任何变换逻辑
        return x * 2

transform = CustomTransform()

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 定义转换
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# 加载数据集
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)

# 使用 DataLoader
train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)

# 查看转换后的数据
for images, labels in train_loader:
    print("图像张量大小:", images.size())  # [batch_size, 1, 128, 128]
    break
