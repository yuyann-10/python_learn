import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1,20,5)
        self.conv2 = nn.Conv2d(20, 20, 5)
      
    def forward(self, s):#前向传播 
        x = F.relu(self.conv1(x))#第一层卷积+激活
        return F.relu(self.conv2(x))#第二层卷积+激活
    

class Tudui(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, input):
        output = input + 1 
        return output
    
tudui = Tudui()
x = torch.tensor(1.0)
y = tudui(x)
print(y)  # 输出结果应为 tensor(2.)