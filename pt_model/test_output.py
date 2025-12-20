import torch

# 1. 加载模型权重
checkpoint = torch.load("simple_cnn_model_new.pt", map_location='cpu')

# 2. 创建模型结构（必须和保存时完全一样）
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 2, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc = nn.Linear(2 * 4 * 4, 5)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)  # 展平
        x = self.fc(x)
        return x

# 3. 创建模型并加载权重
model = SimpleCNN()
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()  # 设置为评估模式

# 4. 创建全1 tensor（1张图，1个通道，8x8大小）
ones_tensor = torch.ones(1, 1, 8, 8)

# 5. 推理
with torch.no_grad():
    output = model(ones_tensor)

# 6. 打印结果
print("输入形状:", ones_tensor.shape)
print("输出:")
print(output)
print("\n具体数值:")
for i, value in enumerate(output[0]):
    print(f"输出[{i}] = {value.item():.6f}")