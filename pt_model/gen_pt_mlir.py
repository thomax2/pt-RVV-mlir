import torch
import torch.nn as nn
import torch_mlir
from torch_mlir import fx
from torch_mlir.compiler_utils import (
    OutputType,
    run_pipeline_with_repro_report
)

# --- 模型定义 ---
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 2, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # 8x8 -> Conv (8x8) -> Pool (4x4)
        self.fc = nn.Linear(2 * 4 * 4, 5)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

def save_model_with_metadata(model, filename, example_input=None):
    """保存模型及其元数据以便后续恢复"""
    save_dict = {
        'model_state_dict': model.state_dict(),
        'model_class': 'SimpleCNN',
        'model_definition': '''
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
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
''',
    }
    
    if example_input is not None:
        save_dict['example_input_shape'] = example_input.shape
        save_dict['example_input_dtype'] = str(example_input.dtype)
    
    torch.save(save_dict, filename)
    print(f"✅ 模型和元数据已保存到: {filename}")

def load_model_from_pt(pt_filename):
    """从PT文件加载模型和权重"""
    # 动态创建模型类
    import torch.nn as nn
    
    # 定义模型类（与保存的相同）
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
            x = torch.flatten(x, 1)
            x = self.fc(x)
            return x
    
    # 加载保存的数据
    checkpoint = torch.load(pt_filename, map_location='cpu')
    
    # 创建模型实例并加载权重
    model = SimpleCNN()
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"✅ 模型已从 {pt_filename} 加载")
    print(f"   模型类: {checkpoint.get('model_class', 'Unknown')}")
    
    if 'example_input_shape' in checkpoint:
        print(f"   示例输入形状: {checkpoint['example_input_shape']}")
    
    return model

def main():
    # 设置随机种子以保证可重现性
    torch.manual_seed(42)
    
    # 输出文件名
    pt_filename = "simple_cnn_model_new.pt"
    mlir_filename = "simple_cnn_tosa_new.mlir"
    
    # --- 步骤1: 创建模型并训练/初始化权重 ---
    print("--- 步骤1: 创建模型 ---")
    model = SimpleCNN()
    
    # 如果你想使用预训练权重，可以在这里加载：
    # model.load_state_dict(torch.load("your_pretrained_weights.pt"))
    
    # 或者为了演示，我们可以保存当前的随机权重
    print(f"模型结构: {model}")
    
    # --- 步骤2: 保存PT文件（包含权重和元数据）---
    print(f"\n--- 步骤2: 保存PyTorch模型到PT文件 ---")
    example_input = torch.randn(1, 1, 8, 8)
    save_model_with_metadata(model, pt_filename, example_input)
    
    # 验证保存的模型
    print("验证保存的PT文件内容:")
    checkpoint = torch.load(pt_filename, map_location='cpu')
    print(f"  - 包含的键: {list(checkpoint.keys())}")
    print(f"  - 权重形状:")
    for key, value in checkpoint['model_state_dict'].items():
        print(f"    {key}: {value.shape}")
    
    # --- 步骤3: 转换为MLIR ---
    print(f"\n--- 步骤3: 转换为TOSA MLIR ---")
    
    # 确保模型在推理模式
    model.eval()
    
    # 使用相同的示例输入
    with torch.no_grad():
        # 导出到Torch MLIR
        torch_module = fx.export_and_import(
            model,
            example_input,
            output_type=OutputType.TORCH,
        )
    
    # 转换到TOSA
    run_pipeline_with_repro_report(
        torch_module,
        "builtin.module(torch-backend-to-tosa-backend-pipeline)",
        "Converting to TOSA"
    )
    
    # --- 步骤4: 保存MLIR文件 ---
    print(f"\n--- 步骤4: 保存MLIR到文件 ---")
    try:
        with open(mlir_filename, "w") as f:
            f.write(str(torch_module))
        print(f"✅ TOSA MLIR 已保存到: {mlir_filename}")
    except Exception as e:
        print(f"❌ 保存MLIR文件时发生错误: {e}")
    
if __name__ == "__main__":
    main()