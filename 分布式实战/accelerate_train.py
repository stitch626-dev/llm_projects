"""
Accelerate 分布式训练学习示例

这个脚本演示了如何使用 HuggingFace Accelerate 库进行分布式训练。
Accelerate 是一个简化 PyTorch 分布式训练的库，它可以自动处理：
- 多GPU训练
- 混合精度训练
- DeepSpeed 集成
- 设备管理（无需手动 .to(device)）
"""

from accelerate import Accelerator, DeepSpeedPlugin
import torch
from torch.utils.data import DataLoader, TensorDataset

class SimpleNet(torch.nn.Module):
    """
    简单的神经网络模型
    注意：使用 Accelerate 时，不需要手动将模型或数据移动到特定设备
    Accelerate 会自动处理设备管理
    """
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SimpleNet, self).__init__()
        # 定义两个全连接层
        self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # 前向传播：输入 -> ReLU激活 -> 输出
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def create_binary_classification_dataset(data_size, input_dim):
    """
    创建一个简单的二分类数据集
    任务：根据输入特征判断样本类别（0或1）
    """
    print(f"创建二分类数据集: {data_size}个样本，{input_dim}维特征...")
    
    # 生成随机特征
    X = torch.randn(data_size, input_dim)
    
    # 创建简单的分类规则：如果前两个特征的和>0，则为类别1，否则为类别0
    y = (X[:, 0] + X[:, 1] > 0).long()
    
    # 添加一些噪声让任务更真实
    noise_indices = torch.randperm(data_size)[:int(data_size * 0.1)]  # 10%噪声
    y[noise_indices] = 1 - y[noise_indices]
    
    print(f"数据集统计: 类别0有{(y==0).sum()}个样本, 类别1有{(y==1).sum()}个样本")
    return X, y

if __name__ == "__main__":
    # ====== 超参数设置 ======
    input_dim = 10      # 输入特征维度
    hidden_dim = 32     # 隐藏层维度
    output_dim = 2      # 输出维度（二分类）
    batch_size = 32     # 批次大小
    data_size = 1000    # 数据集大小（简化用于学习演示）
    num_epochs = 5      # 训练轮数（简化用于学习演示）
    learning_rate = 0.001  # 学习率
    
    # ====== 创建模拟数据集 ======
    input_data, labels = create_binary_classification_dataset(data_size, input_dim)
    
    dataset = TensorDataset(input_data, labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # ====== 初始化模型 ======
    model = SimpleNet(input_dim, hidden_dim, output_dim)
    
    # ====== 配置 Accelerate ======
    print("初始化 Accelerate...")
    
    # 检查是否可以使用 DeepSpeed（可选）
    try:
        deepspeed_plugin = DeepSpeedPlugin(
            zero_stage=2,  # ZeRO-2: 分片优化器状态和梯度
            gradient_clipping=1.0  # 梯度裁剪防止梯度爆炸
        )
        print("启用 DeepSpeed ZeRO-2 优化")
    except:
        print("DeepSpeed 不可用，使用标准分布式训练")
        deepspeed_plugin = None
    
    # 初始化 Accelerator
    # 这是 Accelerate 的核心，它会自动检测环境并配置分布式训练
    accelerator = Accelerator(deepspeed_plugin=deepspeed_plugin)
    
    # ====== 定义优化器和损失函数 ======
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss()  # 用于分类任务
    
    # ====== 关键步骤：prepare() ======
    # accelerator.prepare() 是 Accelerate 的核心方法
    # 它会自动处理模型、数据加载器、优化器的分布式包装
    print("准备分布式训练组件...")
    model, dataloader, optimizer = accelerator.prepare(model, dataloader, optimizer)
    
    # ====== 开始训练 ======
    print(f"开始训练，使用设备: {accelerator.device}")
    print(f"进程数量: {accelerator.num_processes}")
    print(f"是否为主进程: {accelerator.is_main_process}")
    print("-" * 50)
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        num_batches = 0
        
        for batch in dataloader:
            inputs, labels = batch
            # 注意：输入数据已经被 accelerate.prepare() 自动移动到正确的设备
            
            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # 计算准确率
            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()
            
            # 反向传播
            optimizer.zero_grad()
            # 使用 accelerator.backward() 而不是 loss.backward()
            # 这样可以正确处理混合精度和分布式梯度
            accelerator.backward(loss)
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        # 计算平均损失和准确率
        avg_loss = epoch_loss / num_batches
        accuracy = 100 * correct_predictions / total_samples
        
        # 只在主进程打印信息（避免重复输出）
        if accelerator.is_main_process:
            print(f"Epoch {epoch+1}/{num_epochs}")
            print(f"  Loss: {avg_loss:.6f}")
            print(f"  Accuracy: {accuracy:.2f}% ({correct_predictions}/{total_samples})")
    
    # ====== 保存模型 ======
    # 使用 accelerator.save() 确保只在主进程保存，避免冲突
    if accelerator.is_main_process:
        print("保存模型...")
        # 获取原始模型（去除分布式包装）
        unwrapped_model = accelerator.unwrap_model(model)
        torch.save(unwrapped_model.state_dict(), "model.pth")
        print("训练完成！模型已保存到 model.pth")