# Accelerate 分布式训练学习示例

## 文件说明

- `accelerate_train.py` - 使用 Accelerate 库的分布式训练示例脚本
- `simple_train.py` - 简化版训练脚本，不依赖 accelerate（用于测试）
- `config.json` - 单GPU/CPU 配置文件  
- `config_multi_gpu.json` - 多GPU 配置文件

## 依赖安装

### 基础版本（运行 simple_train.py）
```bash
pip install torch
```

### 完整版本（运行 accelerate_train.py）  
```bash
pip install accelerate torch
# 可选：DeepSpeed 支持
pip install deepspeed
```

## 核心概念

### Accelerate 的主要优势
1. **自动设备管理** - 无需手动移动模型和数据到GPU
2. **简化分布式** - 几行代码实现多GPU训练
3. **DeepSpeed集成** - 支持ZeRO优化降低显存占用
4. **混合精度** - 自动处理fp16/bf16加速

### 关键API
- `Accelerator()` - 核心对象，自动检测环境
- `accelerator.prepare()` - 包装模型、数据、优化器
- `accelerator.backward()` - 分布式反向传播
- `accelerator.is_main_process` - 主进程判断

## 运行方式

### 1. 测试环境（推荐先运行）
```bash
python simple_train.py
```
这个脚本使用标准PyTorch，验证数据和模型是否正常工作。

### 2. Accelerate单进程模式
```bash
python accelerate_train.py
# 或者
accelerate launch --config_file config.json accelerate_train.py
```

### 3. Accelerate多GPU模式
```bash
accelerate launch --config_file config_multi_gpu.json accelerate_train.py
```

### 4. 交互式配置
```bash
accelerate config  # 生成自定义配置文件
```

## 配置参数说明

### config.json 参数详解

| 参数 | 说明 |
|------|------|
| `compute_environment` | 计算环境类型，`LOCAL_MACHINE`表示本地机器 |
| `debug` | 是否启用调试模式，输出更多信息 |
| `distributed_type` | 分布式类型，`MULTI_GPU`表示多GPU训练 |
| `downcast_bf16` | 是否将bf16降级为fp16 |
| `enable_cpu_affinity` | 是否启用CPU亲和性绑定 |
| `machine_rank` | 机器排名，多机器训练时使用 |
| `main_training_function` | 主训练函数名 |
| `mixed_precision` | 混合精度设置：`no`/`fp16`/`bf16` |
| `num_machines` | 机器数量 |
| `num_processes` | 进程数量（通常等于GPU数量） |
| `rdzv_backend` | 会合后端，用于进程间通信 |
| `same_network` | 进程是否在同一网络 |
| `use_cpu` | 是否使用CPU训练 |

## 任务说明

### 数据集
- **任务类型**: 二分类问题
- **数据**: 1000个样本，10维随机特征
- **标签规则**: 前两个特征之和 > 0 为类别1，否则为类别0
- **噪声**: 10%的标签噪声增加任务难度

### 模型架构
- **输入层**: 10维特征
- **隐藏层**: 32个神经元 + ReLU激活
- **输出层**: 2个类别的logits
- **总参数**: ~700个参数

### 训练设置
- **优化器**: Adam (lr=0.001)
- **损失函数**: CrossEntropyLoss  
- **批次大小**: 32
- **训练轮数**: 5轮（快速演示）

## 学习重点

1. **对比学习**: 运行`simple_train.py`和`accelerate_train.py`，观察代码差异
2. **核心API**: 理解`prepare()`、`backward()`、`is_main_process`的作用
3. **配置管理**: 掌握不同场景的配置文件用法
4. **扩展性**: 了解如何从单GPU扩展到多GPU/多机器
