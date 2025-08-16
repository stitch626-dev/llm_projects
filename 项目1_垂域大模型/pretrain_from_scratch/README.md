# Qwen2.5-0.5B 继续预训练

基于 Qwen/Qwen2.5-0.5B 模型进行继续预训练的训练脚本。

## 环境要求

- Python 3.8+
- PyTorch 
- transformers
- torch
- wandb (可选，用于训练监控)

安装依赖：
```bash
pip install torch transformers wandb
```

## 数据准备

训练数据应该是 JSONL 格式，每行包含一个包含 "text" 字段的JSON对象：

```json
{"text": "这是训练文本内容..."}
{"text": "另一段训练文本..."}
```

数据文件：
- **默认使用**: `./data/pretrain_100m.jsonl` (100MB，约73K行数据)


## 训练使用

### 基础训练
```bash
cd code
python train_pretrain.py
```

### 自定义参数
```bash
python train_pretrain.py \
    --epochs 3 \
    --batch_size 16 \
    --learning_rate 1e-5 \
    --max_seq_len 1024 \
    --save_interval 500
```

### 分布式训练
```bash
torchrun --nproc_per_node 2 train_pretrain.py --ddp
```

## 主要参数说明

- `--epochs`: 训练轮数，默认1
- `--batch_size`: 批次大小，默认32  
- `--learning_rate`: 学习率，默认5e-4
- `--max_seq_len`: 最大序列长度，默认512
- `--accumulation_steps`: 梯度累积步数，默认8
- `--save_interval`: 保存间隔步数，默认100
- `--data_path`: 训练数据路径，默认./data/pretrain_hq.jsonl
- `--out_dir`: 输出目录，默认../out
- `--use_wandb`: 启用wandb监控

## 模型保存

训练过程中会自动保存checkpoint到 `{out_dir}/checkpoint-{step}` 目录，包含：
- 模型权重
- tokenizer配置

## 训练监控

使用 `--use_wandb` 参数启用wandb监控，可以实时查看：
- 训练损失曲线
- 学习率变化  
- 训练时间估算
