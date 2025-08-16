# 奖励模型训练系统

基于TRL库实现的5类风格奖励模型训练系统，用于训练能够识别**售前**、**售后**、**诚实**、**有帮助**、**无伤害**五种对话风格的分类模型。

## 🎯 项目概述

这个训练系统将我们构造的奖励模型数据集转换为一个实用的分类模型，可以：

- **输入**: 用户问题 + AI回答的组合文本
- **输出**: 5种风格的分类概率和置信度
- **应用**: 强化学习中的奖励信号，内容风格控制，质量评估等

### 支持的风格类别

| 中文标签 | 英文标识 | 类别ID | 描述 |
|---------|---------|--------|------|
| 售前 | sales_oriented | 0 | 销售导向，推荐产品，促进购买 |
| 售后 | after_sales | 1 | 售后服务，解决问题，处理投诉 |
| 诚实 | honesty | 2 | 诚实客观，实事求是，承认限制 |
| 有帮助 | helpful | 3 | 提供帮助，解答疑问，给出建议 |
| 无伤害 | harmless | 4 | 安全无害，拒绝危险，保护用户 |

## 🛠️ 环境要求

### 依赖库
```bash
pip install torch transformers trl datasets scikit-learn numpy
```

### 测试环境
- Python 3.8+
- PyTorch 2.0+
- Transformers 4.30+
- TRL 0.7+

## 📁 文件结构

```
rm_train/
├── rm_train.py          # 主训练脚本
├── README.md            # 本文档
├── requirements.txt     # 依赖列表
└── output/             # 训练输出目录
    └── Qwen2.5-0.5B-Reward/  # 训练好的模型
        ├── config.json
        ├── pytorch_model.bin
        ├── tokenizer.json
        └── label_mapping.json
```

## 🚀 快速开始

### 1. 基本训练
```bash
cd rm_train
python rm_train.py
```

### 2. 自定义参数训练
```bash
python rm_train.py \
    --model_name_or_path "Qwen/Qwen2.5-0.5B-Instruct" \
    --train_file "../rm_data/output_data/training_data.json" \
    --output_dir "./output/my_reward_model" \
    --num_train_epochs 5 \
    --per_device_train_batch_size 8 \
    --learning_rate 2e-5 \
    --demo_inference
```

## ⚙️ 训练参数说明

### 核心参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model_name_or_path` | Qwen/Qwen2.5-0.5B-Instruct | 基础预训练模型 |
| `--train_file` | rm_data/output_data/training_data.json | 训练数据文件 |
| `--output_dir` | rm_train/output/Qwen2.5-0.5B-Reward | 模型输出目录 |
| `--max_length` | 512 | 最大输入序列长度 |
| `--num_train_epochs` | 3 | 训练轮数 |
| `--learning_rate` | 2e-5 | 学习率 |

### 性能参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--per_device_train_batch_size` | 4 | 每设备训练批次大小 |
| `--per_device_eval_batch_size` | 8 | 每设备评估批次大小 |
| `--weight_decay` | 0.01 | 权重衰减 |
| `--seed` | 42 | 随机种子 |

### 功能参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--demo_inference` | False | 训练后执行推理演示 |

## 📊 训练过程

### 1. 数据加载与预处理
```
📂 加载训练数据: rm_data/output_data/training_data.json
✅ 成功加载 26 条训练样本
📊 训练数据标签分布:
   售前: 7 条
   售后: 4 条
   有帮助: 6 条
   无伤害: 4 条
   诚实: 5 条
📊 数据划分:
   训练集: 20 条
   验证集: 6 条
```

### 2. 模型初始化
```
🤖 初始化模型: Qwen/Qwen2.5-0.5B-Instruct
✅ 模型加载完成
   模型类别数: 5
   词汇表大小: 151936
```

### 3. 训练指标监控
- **Accuracy**: 分类准确率
- **F1 Macro**: 宏平均F1分数
- **F1 Weighted**: 加权F1分数
- **Per Class F1**: 每个类别的F1分数

## 📈 模型输出

### 1. 训练结果示例
```
📊 训练后模型评估:
   eval_accuracy: 0.8333
   eval_f1_macro: 0.8200
   eval_f1_weighted: 0.8300
   eval_per_class_f1: {
     '售前': 0.8571,
     '售后': 0.7500,
     '诚实': 0.8000,
     '有帮助': 0.8889,
     '无伤害': 0.8000
   }
```

### 2. 推理输出格式
```python
# 单条推理结果
{
    "predicted_label": "诚实",
    "confidence": 0.856,
    "probabilities": {
        "售前": 0.045,
        "售后": 0.032,
        "诚实": 0.856,
        "有帮助": 0.054,
        "无伤害": 0.013
    }
}
```

### 3. 文件输出
- `config.json`: 模型配置文件
- `pytorch_model.bin`: 训练好的模型权重
- `tokenizer.json`: 分词器配置
- `label_mapping.json`: 标签映射关系

## 🎯 模型推理

### 1. 加载训练好的模型
```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import json

# 加载模型和tokenizer
model_path = "rm_train/output/Qwen2.5-0.5B-Reward"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# 加载标签映射
with open(f"{model_path}/label_mapping.json", 'r') as f:
    label_mapping = json.load(f)
id_to_label = label_mapping['id_to_label']
```

### 2. 推理函数
```python
def predict_style(text, model, tokenizer, id_to_label, max_length=512):
    """预测文本的风格类别"""
    # Tokenize输入
    inputs = tokenizer(
        text, 
        return_tensors="pt", 
        max_length=max_length,
        truncation=True, 
        padding="max_length"
    )
    
    # 预测
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = torch.softmax(outputs.logits, dim=-1)
        predicted_class_id = torch.argmax(outputs.logits, dim=-1).item()
    
    # 构建结果
    predicted_label = id_to_label[str(predicted_class_id)]
    confidence = probabilities[0][predicted_class_id].item()
    
    all_probs = {}
    for i, prob in enumerate(probabilities[0]):
        all_probs[id_to_label[str(i)]] = prob.item()
    
    return {
        "predicted_label": predicted_label,
        "confidence": confidence,
        "probabilities": all_probs
    }
```

### 3. 使用示例
```python
# 测试样例
test_text = "用户问题：这个产品有什么优缺点？\n回答：这个产品确实有很多优点，但也存在一些限制，建议您根据自己的需求来判断是否合适。"

result = predict_style(test_text, model, tokenizer, id_to_label)
print(f"预测风格: {result['predicted_label']}")
print(f"置信度: {result['confidence']:.3f}")
print("各类别概率:")
for style, prob in result['probabilities'].items():
    print(f"  {style}: {prob:.3f}")
```

## 🔧 自定义配置

### 1. 修改训练数据
如果要使用自己的数据，确保JSON格式如下：
```json
[
  {
    "input": "用户问题：[问题]\n回答：[回答]",
    "label": "售前",
    "label_english": "sales_oriented",
    "metadata": {
      "source": "business",
      "scene": "商品咨询"
    }
  }
]
```

### 2. 添加新的风格类别
1. 修改 `rm_train.py` 中的 `label_to_id` 映射
2. 更新训练数据中的标签
3. 重新训练模型

### 3. 调整模型架构
- 更换基础模型：修改 `--model_name_or_path` 参数
- 调整序列长度：修改 `--max_length` 参数
- 优化超参数：调整学习率、批次大小等


## ❓ 常见问题

### Q1: 显存不足怎么办？
**A**: 减小批次大小和序列长度
```bash
--per_device_train_batch_size 2 \
--max_length 256
```

### Q2: 如何提高模型性能？
**A**: 
- 增加训练轮数：`--num_train_epochs 5`
- 调整学习率：`--learning_rate 1e-5`
- 使用更大的基础模型
- 增加训练数据量

### Q3: 如何处理类别不平衡？
**A**: 
- 在数据预处理阶段进行数据增强
- 使用类别权重平衡损失函数
- 调整训练样本的采样策略

### Q4: 训练时间太长怎么办？
**A**:
- 使用GPU加速
- 减少训练轮数和序列长度
- 使用更小的基础模型

### Q5: 如何评估模型质量？
**A**:
- 关注F1分数而不仅仅是准确率
- 查看每个类别的详细指标
- 使用混淆矩阵分析错误类型
- 进行人工评估验证

## 🚀 进阶功能

### 1. 集成到推理服务
```python
from fastapi import FastAPI
import torch

app = FastAPI()
model = None
tokenizer = None

@app.on_event("startup")
async def load_model():
    global model, tokenizer
    model = AutoModelForSequenceClassification.from_pretrained("./output/Qwen2.5-0.5B-Reward")
    tokenizer = AutoTokenizer.from_pretrained("./output/Qwen2.5-0.5B-Reward")

@app.post("/predict")
async def predict(text: str):
    result = predict_style(text, model, tokenizer, id_to_label)
    return result
```

### 2. 批量推理
```python
def batch_predict(texts, model, tokenizer, id_to_label, batch_size=32):
    """批量预测多个文本的风格"""
    results = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        # 批量处理逻辑...
        # 返回批量结果
    return results
```

### 3. 模型蒸馏
使用训练好的大模型来指导更小模型的训练，提高推理速度。

## 📞 技术支持

如果在训练过程中遇到问题，请检查：

1. **环境依赖**: 确保所有依赖库版本正确
2. **数据格式**: 验证训练数据的JSON格式
3. **硬件资源**: 确保有足够的GPU显存或CPU内存
4. **模型权限**: 确保能够访问Hugging Face模型

更多技术细节和最佳实践，请参考项目代码和注释。

---

🎉 **祝您训练顺利！** 如果这个奖励模型在您的项目中发挥了作用，欢迎分享使用体验和改进建议。
