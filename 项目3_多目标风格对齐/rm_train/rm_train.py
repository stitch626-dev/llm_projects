#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
奖励模型训练脚本（TRL版本）
使用TRL库训练5类风格分类的奖励模型：售前、售后、诚实、有帮助、无伤害

特点：
- 使用TRL库的RewardConfig和RewardTrainer
- 基于Qwen2.5-0.5B-Instruct模型
- 支持多分类任务（5个风格类别）
- 集成我们构造的数据集
"""

import torch
import json
import os
from typing import Dict, List, Any
import argparse
from pathlib import Path

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
)
from trl import RewardConfig, RewardTrainer
from datasets import Dataset, DatasetDict
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report


class RewardModelTrainer:
    """奖励模型训练器"""
    
    def __init__(self, args):
        self.args = args
        
        # 风格标签映射
        self.label_to_id = {
            "售前": 0,
            "售后": 1, 
            "诚实": 2,
            "有帮助": 3,
            "无伤害": 4
        }
        self.id_to_label = {v: k for k, v in self.label_to_id.items()}
        
        print("🚀 奖励模型训练器初始化完成")
        print(f"📊 支持的风格类别: {list(self.label_to_id.keys())}")
        print(f"🔢 类别数量: {len(self.label_to_id)}")
    
    def load_dataset(self) -> DatasetDict:
        """加载我们构造的训练数据"""
        print(f"\n📂 加载训练数据: {self.args.train_file}")
        
        try:
            with open(self.args.train_file, 'r', encoding='utf-8') as f:
                train_data = json.load(f)
            
            print(f"✅ 成功加载 {len(train_data)} 条训练样本")
            
            # 统计标签分布
            label_counts = {}
            for item in train_data:
                label = item['label']
                label_counts[label] = label_counts.get(label, 0) + 1
            
            print("📊 训练数据标签分布:")
            for label, count in label_counts.items():
                print(f"   {label}: {count} 条")
            
            # 转换数据格式
            processed_data = []
            for item in train_data:
                processed_item = {
                    'text': item['input'],  # TRL期望的输入字段名
                    'label': self.label_to_id[item['label']],  # 转换为数值标签
                    'label_text': item['label'],  # 保留原始标签用于调试
                    'source': item['metadata']['source']
                }
                processed_data.append(processed_item)
            
            # 创建训练/验证集划分（80/20）
            split_idx = int(len(processed_data) * 0.8)
            train_split = processed_data[:split_idx]
            val_split = processed_data[split_idx:]
            
            dataset = DatasetDict({
                'train': Dataset.from_list(train_split),
                'validation': Dataset.from_list(val_split)
            })
            
            print(f"📊 数据划分:")
            print(f"   训练集: {len(dataset['train'])} 条")
            print(f"   验证集: {len(dataset['validation'])} 条")
            
            return dataset
            
        except FileNotFoundError:
            print(f"❌ 训练数据文件未找到: {self.args.train_file}")
            raise
        except Exception as e:
            print(f"❌ 加载数据时出错: {e}")
            raise
    
    def preprocess_function(self, examples):
        """数据预处理函数"""
        # 对文本进行tokenize
        model_inputs = self.tokenizer(
            examples["text"],
            max_length=self.args.max_length,
            truncation=True,
            padding="max_length",
            return_tensors=None
        )
        
        # 添加标签
        model_inputs["labels"] = examples["label"]
        return model_inputs
    
    def compute_metrics(self, eval_pred):
        """计算评估指标"""
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        
        accuracy = accuracy_score(labels, predictions)
        f1_macro = f1_score(labels, predictions, average='macro')
        f1_weighted = f1_score(labels, predictions, average='weighted')
        
        # 详细的分类报告
        target_names = [self.id_to_label[i] for i in range(len(self.id_to_label))]
        class_report = classification_report(
            labels, predictions, 
            target_names=target_names, 
            output_dict=True, 
            zero_division=0
        )
        
        return {
            'accuracy': accuracy,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'per_class_f1': {name: class_report[name]['f1-score'] 
                            for name in target_names if name in class_report}
        }
    
    def setup_model_and_tokenizer(self):
        """初始化模型和tokenizer"""
        print(f"\n🤖 初始化模型: {self.args.model_name_or_path}")
        
        # 加载tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.args.model_name_or_path,
            trust_remote_code=True
        )
        
        # 设置pad_token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 加载模型（用于分类任务）
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.args.model_name_or_path,
            num_labels=len(self.label_to_id),  # 5个类别
            trust_remote_code=True
        )
        
        # 设置模型的pad_token_id
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        
        print(f"✅ 模型加载完成")
        print(f"   模型类别数: {self.model.config.num_labels}")
        print(f"   词汇表大小: {len(self.tokenizer)}")
    
    def train(self):
        """开始训练"""
        print("\n🎯 开始训练奖励模型...")
        
        # 1. 初始化模型和tokenizer
        self.setup_model_and_tokenizer()
        
        # 2. 加载数据集
        dataset = self.load_dataset()
        
        # 3. 数据预处理
        print("\n🔧 预处理数据...")
        tokenized_dataset = dataset.map(
            self.preprocess_function,
            batched=True,
            remove_columns=dataset["train"].column_names
        )
        
        # 4. 设置训练参数
        training_args = RewardConfig(
            output_dir=self.args.output_dir,
            per_device_train_batch_size=self.args.per_device_train_batch_size,
            per_device_eval_batch_size=self.args.per_device_eval_batch_size,
            eval_strategy="epoch",
            save_strategy="epoch",
            learning_rate=self.args.learning_rate,
            num_train_epochs=self.args.num_train_epochs,
            weight_decay=self.args.weight_decay,
            logging_dir=os.path.join(self.args.output_dir, "logs"),
            logging_steps=10,
            fp16=True if torch.cuda.is_available() else False,
            save_safetensors=True,
            load_best_model_at_end=True,
            metric_for_best_model="eval_f1_macro",
            greater_is_better=True,
            report_to="none",
            remove_unused_columns=False,
        )
        
        # 5. 创建训练器
        trainer = RewardTrainer(
            args=training_args,
            model=self.model,
            processing_class=self.tokenizer,
            train_dataset=tokenized_dataset["train"],
            eval_dataset=tokenized_dataset["validation"],
            compute_metrics=self.compute_metrics,
        )
        
        # 6. 训练前评估
        print("\n📊 训练前模型评估:")
        initial_metrics = trainer.evaluate()
        for key, value in initial_metrics.items():
            if isinstance(value, float):
                print(f"   {key}: {value:.4f}")
            else:
                print(f"   {key}: {value}")
        
        # 7. 开始训练
        print("\n🚀 开始训练...")
        trainer.train()
        
        # 8. 训练后评估
        print("\n📊 训练后模型评估:")
        final_metrics = trainer.evaluate()
        for key, value in final_metrics.items():
            if isinstance(value, float):
                print(f"   {key}: {value:.4f}")
            else:
                print(f"   {key}: {value}")
        
        # 9. 保存模型
        print(f"\n💾 保存模型到: {self.args.output_dir}")
        trainer.save_model()
        self.tokenizer.save_pretrained(self.args.output_dir)
        
        # 10. 保存标签映射
        label_mapping_path = os.path.join(self.args.output_dir, "label_mapping.json")
        with open(label_mapping_path, 'w', encoding='utf-8') as f:
            json.dump({
                'label_to_id': self.label_to_id,
                'id_to_label': self.id_to_label
            }, f, ensure_ascii=False, indent=2)
        
        print(f"✅ 标签映射已保存到: {label_mapping_path}")
        print("\n🎉 训练完成！")
        
        return final_metrics
    
    def demo_inference(self):
        """演示推理过程"""
        print("\n🎯 演示模型推理...")
        
        # 测试样例
        test_samples = [
            "用户问题：这款手机性能怎么样？\n回答：这款手机拥有最新处理器，性能强劲，现在购买还有限时优惠！",
            "用户问题：我的订单有问题\n回答：非常抱歉给您带来不便，我立即为您查询订单状态并解决问题。",
            "用户问题：这个产品真的有效吗？\n回答：产品效果因人而异，我建议您根据自己的需求判断，我们会如实告知产品的优缺点。",
            "用户问题：如何设置密码？\n回答：您可以进入设置-安全-密码管理，按照步骤设置新密码，建议使用复杂密码确保安全。",
            "用户问题：有人要我的验证码\n回答：绝对不要给任何人您的验证码！这是诈骗行为，请立即拒绝并保护好个人信息。"
        ]
        
        expected_labels = ["售前", "售后", "诚实", "有帮助", "无伤害"]
        
        # 加载训练好的模型进行推理
        try:
            model = AutoModelForSequenceClassification.from_pretrained(self.args.output_dir)
            tokenizer = AutoTokenizer.from_pretrained(self.args.output_dir)
            model.eval()
            
            print("🔍 推理结果:")
            print("-" * 80)
            
            for i, (sample, expected) in enumerate(zip(test_samples, expected_labels)):
                # Tokenize输入
                inputs = tokenizer(
                    sample, 
                    return_tensors="pt", 
                    max_length=self.args.max_length,
                    truncation=True, 
                    padding="max_length"
                )
                
                # 预测
                with torch.no_grad():
                    outputs = model(**inputs)
                    probabilities = torch.softmax(outputs.logits, dim=-1)
                    predicted_class_id = torch.argmax(outputs.logits, dim=-1).item()
                
                predicted_label = self.id_to_label[predicted_class_id]
                confidence = probabilities[0][predicted_class_id].item()
                
                # 显示结果
                print(f"\n样本 {i+1}: {expected} (期望)")
                print(f"输入: {sample[:100]}...")
                print(f"预测: {predicted_label} (置信度: {confidence:.3f})")
                print(f"正确: {'✅' if predicted_label == expected else '❌'}")
                
                # 显示所有类别的概率
                print("各类别概率:")
                for class_id, prob in enumerate(probabilities[0]):
                    class_name = self.id_to_label[class_id]
                    print(f"  {class_name}: {prob:.3f}")
        
        except Exception as e:
            print(f"⚠️ 推理演示失败: {e}")
            print("请确保模型训练完成并正确保存")


def main():
    parser = argparse.ArgumentParser(description="奖励模型训练脚本")
    
    # 模型参数
    parser.add_argument(
        "--model_name_or_path", 
        type=str, 
        default="Qwen/Qwen2.5-0.5B-Instruct",
        help="预训练模型名称或路径"
    )
    
    # 数据参数
    parser.add_argument(
        "--train_file", 
        type=str, 
        default="rm_data/output_data/training_data.json",
        help="训练数据文件路径"
    )
    
    # 训练参数
    parser.add_argument("--output_dir", type=str, default="rm_train/output/Qwen2.5-0.5B-Reward", help="输出目录")
    parser.add_argument("--max_length", type=int, default=512, help="最大序列长度")
    parser.add_argument("--per_device_train_batch_size", type=int, default=4, help="训练批次大小")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8, help="评估批次大小")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="学习率")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="训练轮数")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="权重衰减")
    
    # 其他参数
    parser.add_argument("--demo_inference", action="store_true", help="训练后进行推理演示")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    
    args = parser.parse_args()
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # 确保输出目录存在
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("🏆 奖励模型训练系统")
    print("=" * 80)
    print(f"🤖 基础模型: {args.model_name_or_path}")
    print(f"📂 训练数据: {args.train_file}")
    print(f"📁 输出目录: {args.output_dir}")
    print(f"⚙️  训练轮数: {args.num_train_epochs}")
    print(f"📊 批次大小: {args.per_device_train_batch_size}")
    print(f"🎯 最大长度: {args.max_length}")
    
    # 创建训练器并开始训练
    trainer = RewardModelTrainer(args)
    metrics = trainer.train()
    
    # 可选的推理演示
    if args.demo_inference:
        trainer.demo_inference()
    
    print("\n" + "=" * 80)
    print("🎉 训练流程完成！")
    print("=" * 80)


if __name__ == "__main__":
    main()