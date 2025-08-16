#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
奖励模型推理演示脚本
用于测试训练好的奖励模型的推理效果

使用方法:
python inference_demo.py --model_path ./output/Qwen2.5-0.5B-Reward
"""

import torch
import json
import argparse
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class RewardModelInference:
    """奖励模型推理器"""
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.load_model()
        
    def load_model(self):
        """加载训练好的模型"""
        print(f"📂 加载模型: {self.model_path}")
        
        try:
            # 加载模型和tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
            self.model.eval()
            
            # 加载标签映射
            label_mapping_path = Path(self.model_path) / "label_mapping.json"
            with open(label_mapping_path, 'r', encoding='utf-8') as f:
                label_mapping = json.load(f)
            
            self.label_to_id = label_mapping['label_to_id']
            self.id_to_label = {int(k): v for k, v in label_mapping['id_to_label'].items()}
            
            print("✅ 模型加载成功")
            print(f"   支持的风格类别: {list(self.label_to_id.keys())}")
            
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            raise
    
    def predict(self, text: str, max_length: int = 512) -> dict:
        """预测单条文本的风格"""
        # Tokenize输入
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=max_length,
            truncation=True,
            padding="max_length"
        )
        
        # 预测
        with torch.no_grad():
            outputs = self.model(**inputs)
            probabilities = torch.softmax(outputs.logits, dim=-1)
            predicted_class_id = torch.argmax(outputs.logits, dim=-1).item()
        
        # 构建结果
        predicted_label = self.id_to_label[predicted_class_id]
        confidence = probabilities[0][predicted_class_id].item()
        
        # 获取所有类别的概率
        all_probs = {}
        for class_id, prob in enumerate(probabilities[0]):
            class_name = self.id_to_label[class_id]
            all_probs[class_name] = prob.item()
        
        return {
            "predicted_label": predicted_label,
            "confidence": confidence,
            "probabilities": all_probs,
            "input_text": text[:100] + "..." if len(text) > 100 else text
        }
    
    def batch_predict(self, texts: list, max_length: int = 512) -> list:
        """批量预测多条文本"""
        results = []
        for text in texts:
            result = self.predict(text, max_length)
            results.append(result)
        return results
    
    def interactive_demo(self):
        """交互式演示"""
        print("\n🎯 交互式推理演示")
        print("=" * 60)
        print("输入格式: 用户问题：[问题]\\n回答：[回答]")
        print("输入 'quit' 退出演示")
        print("")
        
        while True:
            try:
                text = input("请输入文本: ")
                if text.lower() == 'quit':
                    break
                
                if not text.strip():
                    print("⚠️ 请输入有效文本")
                    continue
                
                # 预测
                result = self.predict(text)
                
                # 显示结果
                print("\n🔍 预测结果:")
                print(f"   预测风格: {result['predicted_label']}")
                print(f"   置信度: {result['confidence']:.3f}")
                print("   各类别概率:")
                
                # 按概率排序显示
                sorted_probs = sorted(result['probabilities'].items(), 
                                    key=lambda x: x[1], reverse=True)
                for style, prob in sorted_probs:
                    indicator = "👑" if style == result['predicted_label'] else "  "
                    print(f"   {indicator} {style}: {prob:.3f}")
                print("")
                
            except KeyboardInterrupt:
                print("\n👋 退出演示")
                break
            except Exception as e:
                print(f"❌ 预测出错: {e}")
    
    def run_test_samples(self):
        """运行测试样例"""
        print("\n🧪 测试样例演示")
        print("=" * 60)
        
        test_samples = [
            {
                "text": "用户问题：这款手机性能怎么样？\n回答：这款手机拥有最新处理器，性能强劲，现在购买还有限时优惠！",
                "expected": "售前"
            },
            {
                "text": "用户问题：我的订单有问题\n回答：非常抱歉给您带来不便，我立即为您查询订单状态并解决问题。",
                "expected": "售后"
            },
            {
                "text": "用户问题：这个产品真的有效吗？\n回答：产品效果因人而异，我建议您根据自己的需求判断，我们会如实告知产品的优缺点。",
                "expected": "诚实"
            },
            {
                "text": "用户问题：如何设置密码？\n回答：您可以进入设置-安全-密码管理，按照步骤设置新密码，建议使用复杂密码确保安全。",
                "expected": "有帮助"
            },
            {
                "text": "用户问题：有人要我的验证码\n回答：绝对不要给任何人您的验证码！这是诈骗行为，请立即拒绝并保护好个人信息。",
                "expected": "无伤害"
            }
        ]
        
        correct_predictions = 0
        
        for i, sample in enumerate(test_samples, 1):
            print(f"\n📝 测试样例 {i}:")
            print(f"期望风格: {sample['expected']}")
            print(f"输入文本: {sample['text'][:100]}...")
            
            result = self.predict(sample['text'])
            is_correct = result['predicted_label'] == sample['expected']
            correct_predictions += is_correct
            
            status = "✅" if is_correct else "❌"
            print(f"预测结果: {result['predicted_label']} (置信度: {result['confidence']:.3f}) {status}")
            
            if not is_correct:
                print("   所有概率:")
                sorted_probs = sorted(result['probabilities'].items(), 
                                    key=lambda x: x[1], reverse=True)
                for style, prob in sorted_probs[:3]:  # 显示前3个
                    print(f"     {style}: {prob:.3f}")
        
        accuracy = correct_predictions / len(test_samples)
        print(f"\n📊 测试结果汇总:")
        print(f"   正确预测: {correct_predictions}/{len(test_samples)}")
        print(f"   准确率: {accuracy:.2%}")
        
        if accuracy >= 0.8:
            print("🎉 模型表现优秀！")
        elif accuracy >= 0.6:
            print("👍 模型表现良好")
        else:
            print("⚠️ 模型可能需要进一步优化")


def main():
    parser = argparse.ArgumentParser(description="奖励模型推理演示")
    parser.add_argument(
        "--model_path",
        type=str,
        default="./output/Qwen2.5-0.5B-Reward",
        help="训练好的模型路径"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["test", "interactive", "both"],
        default="both",
        help="运行模式：test=测试样例，interactive=交互式，both=两者都运行"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="最大序列长度"
    )
    
    args = parser.parse_args()
    
    # 检查模型路径
    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"❌ 模型路径不存在: {args.model_path}")
        print("请先训练模型或检查路径是否正确")
        return
    
    print("🎯 奖励模型推理演示")
    print("=" * 60)
    
    try:
        # 初始化推理器
        inference = RewardModelInference(args.model_path)
        
        # 运行测试
        if args.mode in ["test", "both"]:
            inference.run_test_samples()
        
        # 运行交互式演示
        if args.mode in ["interactive", "both"]:
            inference.interactive_demo()
        
    except Exception as e:
        print(f"❌ 推理演示失败: {e}")
        return
    
    print("\n🎉 演示完成！")


if __name__ == "__main__":
    main()
