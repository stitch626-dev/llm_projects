#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
奖励模型分类数据构造和处理系统（重构版）
用于构建5类风格的分类数据集：售前、售后、诚实、有帮助、无伤害

特点：
- 数据和代码分离：原始数据存储为JSON文件
- 使用相对路径：便于项目迁移和维护
- 配置化管理：通过配置文件管理映射关系和参数

数据流程：
1. 从JSON文件加载业务数据和开源数据
2. 场景标签映射到风格维度
3. 质量筛选（保留4星以上）
4. 对话拆分
5. 数据重写和清洗
6. 去重处理
7. 最终数据集整合
8. 按风格保存独立数据文件
"""

import json
import os
import random
import re
from typing import List, Dict, Tuple
from collections import defaultdict, Counter
import hashlib
from datetime import datetime
from pathlib import Path


class RewardModelDataProcessor:
    """奖励模型数据处理器（重构版）"""
    
    def __init__(self, config_path: str = "rm_data/input_data/config.json"):
        """
        初始化处理器
        
        Args:
            config_path: 配置文件路径（相对路径）
        """
        self.config_path = config_path
        self.config = self.load_config()
        
        # 从配置文件中加载映射关系
        self.scene_to_style_mapping = self.config["scene_to_style_mapping"]
        self.style_to_english = self.config["style_to_english"]
        self.processing_settings = self.config["processing_settings"]
        
        # 设置路径
        self.input_data_dir = self.config["paths"]["input_data_dir"]
        self.output_data_dir = self.config["paths"]["output_data_dir"]
        
        # 确保输出目录存在
        Path(self.output_data_dir).mkdir(parents=True, exist_ok=True)
        
        print("=" * 60)
        print("🚀 奖励模型数据处理系统初始化完成（重构版）")
        print(f"📁 输入目录: {self.input_data_dir}")
        print(f"📁 输出目录: {self.output_data_dir}")
        print("=" * 60)
    
    def load_config(self) -> Dict:
        """加载配置文件"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            print(f"✅ 成功加载配置文件: {self.config_path}")
            return config
        except FileNotFoundError:
            print(f"❌ 配置文件未找到: {self.config_path}")
            raise
        except json.JSONDecodeError as e:
            print(f"❌ 配置文件格式错误: {e}")
            raise
    
    def load_business_data(self) -> List[Dict]:
        """
        步骤1: 从JSON文件加载业务数据
        """
        print("\n📊 步骤1: 从文件加载业务数据...")
        
        business_file = os.path.join(self.input_data_dir, self.config["paths"]["business_data_file"])
        
        try:
            with open(business_file, 'r', encoding='utf-8') as f:
                business_json = json.load(f)
            
            conversations = business_json["conversations"]
            print(f"✅ 成功加载业务数据文件: {business_file}")
            print(f"   共 {len(conversations)} 条对话数据")
            
            # 统计场景分布
            scene_counts = Counter([conv['scene'] for conv in conversations])
            print("   场景分布:")
            for scene, count in scene_counts.items():
                print(f"     {scene}: {count} 条")
            
            return conversations
            
        except FileNotFoundError:
            print(f"❌ 业务数据文件未找到: {business_file}")
            raise
        except Exception as e:
            print(f"❌ 加载业务数据时出错: {e}")
            raise
    
    def load_opensource_data(self) -> List[Dict]:
        """
        步骤2: 从JSON文件加载开源数据
        """
        print("\n📚 步骤2: 从文件加载开源数据...")
        
        opensource_file = os.path.join(self.input_data_dir, self.config["paths"]["opensource_data_file"])
        
        try:
            with open(opensource_file, 'r', encoding='utf-8') as f:
                opensource_json = json.load(f)
            
            samples = opensource_json["samples"]
            print(f"✅ 成功加载开源数据文件: {opensource_file}")
            print(f"   共 {len(samples)} 条样本数据")
            
            # 统计风格分布
            style_counts = Counter([sample['style'] for sample in samples])
            print("   风格分布:")
            for style, count in style_counts.items():
                print(f"     {style}: {count} 条")
            
            return samples
            
        except FileNotFoundError:
            print(f"❌ 开源数据文件未找到: {opensource_file}")
            raise
        except Exception as e:
            print(f"❌ 加载开源数据时出错: {e}")
            raise
    
    def map_scenes_to_styles(self, business_data: List[Dict]) -> List[Dict]:
        """
        步骤3: 场景标签映射到风格维度
        将业务场景标签映射到5类风格维度
        """
        print("\n🔄 步骤3: 场景标签映射到风格维度...")
        
        mapped_data = []
        unmapped_scenes = []
        
        for data in business_data:
            scene = data['scene']
            if scene in self.scene_to_style_mapping:
                mapped_style = self.scene_to_style_mapping[scene]
                data['style'] = mapped_style
                data['style_english'] = self.style_to_english[mapped_style]
                mapped_data.append(data)
            else:
                unmapped_scenes.append(scene)
                print(f"⚠️  未找到场景 '{scene}' 的映射，跳过")
        
        if unmapped_scenes:
            print(f"⚠️  共有 {len(unmapped_scenes)} 个场景未映射: {set(unmapped_scenes)}")
        
        print(f"✅ 场景映射完成，共 {len(mapped_data)} 条数据")
        style_counts = Counter([d['style'] for d in mapped_data])
        print("   映射后风格分布:")
        for style, count in style_counts.items():
            print(f"     {style}: {count} 条")
            
        return mapped_data
    
    def quality_filtering(self, data: List[Dict], data_type: str = "business") -> List[Dict]:
        """
        步骤4: 质量筛选
        只保留用户反馈比较高的数据
        """
        min_score = self.processing_settings["min_score"]
        print(f"\n🔍 步骤4: 质量筛选-{data_type}（保留 >= {min_score} 星评分）...")
        
        filtered_data = []
        for item in data:
            # 业务数据有user_satisfaction字段
            if 'user_satisfaction' in item:
                if item['user_satisfaction'] >= min_score:
                    filtered_data.append(item)
            # 开源数据有score字段
            elif 'score' in item:
                if item['score'] >= min_score:
                    filtered_data.append(item)
        
        filter_rate = len(filtered_data) / len(data) * 100 if len(data) > 0 else 0
        print(f"✅ 质量筛选完成")
        print(f"   原始数据: {len(data)} 条")  
        print(f"   筛选后: {len(filtered_data)} 条")
        print(f"   筛选率: {filter_rate:.1f}%")
        
        return filtered_data
    
    def split_conversations(self, business_data: List[Dict]) -> List[Dict]:
        """
        步骤5: 对话拆分
        将多轮对话拆分为单轮问答对
        """
        print("\n✂️  步骤5: 对话拆分...")
        
        split_data = []
        for data in business_data:
            conversations = data.get('conversations', [])
            for i, conv in enumerate(conversations):
                split_item = {
                    'original_dialogue_id': data['dialogue_id'],
                    'turn_id': f"{data['dialogue_id']}_turn_{i+1}",
                    'scene': data['scene'],
                    'style': data['style'],
                    'style_english': data['style_english'], 
                    'user_query': conv['user'],
                    'agent_reply': conv['agent'],
                    'user_satisfaction': data['user_satisfaction']
                }
                split_data.append(split_item)
        
        print(f"✅ 对话拆分完成")
        print(f"   原始对话: {len(business_data)} 条")
        print(f"   拆分后问答对: {len(split_data)} 条")
        
        return split_data
    
    def clean_and_rewrite_data(self, data: List[Dict], data_type: str = "business") -> List[Dict]:
        """
        步骤6: 数据清洗和重写
        规范化问答格式，去除特殊字符
        """
        print(f"\n🧹 步骤6: 数据清洗和重写-{data_type}...")
        
        cleaned_data = []
        for item in data:
            # 模拟数据清洗：去除特殊字符，规范化格式
            cleaned_item = item.copy()
            
            # 清洗用户问题
            user_query = item['user_query']
            user_query = re.sub(r'[^\w\s\u4e00-\u9fff？！，。？！]', '', user_query)
            user_query = user_query.strip()
            
            # 清洗客服回复  
            agent_reply = item['agent_reply']
            agent_reply = re.sub(r'[^\w\s\u4e00-\u9fff？！，。？！～]', '', agent_reply)
            agent_reply = agent_reply.strip()
            
            # 确保回复长度合理
            if len(user_query) > 0 and len(agent_reply) > 0:
                cleaned_item['user_query'] = user_query
                cleaned_item['agent_reply'] = agent_reply
                cleaned_data.append(cleaned_item)
        
        print(f"✅ 数据清洗完成")  
        print(f"   清洗前: {len(data)} 条")
        print(f"   清洗后: {len(cleaned_data)} 条")
        
        return cleaned_data
    
    def deduplicate_data(self, data: List[Dict], data_type: str = "business") -> List[Dict]:
        """
        步骤7: 去重处理
        使用精确去重和语义去重
        """
        print(f"\n🔄 步骤7: 去重处理-{data_type}...")
        
        # 第一步：精确去重（使用hash）
        seen_hashes = set()
        deduplicated_data = []
        
        for item in data:
            # 创建内容的hash
            content = item['user_query'] + item['agent_reply']
            content_hash = hashlib.md5(content.encode('utf-8')).hexdigest()
            
            if content_hash not in seen_hashes:
                seen_hashes.add(content_hash)
                deduplicated_data.append(item)
        
        # 第二步：语义去重（简单相似度检查）
        duplicate_threshold = self.processing_settings["duplicate_threshold"]
        final_data = []
        
        for item in deduplicated_data:
            is_similar = False
            for existing_item in final_data:
                # 简单的相似度检查
                if (item.get('style') == existing_item.get('style') and 
                    len(set(item['user_query']) & set(existing_item['user_query'])) > len(item['user_query']) * duplicate_threshold):
                    is_similar = True
                    break
            
            if not is_similar:
                final_data.append(item)
        
        print(f"✅ 去重完成")
        print(f"   去重前: {len(data)} 条")
        print(f"   精确去重后: {len(deduplicated_data)} 条") 
        print(f"   语义去重后: {len(final_data)} 条")
        
        return final_data
    
    def integrate_final_dataset(self, business_data: List[Dict], open_source_data: List[Dict]) -> Dict:
        """
        步骤8: 整合最终数据集
        合并业务数据和开源数据，形成最终训练数据集
        """
        print("\n🔗 步骤8: 整合最终数据集...")
        
        # 将开源数据转换为统一格式
        formatted_open_source = []
        for item in open_source_data:
            formatted_item = {
                'turn_id': f"opensource_{len(formatted_open_source)+1}",
                'scene': 'opensource',
                'style': item['style'] if item['style'] in ['helpful', 'harmless', 'honesty'] else 'helpful',
                'style_english': item['style'] if item['style'] in ['helpful', 'harmless', 'honesty'] else 'helpful',
                'user_query': item['user_query'],
                'agent_reply': item['agent_reply'],
                'score': item['score'],
                'source': 'opensource'
            }
            # 映射style到中文
            style_mapping = {'helpful': '有帮助', 'harmless': '无伤害', 'honesty': '诚实'}
            if formatted_item['style'] in style_mapping:
                formatted_item['style'] = style_mapping[formatted_item['style']]
            formatted_open_source.append(formatted_item)
        
        # 为业务数据添加source标识
        for item in business_data:
            item['source'] = 'business'
        
        # 合并数据
        all_data = business_data + formatted_open_source
        
        # 按风格分组
        grouped_data = defaultdict(list)
        for item in all_data:
            grouped_data[item['style']].append(item)
        
        # 计算数据分布
        total_count = len(all_data)
        business_count = len(business_data)
        opensource_count = len(formatted_open_source)
        
        print(f"✅ 数据集整合完成")
        print(f"   总数据量: {total_count} 条")
        print(f"   业务数据: {business_count} 条 ({business_count/total_count*100:.1f}%)")
        print(f"   开源数据: {opensource_count} 条 ({opensource_count/total_count*100:.1f}%)")
        print("\n   各风格分布:")
        for style, items in grouped_data.items():
            print(f"     {style}: {len(items)} 条 ({len(items)/total_count*100:.1f}%)")
        
        # 构造最终数据集格式
        final_dataset = {
            'metadata': {
                'total_samples': total_count,
                'business_ratio': f"{business_count/total_count*100:.1f}%",
                'opensource_ratio': f"{opensource_count/total_count*100:.1f}%", 
                'creation_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'style_distribution': {style: len(items) for style, items in grouped_data.items()},
                'processing_settings': self.processing_settings
            },
            'styles': {}
        }
        
        # 按风格组织数据
        for style, items in grouped_data.items():
            style_english = self.style_to_english.get(style, style.lower())
            final_dataset['styles'][style] = {
                'style_name': style,
                'style_english': style_english,
                'sample_count': len(items),
                'samples': [
                    {
                        'user_query': item['user_query'],
                        'agent_reply': item['agent_reply'],
                        'source': item['source'],
                        'turn_id': item.get('turn_id', ''),
                        'scene': item.get('scene', '')
                    }
                    for item in items
                ]
            }
        
        return final_dataset
    
    def save_style_datasets(self, final_dataset: Dict):
        """
        保存各风格的独立数据文件
        """
        print("\n💾 保存各风格独立数据文件...")
        
        for style, style_data in final_dataset['styles'].items():
            # 为每个风格创建独立的数据文件
            style_filename = f"style_{style_data['style_english']}.json"
            style_filepath = os.path.join(self.output_data_dir, style_filename)
            
            style_dataset = {
                'metadata': {
                    'style_name': style,
                    'style_english': style_data['style_english'],
                    'sample_count': style_data['sample_count'],
                    'creation_time': final_dataset['metadata']['creation_time']
                },
                'samples': style_data['samples']
            }
            
            with open(style_filepath, 'w', encoding='utf-8') as f:
                json.dump(style_dataset, f, ensure_ascii=False, indent=2)
            
            print(f"   ✅ {style}({style_data['style_english']}): {style_filename}")
    
    def generate_training_format(self, dataset: Dict) -> List[Dict]:
        """
        生成最终的训练格式数据
        转换为分类模型训练所需的格式
        """
        print("\n🎯 生成最终训练格式...")
        
        training_data = []
        
        for style, style_data in dataset['styles'].items():
            for sample in style_data['samples']:
                # 组合用户问题和回答作为输入
                input_text = f"用户问题：{sample['user_query']}\n回答：{sample['agent_reply']}"
                
                # 创建训练样本
                training_sample = {
                    'input': input_text,
                    'label': style,
                    'label_english': style_data['style_english'],
                    'metadata': {
                        'source': sample['source'],
                        'scene': sample['scene'],
                        'turn_id': sample['turn_id']
                    }
                }
                training_data.append(training_sample)
        
        print(f"✅ 训练格式生成完成，共 {len(training_data)} 条训练样本")
        
        # 展示每个类别的样本数量
        label_counts = Counter([item['label'] for item in training_data])
        print("   标签分布:")
        for label, count in label_counts.items():
            print(f"     {label}: {count} 条")
            
        return training_data
    
    def save_results(self, dataset: Dict, training_data: List[Dict]):
        """保存处理结果到输出目录"""
        print("\n💾 保存处理结果...")
        
        # 保存完整数据集
        final_dataset_path = os.path.join(self.output_data_dir, "final_dataset.json")
        with open(final_dataset_path, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, ensure_ascii=False, indent=2)
        
        # 保存训练格式数据
        training_data_path = os.path.join(self.output_data_dir, "training_data.json")
        with open(training_data_path, 'w', encoding='utf-8') as f:
            json.dump(training_data, f, ensure_ascii=False, indent=2)
        
        # 保存各风格独立数据文件
        self.save_style_datasets(dataset)
        
        print("✅ 结果保存完成")
        print(f"   📄 final_dataset.json - 完整数据集")
        print(f"   📄 training_data.json - 训练格式数据") 
        print(f"   📁 style_*.json - 各风格独立数据文件")
        print(f"   📁 所有文件保存在: {self.output_data_dir}")
    
    def run_complete_pipeline(self):
        """运行完整的数据处理流水线"""
        print("🎬 开始完整数据处理流水线...")
        print("=" * 60)
        
        try:
            # 步骤1: 从文件加载业务数据
            business_data = self.load_business_data()
            
            # 步骤2: 从文件加载开源数据 
            open_source_data = self.load_opensource_data()
            
            # 步骤3: 场景标签映射
            mapped_business_data = self.map_scenes_to_styles(business_data)
            
            # 步骤4: 质量筛选
            filtered_business_data = self.quality_filtering(mapped_business_data, "business")
            filtered_open_source_data = self.quality_filtering(open_source_data, "opensource")
            
            # 步骤5: 对话拆分（仅针对业务数据）
            split_business_data = self.split_conversations(filtered_business_data)
            
            # 步骤6: 数据清洗
            cleaned_business_data = self.clean_and_rewrite_data(split_business_data, "business")
            cleaned_open_source_data = self.clean_and_rewrite_data(filtered_open_source_data, "opensource")
            
            # 步骤7: 去重处理
            dedup_business_data = self.deduplicate_data(cleaned_business_data, "business")
            dedup_open_source_data = self.deduplicate_data(cleaned_open_source_data, "opensource")
            
            # 步骤8: 整合最终数据集
            final_dataset = self.integrate_final_dataset(dedup_business_data, dedup_open_source_data)
            
            # 生成训练格式
            training_data = self.generate_training_format(final_dataset)
            
            # 保存结果
            self.save_results(final_dataset, training_data)
            
            print("\n" + "=" * 60)
            print("🎉 数据处理流水线完成！")
            print("=" * 60)
            
            return final_dataset, training_data
            
        except Exception as e:
            print(f"❌ 处理过程中出现错误: {str(e)}")
            raise e


def main():
    """主函数"""
    # 使用相对路径初始化处理器
    processor = RewardModelDataProcessor()
    final_dataset, training_data = processor.run_complete_pipeline()
    
    # 展示一些样本结果
    print("\n📋 训练数据样本展示:")
    print("-" * 50)
    
    # 展示每种风格的一个样本
    shown_labels = set()
    for sample in training_data:
        label = sample['label']
        if label not in shown_labels:
            print(f"\n🏷️  标签: {label} ({sample['label_english']})")
            print(f"📝 输入: {sample['input'][:100]}...")
            print(f"🏪 来源: {sample['metadata']['source']}")
            shown_labels.add(label)
            
        if len(shown_labels) >= 5:  # 展示前5种不同的标签
            break
    
    print(f"\n📊 数据集统计:")
    print(f"   总样本数: {len(training_data)}")
    print(f"   标签类别数: {len(set([item['label'] for item in training_data]))}")
    
    # 展示文件结构
    print(f"\n📁 输出文件结构:")
    print(f"   {processor.output_data_dir}/")
    print(f"   ├── final_dataset.json")
    print(f"   ├── training_data.json")
    for style, style_data in final_dataset['styles'].items():
        style_english = style_data['style_english']
        print(f"   └── style_{style_english}.json ({style})")


if __name__ == "__main__":
    main()
