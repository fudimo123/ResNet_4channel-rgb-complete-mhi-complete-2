import os
import cv2
import numpy as np
from smic_data_parser import SMICDataParser
from dynamic_image_generator import DynamicImageGenerator

# 配置
RAW_VIDEO_DIR = '../data/SMIC2'
DYNAMIC_IMAGE_DIR = './smic_dynamic_data2'

# SMIC情感映射
EMOTION_MAP = {
    'positive': 0,
    'negative': 1,
    'surprise': 2
}

def load_image_sequence(sequence_path, image_files):
    """加载图像序列"""
    images = []
    for img_file in image_files:
        img_path = os.path.join(sequence_path, img_file)
        img = cv2.imread(img_path)
        if img is not None:
            images.append(img)
    return images

def main():
    # 初始化数据解析器和动态图像生成器
    data_parser = SMICDataParser(RAW_VIDEO_DIR, EMOTION_MAP)
    dynamic_generator = DynamicImageGenerator(DYNAMIC_IMAGE_DIR)
    
    # 获取所有样本
    samples = data_parser.get_samples()
    
    print(f"找到 {len(samples)} 个micro表情样本")
    
    generated_count = 0
    
    for sample in samples:
        subject = sample['subject']
        sequence = sample['sequence']
        emotion = sample['emotion']
        sequence_path = sample['raw_video_path']
        image_files = sample['image_files']
        
        print(f"处理: 受试者{subject}, 序列{sequence}, 情感{emotion}")
        
        # 加载图像序列
        image_sequence = load_image_sequence(sequence_path, image_files)
        
        if len(image_sequence) == 0:
            print(f"  警告: 序列 {sequence} 没有有效图像")
            continue
            
        # 生成动态图像
        dynamic_image = dynamic_generator.generate_dynamic_image(image_sequence)
        
        if dynamic_image is not None:
            # 保存动态图像
            dynamic_generator.save_dynamic_image(subject, sequence, dynamic_image)
            generated_count += 1
            print(f"  成功生成动态图像: {generated_count}")
        else:
            print(f"  失败: 无法生成动态图像")
    
    print(f"\n总共生成了 {generated_count} 个动态图像")
    print(f"动态图像保存在: {DYNAMIC_IMAGE_DIR}")

if __name__ == '__main__':
    main()