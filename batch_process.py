#!/usr/bin/env python3

import os
import argparse
import glob
from tqdm import tqdm
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from background_replacement import (
    device, build_sam2, SAM2AutomaticMaskGenerator,
    show_anns, calculate_color_consistency, 
    auto_select_masks_by_area_and_color, replace_background_with_green_screen
)
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
import cv2

def process_single_image(image_path, output_dir, sam2_model, mask_generator, num_masks=2):

    try:
        filename = os.path.basename(image_path)
        name_without_ext = os.path.splitext(filename)[0]
        
        print(f"\n处理图像: {filename}")

        image = Image.open(image_path)
        image_array = np.array(image.convert("RGB"))
        

        masks = mask_generator.generate(image_array)
        print(f"生成 {len(masks)} 个掩码")
        
   
        selected_indices = auto_select_masks_by_area_and_color(masks, image_array, num_to_select=num_masks)
        
        if len(selected_indices) == 0:
            print("警告：没有选择到合适的掩码，将使用面积最大的两个掩码")
            sorted_masks = sorted(masks, key=lambda x: x['area'], reverse=True)
            selected_indices = [i for i in range(min(2, len(masks)))]
        
        print(f"最终选择的掩码索引: {selected_indices}")
        
        # 替换背景为绿幕
        green_screen_result, foreground_mask = replace_background_with_green_screen(
            image_array, masks, selected_indices
        )
        

        green_screen_image = Image.fromarray(green_screen_result.astype(np.uint8))
        green_screen_output_path = os.path.join(output_dir, f"{name_without_ext}_green_screen.jpg")
        green_screen_image.save(green_screen_output_path)
        
        plt.figure(figsize=(20, 20))
        plt.imshow(image_array)
        show_anns(masks)
        plt.axis('off')
        segmentation_output_path = os.path.join(output_dir, f"{name_without_ext}_segmentation.png")
        plt.savefig(segmentation_output_path, bbox_inches='tight', pad_inches=0, dpi=100)
        plt.close()
        
        mask_image = Image.fromarray(foreground_mask * 255)
        mask_output_path = os.path.join(output_dir, f"{name_without_ext}_mask.png")
        mask_image.save(mask_output_path)
        
        print(f"结果已保存:")
        print(f"  - 绿幕图像: {green_screen_output_path}")
        print(f"  - 分割结果: {segmentation_output_path}")
        print(f"  - 前景掩码: {mask_output_path}")
        
        return True
        
    except Exception as e:
        print(f"处理图像 {image_path} 时出错: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='批量处理图像 - SAM2语义分割与绿幕背景替换')
    parser.add_argument('--input-dir', '-i', type=str, required=True,
                       help='输入图像目录路径')
    parser.add_argument('--output-dir', '-o', type=str, required=True,
                       help='输出结果目录路径')
    parser.add_argument('--num-masks', '-n', type=int, default=2,
                       help='每张图像选择的掩码数量 (默认: 2)')
    parser.add_argument('--extensions', '-e', type=str, nargs='+', 
                       default=['jpg', 'jpeg', 'png', 'bmp', 'tiff'],
                       help='支持的图像扩展名 (默认: jpg jpeg png bmp tiff)')
    
    args = parser.parse_args()
    
    # 检查输入目录是否存在
    if not os.path.exists(args.input_dir):
        print(f"错误: 输入目录不存在: {args.input_dir}")
        return
    

    os.makedirs(args.output_dir, exist_ok=True)
    
    image_files = []
    for ext in args.extensions:
        pattern = os.path.join(args.input_dir, f"*.{ext}")
        image_files.extend(glob.glob(pattern))
        pattern = os.path.join(args.input_dir, f"*.{ext.upper()}")
        image_files.extend(glob.glob(pattern))
    
    if not image_files:
        print(f"在目录 {args.input_dir} 中未找到图像文件")
        print(f"支持的格式: {', '.join(args.extensions)}")
        return
    
    print(f"找到 {len(image_files)} 个图像文件")
    print(f"输出目录: {args.output_dir}")
    print(f"每张图像选择掩码数量: {args.num_masks}")
    
    print("\n加载SAM2模型...")
    sam2_checkpoint = "./checkpoints/sam2.1_hiera_large.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
    
    if not os.path.exists(sam2_checkpoint):
        print(f"错误: 模型文件不存在: {sam2_checkpoint}")
        print("请先运行: cd checkpoints && ./download_ckpts.sh && cd ..")
        return
    
    sam2 = build_sam2(model_cfg, sam2_checkpoint, device=device, apply_postprocessing=False)
    mask_generator = SAM2AutomaticMaskGenerator(sam2)
    print("模型加载完成")
    
    success_count = 0
    failed_count = 0
    
    print(f"\n开始批量处理...")
    for image_path in tqdm(image_files, desc="处理进度"):
        success = process_single_image(
            image_path, args.output_dir, sam2, mask_generator, args.num_masks
        )
        
        if success:
            success_count += 1
        else:
            failed_count += 1
    

    print(f"\n批量处理完成!")
    print(f"成功处理: {success_count} 张图像")
    print(f"处理失败: {failed_count} 张图像")
    print(f"输出目录: {args.output_dir}")

if __name__ == "__main__":
    main()