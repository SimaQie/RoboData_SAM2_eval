import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
import cv2

# select the device for computation
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"using device: {device}")

if device.type == "cuda":
    # use bfloat16 for the entire notebook
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
elif device.type == "mps":
    print(
        "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
        "give numerically different outputs and sometimes degraded performance on MPS. "
        "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
    )
np.random.seed(3)

def show_anns(anns, borders=True):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:, :, 3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.5]])
        img[m] = color_mask 
        if borders:
            import cv2
            contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
            # Try to smooth contours
            contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
            cv2.drawContours(img, contours, -1, (0, 0, 1, 0.4), thickness=1) 

    ax.imshow(img)

def calculate_color_consistency(image, mask):
    """
    计算掩码区域的颜色一致性
    返回值越小表示颜色越一致
    """
    masked_region = image[mask]
    if len(masked_region) == 0:
        return float('inf')
    
    # 计算RGB三个通道的标准差，然后取平均值
    color_std = np.std(masked_region, axis=0)
    avg_std = np.mean(color_std)
    return avg_std

def auto_select_masks_by_area_and_color(masks, image, num_to_select=2):
    """
    基于面积和颜色一致性自动选择掩码
    优先考虑面积，其次考虑颜色一致性
    """
    total_pixels = image.shape[0] * image.shape[1]
    
    # 计算每个掩码的特征
    mask_scores = []
    
    for i, mask in enumerate(masks):
        area_ratio = mask['area'] / total_pixels
        
        # 面积得分：中等面积的掩码得分更高
        if area_ratio < 0.005:  # 太小，可能是噪声
            area_score = 0
        elif area_ratio < 0.02:  # 偏小
            area_score = 1
        elif area_ratio < 0.3:   # 理想范围
            area_score = 3
        elif area_ratio < 0.6:   # 偏大，可能是背景
            area_score = 4
        else:                    # 太大，很可能是背景
            area_score = 5
        
        # 颜色一致性得分：颜色越一致得分越高
        color_std = calculate_color_consistency(image, mask['segmentation'])
        if color_std < 15:       # 颜色非常一致
            color_score = 3
        elif color_std < 40:     # 颜色比较一致
            color_score = 2
        elif color_std < 60:     # 颜色一致性一般
            color_score = 1
        else:                    # 颜色不一致
            color_score = 0
        
        # 综合得分：优先面积，其次颜色一致性
        total_score = area_ratio * 0.9 + (1 - color_std / 255) * 0.1  # 面积和颜色标准化后加权
        mask_scores.append({
            'index': i,
            'area_ratio': area_ratio,
            'color_std': color_std,
            'total_score': total_score
        })
    
    # 按总分排序
    mask_scores.sort(key=lambda x: x['total_score'], reverse=True)
    
    # 选择得分最高的掩码
    selected_indices = [m['index'] for m in mask_scores[:num_to_select] if m['total_score'] > 0]
    
    # 打印选择详情
    print("\n=== 自动选择详情 ===")
    for i, m in enumerate(mask_scores[:5]):  # 显示前5个
        status = "✓" if m['index'] in selected_indices else "✗"
        print(f"{status} 掩码 {m['index']:2d}: 面积比率={m['area_ratio']:.4f}, "
              f"颜色标准差={m['color_std']:.2f}, 总分={m['total_score']}")
    
    return selected_indices


def replace_background_with_green_screen(original_image, masks, target_mask_indices):

    print(f"正在处理 {len(target_mask_indices)} 个选中的掩码...")
    
    # 创建前景掩码（选中的区域为1，背景为0）
    foreground_mask = np.zeros(original_image.shape[:2], dtype=bool)
    
    for idx in target_mask_indices:
        if 0 <= idx < len(masks):
            mask = masks[idx]['segmentation']
            foreground_mask = np.logical_or(foreground_mask, mask)
            print(f"  添加掩码 {idx}, 面积: {mask.sum()} 像素")
    
    print(f"前景掩码总面积: {foreground_mask.sum()} 像素")
    
    # 创建绿幕背景 
    green_background = np.zeros_like(original_image)
    green_background[:, :, 1] = 255 
    

    result_image = np.where(
        foreground_mask[:, :, np.newaxis],  # 条件：前景掩码
        green_background,                   # 假值：替换为绿幕
        original_image                     # 真值：保留原图
        
    )
    
    return result_image, foreground_mask.astype(np.uint8)


image = Image.open('./notebooks/images/plates.jpg')
image = np.array(image.convert("RGB"))
print(image.shape)

plt.figure(figsize=(20, 20))
plt.imshow(image)
plt.axis('off')
plt.show()

from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator


sam2_checkpoint = "./checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

sam2 = build_sam2(model_cfg, sam2_checkpoint, device=device, apply_postprocessing=False)

mask_generator = SAM2AutomaticMaskGenerator(sam2)
masks = mask_generator.generate(image)

print(len(masks))
print(masks[0].keys())


plt.figure(figsize=(20, 20))
plt.imshow(image)
show_anns(masks)
plt.axis('off')


plt.savefig('segmentation_result.png', bbox_inches='tight', pad_inches=0, dpi=100)
plt.close() 

print("分割结果已保存为 'segmentation_result.png'")


print(f"\n开始自动选择掩码...")
selected_indices = auto_select_masks_by_area_and_color(masks, image, num_to_select=2)

# if len(selected_indices) == 0:
#     print("警告：没有选择到合适的掩码，将使用面积最大的两个掩码")
#     # 按面积排序选择最大的两个
#     sorted_masks = sorted(masks, key=lambda x: x['area'], reverse=True)
#     selected_indices = [masks.index(sorted_masks[0]), masks.index(sorted_masks[1])]

print(f"\n最终选择的掩码索引: ")
print(selected_indices)

# 替换背景为绿幕
green_screen_result, foreground_mask = replace_background_with_green_screen(
    image, masks, selected_indices
)

green_screen_image = Image.fromarray(green_screen_result.astype(np.uint8))
green_screen_image.save('green_screen_result.jpg')
print("绿幕背景结果已保存为 'green_screen_result.jpg'")

# # 保存前景掩码
# mask_image = Image.fromarray(foreground_mask * 255)
# mask_image.save('foreground_mask.png')
# print("前景掩码已保存为 'foreground_mask.png'")

# # 显示选择结果的可视化
# plt.figure(figsize=(20, 20))
# plt.imshow(image)

# # 高亮显示被选中的掩码
# for i, mask_data in enumerate(masks):
#     mask = mask_data['segmentation']
#     if i in selected_indices:
#         # 绿色表示选中
#         color_mask = np.concatenate([[0, 1, 0], [0.7]])  # 绿色，半透明
#     else:
#         # 红色表示未选中
#         color_mask = np.concatenate([[1, 0, 0], [0.3]])  # 红色，较低透明度
    
#     img = np.ones((mask.shape[0], mask.shape[1], 4))
#     img[:, :, 3] = 0
#     img[mask] = color_mask
#     plt.imshow(img)

# plt.axis('off')
# plt.title(f'自动选择结果 (绿色:选中, 红色:未选中)')
# plt.savefig('auto_selection_visualization.png', bbox_inches='tight', pad_inches=0, dpi=100)
# plt.close()
# print("自动选择可视化结果已保存为 'auto_selection_visualization.png'")


# fig, axes = plt.subplots(1, 3, figsize=(30, 10))


# axes[0].imshow(image)
# axes[0].set_title('原始图像')
# axes[0].axis('off')

# axes[1].imshow(image)
# show_anns(masks)
# axes[1].set_title('语义分割结果')
# axes[1].axis('off')

# axes[2].imshow(green_screen_result)
# axes[2].set_title('绿幕背景替换')
# axes[2].axis('off')

# plt.tight_layout()
# plt.savefig('final_comparison.png', bbox_inches='tight', pad_inches=0, dpi=100)
# plt.close()
# print("最终对比结果已保存为 'final_comparison.png'")
