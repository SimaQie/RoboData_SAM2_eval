基于您的安装方法，我重新编写一版完整的README：

# SAM2 语义分割与绿幕背景替换

## 项目描述

本项目基于 Meta 的 SAM2 (Segment Anything Model 2) 模型，实现机器人桌面操作场景的语义分割，并自动将背景替换为绿幕。特别适用于机器人操作数据的预处理和背景标准化。

## 功能特性

- ✅ 使用 SAM2 进行高质量的语义分割
- ✅ 基于面积和颜色一致性的自动掩码选择
- ✅ 智能绿幕背景替换
- ✅ 支持多种硬件加速（CUDA、MPS、CPU）
- ✅ 详细的调试信息和可视化输出

## 快速开始

### 1. 克隆仓库
```bash
git clone <repository-url>
cd sam2-background-replacement
```

### 2. 安装依赖
```bash
pip install -e .
```

### 3. 下载模型
```bash
cd checkpoints && \
./download_ckpts.sh && \
cd ..
```

### 4. 运行示例
```bash
python sam2_background_replacement.py
```

## 项目结构

```
.
├── setup.py                    # 包安装配置
├── sam2_background_replacement.py    # 主程序文件
├── checkpoints/
│   ├── download_ckpts.sh       # 模型下载脚本
│   └── sam2.1_hiera_large.pt   # SAM2 模型权重（下载后）
├── configs/
│   └── sam2.1/
│       └── sam2.1_hiera_l.yaml # 模型配置文件
├── notebooks/
│   └── images/
│       └── plate_hand.jpg      # 示例输入图像
└── scripts/
    └── batch_process.py        # 批量处理脚本（可选）
```

## 使用方法

### 基本使用

直接运行主程序文件处理默认图像：

```bash
python sam2_background_replacement.py
```

### 处理自定义图像

修改代码中的图像路径：
```python
image = Image.open('./notebooks/images/your_image.jpg')
```

### 批量处理

处理整个目录的图像：
```bash
python scripts/batch_process.py --input-dir ./input_images/ --output-dir ./output_results/
```

## 输出文件

程序运行后会生成以下文件：

- `segmentation_result.png` - 完整的语义分割结果可视化
- `green_screen_result.jpg` - 绿幕背景替换后的图像
- 控制台输出详细的掩码选择信息

## 算法原理

### 掩码自动选择策略

程序使用基于面积和颜色一致性的评分系统自动选择需要保留的前景掩码：

**综合评分：** `总分 = 面积占比 × 0.9 + （1- 颜色标准差/255）* 0.1

### 绿幕替换逻辑

选中的掩码区域替换为绿幕背景，其他区域保留原图：
```python
result = np.where(foreground_mask, green_background, original_image)
```

## 配置说明

### 关键参数调整

在 `auto_select_masks_by_area_and_color` 函数中可调整：

- `num_to_select`: 选择掩码的数量（默认：2）
- 面积评分阈值
- 颜色一致性评分阈值

### 硬件配置

程序自动检测可用硬件：
- 优先使用 CUDA (NVIDIA GPU)
- 其次使用 MPS (Apple Silicon)
- 最后使用 CPU

## 示例输出

运行程序后，控制台会显示类似以下信息：

```
=== 自动选择详情 ===
✓ 掩码  2: 面积比率=0.0731, 颜色标准差=13.28, 面积得分=3, 颜色得分=3, 总分=33
✓ 掩码  0: 面积比率=0.0065, 颜色标准差=15.14, 面积得分=1, 颜色得分=3, 总分=13
✗ 掩码  1: 面积比率=0.5465, 颜色标准差=15.85, 面积得分=4, 颜色得分=3, 总分=43
```

## 依赖说明

通过 `pip install -e .` 安装的依赖包括：

- `torch` - PyTorch深度学习框架
- `torchvision` - 计算机视觉工具
- `opencv-python` - 图像处理
- `matplotlib` - 可视化
- `pillow` - 图像处理
- `numpy` - 数值计算

## 故障排除

### 常见问题

1. **模型下载失败**
   ```bash
   # 手动检查网络连接，或使用代理
   cd checkpoints
   wget https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2.1_hiera_large.pt
   ```

2. **CUDA内存不足**
   ```python
   # 在代码中减小图像尺寸或批处理大小
   image = image.resize((512, 512))
   ```

3. **依赖冲突**
   ```bash
   pip install --force-reinstall torch torchvision
   ```

4. **权限问题**
   ```bash
   chmod +x checkpoints/download_ckpts.sh
   ```

## 开发指南

### 扩展功能

要添加新的掩码选择策略，可以继承或修改 `auto_select_masks_by_area_and_color` 函数：

```python
def custom_mask_selector(masks, image):
    # 实现自定义选择逻辑
    # 返回选中的掩码索引列表
    return selected_indices
```

### 测试

运行测试用例：
```bash
python -m pytest tests/
```

## 许可证

本项目基于 SAM2 模型的许可证。请确保遵守：

- SAM2: [Apache 2.0 License](https://github.com/facebookresearch/sam2/blob/main/LICENSE)
- 本项目代码: MIT License

