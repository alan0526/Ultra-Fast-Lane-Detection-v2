#!/usr/bin/env python3
"""
详细的车道线检测调试脚本
分析模型输出和坐标转换过程
"""

import argparse
import torch
import cv2
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from model.model_culane import parsingNet
import matplotlib.pyplot as plt

def load_model(model_path, device='cuda'):
    """加载预训练模型"""
    print(f"Loading model from {model_path}")
    
    # 模型配置
    net = parsingNet(pretrained=False, backbone='34', 
                    num_grid_row=200, num_cls_row=72, num_grid_col=100, num_cls_col=81,
                    num_lane_on_row=4, num_lane_on_col=4, use_aux=False,
                    input_height=320, input_width=1600, fc_norm=True)
    
    # 加载权重
    state_dict = torch.load(model_path, map_location='cpu')['model']
    
    # Create compatible state dict
    compatible_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            compatible_state_dict[k[7:]] = v
        else:
            compatible_state_dict[k] = v
    
    net.load_state_dict(compatible_state_dict, strict=False)
    
    net.eval()
    net.to(device)
    return net

def debug_pred2coords(pred_dict, row_anchor, col_anchor, original_image_width, original_image_height):
    """调试版本的坐标转换函数"""
    print(f"预测输出类型: {type(pred_dict)}")
    print(f"预测输出键: {pred_dict.keys()}")
    print(f"原始图像尺寸: {original_image_width}x{original_image_height}")
    
    # 获取行预测
    loc_row = pred_dict['loc_row']  # [batch_size, num_grid_row, num_cls_row, num_lane_row]
    exist_row = pred_dict['exist_row']  # [batch_size, 2, num_cls_row, num_lane_row]
    
    print(f"行位置预测形状: {loc_row.shape}")
    print(f"行存在预测形状: {exist_row.shape}")
    
    batch_size, num_grid_row, num_cls_row, num_lane_row = loc_row.shape
    print(f"批次大小: {batch_size}, 网格行数: {num_grid_row}, 分类行数: {num_cls_row}, 车道行数: {num_lane_row}")
    
    # 获取最大概率的位置
    max_indices = torch.argmax(loc_row, dim=2)  # [batch_size, num_grid_row, num_lane_row]
    print(f"最大索引形状: {max_indices.shape}")
    
    # 获取存在概率
    exist_prob = torch.softmax(exist_row, dim=1)[:, 1, :, :]  # [batch_size, num_cls_row, num_lane_row]
    print(f"存在概率形状: {exist_prob.shape}")
    
    # 转换为坐标
    lanes = []
    for lane_idx in range(num_lane_row):
        lane_points = []
        for row_idx in range(num_grid_row):
            col_idx = max_indices[0, row_idx, lane_idx].item()
            if col_idx < num_cls_row - 1:  # 不是背景类
                # 检查存在概率 - 注意这里的索引应该是 [batch, row_anchor_idx, lane_idx]
                # 需要将 row_idx 映射到 row_anchor 索引
                if row_idx < len(row_anchor):
                    row_anchor_idx = row_idx * num_cls_row // num_grid_row  # 映射到锚点索引
                    if row_anchor_idx < exist_prob.shape[1]:
                        exist_score = exist_prob[0, row_anchor_idx, lane_idx].item()
                        if exist_score > 0.5:  # 存在阈值
                            # 计算实际坐标
                            y = row_anchor[row_anchor_idx] * original_image_height
                            x = col_anchor[col_idx] * original_image_width
                            lane_points.append((int(x), int(y)))
        
        if len(lane_points) > 0:
            lanes.append(lane_points)
            print(f"车道 {lane_idx}: {len(lane_points)} 个点")
    
    return lanes

def visualize_predictions(pred, row_anchor, col_anchor):
    """可视化模型预测结果"""
    print("\n=== 可视化预测结果 ===")
    
    # 创建图形
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    
    # 行预测的存在性
    exist_row = pred['exist_row'][0].cpu().numpy()  # (72, 4)
    for i in range(4):
        axes[0, i].plot(exist_row[:, i])
        axes[0, i].set_title(f'Row Lane {i} Existence')
        axes[0, i].set_xlabel('Grid Position')
        axes[0, i].set_ylabel('Existence Probability')
    
    # 列预测的存在性
    exist_col = pred['exist_col'][0].cpu().numpy()  # (81, 4)
    for i in range(4):
        axes[1, i].plot(exist_col[:, i])
        axes[1, i].set_title(f'Col Lane {i} Existence')
        axes[1, i].set_xlabel('Grid Position')
        axes[1, i].set_ylabel('Existence Probability')
    
    plt.tight_layout()
    plt.savefig('prediction_analysis.png', dpi=150, bbox_inches='tight')
    print("预测分析图保存为 prediction_analysis.png")
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Debug lane detection')
    parser.add_argument('--model', type=str, default='culane_res34.pth', help='Model path')
    parser.add_argument('--image', type=str, default='test_frame.jpg', help='Input image path')
    parser.add_argument('--output', type=str, default='debug_result.jpg', help='Output image path')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 加载模型
    model = load_model(args.model, device)
    
    # 图像预处理配置
    crop_ratio = 0.6
    train_height = 320
    resize_height = int(train_height / crop_ratio)  # 533
    
    print(f"\n=== 图像预处理配置 ===")
    print(f"裁剪比例: {crop_ratio}")
    print(f"训练高度: {train_height}")
    print(f"调整后高度: {resize_height}")
    
    # 图像变换
    img_transforms = transforms.Compose([
        transforms.Resize((resize_height, 1600)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    
    # 加载和预处理图像
    print(f"\n=== 加载图像 {args.image} ===")
    img_pil = Image.open(args.image).convert('RGB')
    original_width, original_height = img_pil.size
    print(f"原始图像尺寸: {original_width}x{original_height}")
    
    # 预处理
    img_tensor = img_transforms(img_pil)
    print(f"变换后张量形状: {img_tensor.shape}")
    
    # 关键步骤：底部裁剪
    img_tensor = img_tensor[:, -train_height:, :]
    print(f"裁剪后张量形状: {img_tensor.shape}")
    
    img_tensor = img_tensor.unsqueeze(0).to(device)
    print(f"最终输入形状: {img_tensor.shape}")
    
    # 锚点配置
    row_anchor = np.linspace(0.42, 1, 72)
    col_anchor = np.linspace(0, 1, 81)
    
    print(f"\n=== 锚点配置 ===")
    print(f"行锚点: {len(row_anchor)} 个，范围 {row_anchor[0]:.3f} - {row_anchor[-1]:.3f}")
    print(f"列锚点: {len(col_anchor)} 个，范围 {col_anchor[0]:.3f} - {col_anchor[-1]:.3f}")
    
    # 模型推理
    print(f"\n=== 模型推理 ===")
    with torch.no_grad():
        out_net = model(img_tensor)
    
    print(f"模型输出类型: {type(out_net)}")
    if isinstance(out_net, dict):
        for key, value in out_net.items():
            print(f"  {key}: {value.shape}")
    
    # 可视化预测结果
    visualize_predictions(out_net, row_anchor, col_anchor)
    
    # 调试坐标转换
    coords = debug_pred2coords(out_net, row_anchor, col_anchor, 
                              original_image_width=original_width, 
                              original_image_height=original_height)
    
    # 绘制结果
    print(f"\n=== 绘制结果 ===")
    img_cv = cv2.imread(args.image)
    colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0)]
    
    for i, lane in enumerate(coords):
        color = colors[i % len(colors)]
        print(f"绘制车道线 {i+1}，颜色: {color}，点数: {len(lane)}")
        
        for j, (x, y) in enumerate(lane):
            cv2.circle(img_cv, (x, y), 3, color, -1)
            if j > 0:
                cv2.line(img_cv, lane[j-1], (x, y), color, 2)
    
    cv2.imwrite(args.output, img_cv)
    print(f"调试结果保存到: {args.output}")
    
    print(f"\n=== 检测总结 ===")
    print(f"检测到 {len(coords)} 条车道线")
    for i, lane in enumerate(coords):
        print(f"车道线 {i+1}: {len(lane)} 个点")

if __name__ == '__main__':
    main()