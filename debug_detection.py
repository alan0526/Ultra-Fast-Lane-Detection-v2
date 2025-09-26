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

def debug_pred2coords(pred, row_anchor, col_anchor, local_width=1, original_image_width=1640, original_image_height=590):
    """调试版本的坐标转换函数 - 基于demo.py的实现"""
    print(f"\n=== 调试坐标转换过程 ===")
    print(f"预测输出类型: {type(pred)}")
    print(f"预测输出键: {pred.keys()}")
    print(f"原始图像尺寸: {original_image_width}x{original_image_height}")
    
    batch_size, num_grid_row, num_cls_row, num_lane_row = pred['loc_row'].shape
    batch_size, num_grid_col, num_cls_col, num_lane_col = pred['loc_col'].shape
    
    print(f"行预测形状: {pred['loc_row'].shape}")
    print(f"列预测形状: {pred['loc_col'].shape}")
    print(f"行存在性形状: {pred['exist_row'].shape}")
    print(f"列存在性形状: {pred['exist_col'].shape}")

    max_indices_row = pred['loc_row'].argmax(1).cpu()
    # n , num_cls, num_lanes
    valid_row = pred['exist_row'].argmax(1).cpu()
    # n, num_cls, num_lanes

    max_indices_col = pred['loc_col'].argmax(1).cpu()
    # n , num_cls, num_lanes
    valid_col = pred['exist_col'].argmax(1).cpu()
    # n, num_cls, num_lanes

    pred['loc_row'] = pred['loc_row'].cpu()
    pred['loc_col'] = pred['loc_col'].cpu()

    coords = []

    row_lane_idx = [1,2]  # 中间车道线
    col_lane_idx = [0,3]  # 边缘车道线
    
    print(f"处理行车道线索引: {row_lane_idx}")
    print(f"处理列车道线索引: {col_lane_idx}")

    for i in row_lane_idx:
        print(f"\n--- 处理行车道线 {i} ---")
        tmp = []
        valid_sum = valid_row[0,:,i].sum()
        threshold = num_cls_row / 2
        print(f"有效点数: {valid_sum}, 阈值: {threshold}")
        
        if valid_sum > threshold:
            print(f"车道线 {i} 通过阈值检查")
            for k in range(valid_row.shape[1]):
                if valid_row[0,k,i]:
                    all_ind = torch.tensor(list(range(max(0,max_indices_row[0,k,i] - local_width), min(num_grid_row-1, max_indices_row[0,k,i] + local_width) + 1)))
                    
                    out_tmp = (pred['loc_row'][0,all_ind,k,i].softmax(0) * all_ind.float()).sum() + 0.5
                    out_tmp = out_tmp / (num_grid_row-1) * original_image_width
                    y_coord = int(row_anchor[k] * original_image_height)
                    x_coord = int(out_tmp)
                    tmp.append((x_coord, y_coord))
            
            print(f"车道线 {i} 检测到 {len(tmp)} 个点")
            if len(tmp) > 0:
                coords.append(tmp)
        else:
            print(f"车道线 {i} 未通过阈值检查")

    for i in col_lane_idx:
        print(f"\n--- 处理列车道线 {i} ---")
        tmp = []
        valid_sum = valid_col[0,:,i].sum()
        threshold = num_cls_col / 4
        print(f"有效点数: {valid_sum}, 阈值: {threshold}")
        
        if valid_sum > threshold:
            print(f"车道线 {i} 通过阈值检查")
            for k in range(valid_col.shape[1]):
                if valid_col[0,k,i]:
                    all_ind = torch.tensor(list(range(max(0,max_indices_col[0,k,i] - local_width), min(num_grid_col-1, max_indices_col[0,k,i] + local_width) + 1)))
                    
                    out_tmp = (pred['loc_col'][0,all_ind,k,i].softmax(0) * all_ind.float()).sum() + 0.5

                    out_tmp = out_tmp / (num_grid_col-1) * original_image_height
                    x_coord = int(col_anchor[k] * original_image_width)
                    y_coord = int(out_tmp)
                    tmp.append((x_coord, y_coord))
            
            print(f"车道线 {i} 检测到 {len(tmp)} 个点")
            if len(tmp) > 0:
                coords.append(tmp)
        else:
            print(f"车道线 {i} 未通过阈值检查")

    print(f"\n总共检测到 {len(coords)} 条车道线")
    for i, lane in enumerate(coords):
        print(f"车道线 {i+1}: {len(lane)} 个点")
        if lane:
            print(f"  起点: {lane[0]}, 终点: {lane[-1]}")

    return coords

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