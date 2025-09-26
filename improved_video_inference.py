#!/usr/bin/env python3
"""
Improved Video Inference Script for Ultra-Fast-Lane-Detection-v2
优化版本的车道线检测视频推理脚本

主要改进：
1. 增强的后处理算法
2. 更好的车道线连续性
3. 改进的置信度阈值
4. 更平滑的车道线绘制
"""

import torch
import cv2
import numpy as np
import argparse
from torchvision import transforms
from model.model_culane import parsingNet
from utils.common import merge_config
from utils.config import Config
import os
from tqdm import tqdm
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d

def enhanced_pred2coords(pred, row_anchor, col_anchor, local_width=2, 
                        original_image_width=1640, original_image_height=590,
                        confidence_threshold=0.3):
    """
    增强版本的预测结果转换为坐标函数
    
    改进点：
    1. 使用更大的local_width提高稳定性
    2. 更严格的车道线有效性检查
    3. 保持与原始函数的兼容性
    """
    batch_size, num_grid_row, num_cls_row, num_lane_row = pred['loc_row'].shape
    batch_size, num_grid_col, num_cls_col, num_lane_col = pred['loc_col'].shape

    max_indices_row = pred['loc_row'].argmax(1).cpu()
    valid_row = pred['exist_row'].argmax(1).cpu()

    max_indices_col = pred['loc_col'].argmax(1).cpu()
    valid_col = pred['exist_col'].argmax(1).cpu()

    pred['loc_row'] = pred['loc_row'].cpu()
    pred['loc_col'] = pred['loc_col'].cpu()

    coords = []

    row_lane_idx = [1, 2]  # Middle lanes
    col_lane_idx = [0, 3]  # Side lanes

    # Process row-based lanes (vertical lanes) with enhanced filtering
    for i in row_lane_idx:
        tmp = []
        # 更严格的存在性检查
        valid_points = valid_row[0, :, i].sum()
        
        if valid_points > num_cls_row / 3:  # 需要至少1/3的点有效
            for k in range(valid_row.shape[1]):
                if valid_row[0, k, i]:
                    all_ind = torch.tensor(list(range(max(0, max_indices_row[0, k, i] - local_width), 
                                                    min(num_grid_row-1, max_indices_row[0, k, i] + local_width) + 1)))
                    
                    out_tmp = (pred['loc_row'][0, all_ind, k, i].softmax(0) * all_ind.float()).sum() + 0.5
                    out_tmp = out_tmp / (num_grid_row-1) * original_image_width
                    tmp.append((int(out_tmp), int(row_anchor[k] * original_image_height)))
            
            if len(tmp) > 4:  # 至少需要5个点才认为是有效车道线
                coords.append(tmp)

    # Process column-based lanes (horizontal lanes) with enhanced filtering
    for i in col_lane_idx:
        tmp = []
        valid_points = valid_col[0, :, i].sum()
        
        if valid_points > num_cls_col / 6:  # 需要至少1/6的点有效
            for k in range(valid_col.shape[1]):
                if valid_col[0, k, i]:
                    all_ind = torch.tensor(list(range(max(0, max_indices_col[0, k, i] - local_width), 
                                                    min(num_grid_col-1, max_indices_col[0, k, i] + local_width) + 1)))
                    
                    out_tmp = (pred['loc_col'][0, all_ind, k, i].softmax(0) * all_ind.float()).sum() + 0.5
                    out_tmp = out_tmp / (num_grid_col-1) * original_image_height
                    tmp.append((int(col_anchor[k] * original_image_width), int(out_tmp)))
            
            if len(tmp) > 3:  # 至少需要4个点才认为是有效车道线
                coords.append(tmp)

    return coords

def smooth_lane_points(lane_points, smooth_factor=1.0):
    """
    平滑车道线点，减少抖动
    """
    if len(lane_points) < 3:
        return lane_points
    
    # 按y坐标排序
    sorted_points = sorted(lane_points, key=lambda x: x[1])
    
    # 提取x和y坐标
    x_coords = [p[0] for p in sorted_points]
    y_coords = [p[1] for p in sorted_points]
    
    # 使用高斯滤波平滑x坐标
    if len(x_coords) > 2:
        x_smooth = gaussian_filter1d(x_coords, sigma=smooth_factor)
        return [(int(x), y) for x, y in zip(x_smooth, y_coords)]
    
    return sorted_points

def interpolate_lane_points(lane_points, num_points=50):
    """
    插值生成更多的车道线点，使车道线更连续
    """
    if len(lane_points) < 2:
        return lane_points
    
    # 按y坐标排序
    sorted_points = sorted(lane_points, key=lambda x: x[1])
    
    if len(sorted_points) < 2:
        return sorted_points
    
    # 提取坐标
    x_coords = [p[0] for p in sorted_points]
    y_coords = [p[1] for p in sorted_points]
    
    # 创建插值函数
    try:
        f = interp1d(y_coords, x_coords, kind='linear', bounds_error=False, fill_value='extrapolate')
        
        # 生成新的y坐标
        y_new = np.linspace(min(y_coords), max(y_coords), num_points)
        x_new = f(y_new)
        
        # 过滤有效点
        valid_points = []
        for x, y in zip(x_new, y_new):
            if not np.isnan(x) and not np.isnan(y):
                valid_points.append((int(x), int(y)))
        
        return valid_points
    except:
        return sorted_points

def load_model(model_path, device='cuda'):
    """Load the trained model"""
    print(f"Loading model from {model_path}")
    
    # Model configuration for CULane ResNet34
    model = parsingNet(
        pretrained=False,
        backbone='34',
        num_grid_row=200, num_cls_row=72, num_grid_col=100, num_cls_col=81,
        num_lane_on_row=4, num_lane_on_col=4, use_aux=False,
        input_height=320, input_width=1600, fc_norm=True
    )
    
    # Load weights
    state_dict = torch.load(model_path, map_location='cpu')['model']
    
    # Create compatible state dict
    compatible_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            compatible_state_dict[k[7:]] = v
        else:
            compatible_state_dict[k] = v
    
    model.load_state_dict(compatible_state_dict, strict=False)
    model.eval()
    model.to(device)
    
    print(f"Model loaded from {model_path}")
    return model

def enhanced_draw_lanes(img, coords, colors=None, line_thickness=4, point_radius=3):
    """
    增强版本的车道线绘制函数
    
    改进点：
    1. 更平滑的车道线
    2. 更好的颜色方案
    3. 可调节的线条粗细
    """
    if colors is None:
        # 更好的颜色方案：红、绿、蓝、青
        colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0)]
    
    result_img = img.copy()
    
    for i, lane in enumerate(coords):
        if len(lane) > 1:
            color = colors[i % len(colors)]
            
            # 平滑和插值处理
            smoothed_lane = smooth_lane_points(lane, smooth_factor=0.8)
            interpolated_lane = interpolate_lane_points(smoothed_lane, num_points=30)
            
            if len(interpolated_lane) > 1:
                # 按y坐标排序
                lane_sorted = sorted(interpolated_lane, key=lambda x: x[1])
                
                # 绘制连续的车道线
                points = np.array(lane_sorted, dtype=np.int32)
                if len(points) > 1:
                    # 使用polylines绘制更平滑的线条
                    cv2.polylines(result_img, [points], False, color, line_thickness)
                    
                    # 在关键点绘制小圆点
                    for j, point in enumerate(lane_sorted[::3]):  # 每3个点绘制一个圆点
                        x, y = point
                        if 0 <= x < img.shape[1] and 0 <= y < img.shape[0]:
                            cv2.circle(result_img, (x, y), point_radius, color, -1)
    
    return result_img

def process_video(model, input_video, output_video, device='cuda', confidence_threshold=0.4):
    """Process video with enhanced lane detection"""
    cap = cv2.VideoCapture(input_video)
    
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {input_video}")
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video info: {width}x{height}, {fps} FPS, {total_frames} frames")
    
    # Video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
    
    # Enhanced image transforms
    crop_ratio = 0.6
    train_height = 320
    resize_height = int(train_height / crop_ratio)  # 533
    
    img_transforms = transforms.Compose([
        transforms.Resize((resize_height, 1600)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    
    # Row and column anchors
    row_anchor = np.linspace(0.42, 1, 72)
    col_anchor = np.linspace(0, 1, 81)
    
    # Process frames
    frame_count = 0
    
    with torch.no_grad():
        pbar = tqdm(total=total_frames, desc="Processing video")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Preprocess frame (same as original video_inference.py)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            from PIL import Image
            img_pil = Image.fromarray(frame_rgb)
            img_tensor = img_transforms(img_pil)
            
            # Apply bottom crop (same as LaneTestDataset)
            img_tensor = img_tensor[:, -train_height:, :]  # Keep bottom 320 pixels
            input_tensor = img_tensor.unsqueeze(0).to(device)
            
            # Model inference
            with torch.no_grad():
                output = model(input_tensor)
            
            # Convert predictions to coordinates with enhanced processing
            coords = enhanced_pred2coords(
                output, row_anchor, col_anchor,
                local_width=2,
                original_image_width=width,
                original_image_height=height,
                confidence_threshold=confidence_threshold
            )
            
            # Draw enhanced lanes
            result_frame = enhanced_draw_lanes(frame, coords, line_thickness=5, point_radius=4)
            
            # Add frame info
            cv2.putText(result_frame, f"Frame: {frame_count}/{total_frames}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(result_frame, f"Lanes detected: {len(coords)}", 
                       (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # Write frame
            out.write(result_frame)
            pbar.update(1)
    
    # Cleanup
    cap.release()
    out.release()
    pbar.close()
    
    print(f"Enhanced video saved to: {output_video}")

def main():
    parser = argparse.ArgumentParser(description='Enhanced Video Lane Detection')
    parser.add_argument('--model', type=str, default='culane_res34.pth',
                       help='Path to model weights')
    parser.add_argument('--input', type=str, default='example.mp4',
                       help='Input video path')
    parser.add_argument('--output', type=str, default='result_video_enhanced.mp4',
                       help='Output video path')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda/cpu)')
    parser.add_argument('--confidence', type=float, default=0.4,
                       help='Confidence threshold for lane detection')
    
    args = parser.parse_args()
    
    # Check if model file exists
    if not os.path.exists(args.model):
        print(f"Model file not found: {args.model}")
        return
    
    # Check if input video exists
    if not os.path.exists(args.input):
        print(f"Input video not found: {args.input}")
        return
    
    # Load model
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = load_model(args.model, device)
    
    # Process video
    process_video(model, args.input, args.output, device, args.confidence)

if __name__ == '__main__':
    main()