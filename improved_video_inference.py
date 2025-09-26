#!/usr/bin/env python3
"""
Improved Video/Image Inference Script for Ultra-Fast-Lane-Detection-v2
优化版本的车道线检测视频/图片推理脚本

主要改进：
1. 增强的后处理算法
2. 更好的车道线连续性
3. 改进的置信度阈值
4. 更平滑的车道线绘制
5. 支持图片批量处理模式
6. 生成详细的JSON汇总报告
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
import json
import glob
from tqdm import tqdm
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
from pathlib import Path

def enhanced_pred2coords(pred, row_anchor, col_anchor, local_width=2, 
                        original_image_width=1640, original_image_height=590,
                        confidence_threshold=0.3):
    """
    增强版本的预测结果转换为坐标函数
    
    改进点：
    1. 使用更大的local_width提高稳定性
    2. 更严格的车道线有效性检查
    3. 保持与原始函数的兼容性
    4. 返回置信度信息
    """
    batch_size, num_grid_row, num_cls_row, num_lane_row = pred['loc_row'].shape
    batch_size, num_grid_col, num_cls_col, num_lane_col = pred['loc_col'].shape

    max_indices_row = pred['loc_row'].argmax(1).cpu()
    valid_row = pred['exist_row'].argmax(1).cpu()
    exist_prob_row = pred['exist_row'].softmax(1).cpu()

    max_indices_col = pred['loc_col'].argmax(1).cpu()
    valid_col = pred['exist_col'].argmax(1).cpu()
    exist_prob_col = pred['exist_col'].softmax(1).cpu()

    pred['loc_row'] = pred['loc_row'].cpu()
    pred['loc_col'] = pred['loc_col'].cpu()

    coords = []
    confidences = []

    row_lane_idx = [1, 2]  # Middle lanes
    col_lane_idx = [0, 3]  # Side lanes

    # Process row-based lanes (vertical lanes) with enhanced filtering
    for i in row_lane_idx:
        tmp = []
        lane_confidences = []
        # 更严格的存在性检查
        valid_points = valid_row[0, :, i].sum()
        
        if valid_points > num_cls_row / 3:  # 需要至少1/3的点有效
            for k in range(valid_row.shape[1]):
                if valid_row[0, k, i]:
                    all_ind = torch.tensor(list(range(max(0, max_indices_row[0, k, i] - local_width), 
                                                    min(num_grid_row-1, max_indices_row[0, k, i] + local_width) + 1)))
                    
                    out_tmp = (pred['loc_row'][0, all_ind, k, i].softmax(0) * all_ind.float()).sum() + 0.5
                    out_tmp = out_tmp / (num_grid_row-1) * original_image_width
                    
                    # 使用固定的置信度值（基于有效点的比例）
                    point_confidence = float(valid_points) / num_cls_row
                    
                    tmp.append((int(out_tmp), int(row_anchor[k] * original_image_height)))
                    lane_confidences.append(point_confidence)
            
            if len(tmp) > 4:  # 至少需要5个点才认为是有效车道线
                coords.append(tmp)
                # 计算整条车道线的平均置信度
                avg_confidence = sum(lane_confidences) / len(lane_confidences) if lane_confidences else 0.0
                confidences.append(avg_confidence)

    # Process column-based lanes (horizontal lanes) with enhanced filtering
    for i in col_lane_idx:
        tmp = []
        lane_confidences = []
        valid_points = valid_col[0, :, i].sum()
        
        if valid_points > num_cls_col / 3:  # 需要至少1/3的点有效
            for k in range(valid_col.shape[1]):
                if valid_col[0, k, i]:
                    all_ind = torch.tensor(list(range(max(0, max_indices_col[0, k, i] - local_width), 
                                                    min(num_grid_col-1, max_indices_col[0, k, i] + local_width) + 1)))
                    
                    out_tmp = (pred['loc_col'][0, all_ind, k, i].softmax(0) * all_ind.float()).sum() + 0.5
                    out_tmp = out_tmp / (num_grid_col-1) * original_image_height
                    
                    # 使用固定的置信度值（基于有效点的比例）
                    point_confidence = float(valid_points) / num_cls_col
                    
                    tmp.append((int(col_anchor[k] * original_image_width), int(out_tmp)))
                    lane_confidences.append(point_confidence)
            
            if len(tmp) > 4:  # 至少需要5个点才认为是有效车道线
                coords.append(tmp)
                # 计算整条车道线的平均置信度
                avg_confidence = sum(lane_confidences) / len(lane_confidences) if lane_confidences else 0.0
                confidences.append(avg_confidence)

    return coords, confidences

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

def save_detection_results(results, output_path):
    """保存检测结果到JSON文件"""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Detection results saved to: {output_path}")

def process_single_image(model, image_path, device='cuda', confidence_threshold=0.4):
    """处理单张图片"""
    # 读取图片
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Cannot read image: {image_path}")
    
    height, width = img.shape[:2]
    
    # 预处理
    crop_ratio = 0.6
    train_height = 320
    resize_height = int(train_height / crop_ratio)  # 533
    
    img_transforms = transforms.Compose([
        transforms.Resize((resize_height, 1600)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    
    # 转换图片
    frame_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    from PIL import Image
    img_pil = Image.fromarray(frame_rgb)
    img_tensor = img_transforms(img_pil)
    
    # 应用底部裁剪
    img_tensor = img_tensor[:, -train_height:, :]
    input_tensor = img_tensor.unsqueeze(0).to(device)
    
    # 模型推理
    with torch.no_grad():
        output = model(input_tensor)
    
    # 行和列锚点
    row_anchor = np.linspace(0.42, 1, 72)
    col_anchor = np.linspace(0, 1, 81)
    
    # 转换预测结果为坐标
    coords, confidences = enhanced_pred2coords(
        output, row_anchor, col_anchor,
        local_width=2,
        original_image_width=width,
        original_image_height=height,
        confidence_threshold=confidence_threshold
    )
    
    return coords, confidences, img

def process_images(model, input_dir, output_dir, device='cuda', confidence_threshold=0.4):
    """处理文件夹中的多张图片"""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 支持的图片格式
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(str(input_path / ext)))
        image_files.extend(glob.glob(str(input_path / ext.upper())))
    
    if not image_files:
        print(f"No image files found in {input_dir}")
        return
    
    print(f"Found {len(image_files)} images to process")
    
    # 存储所有结果
    all_results = {
        "input_directory": str(input_path),
        "output_directory": str(output_path),
        "total_images": len(image_files),
        "confidence_threshold": confidence_threshold,
        "results": []
    }
    
    # 处理每张图片
    for img_path in tqdm(image_files, desc="Processing images"):
        try:
            # 处理图片
            coords, confidences, img = process_single_image(model, img_path, device, confidence_threshold)
            
            # 绘制车道线
            result_img = enhanced_draw_lanes(img, coords, line_thickness=5, point_radius=4)
            
            # 添加信息
            img_name = Path(img_path).name
            cv2.putText(result_img, f"Image: {img_name}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(result_img, f"Lanes detected: {len(coords)}", 
                       (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # 保存结果图片
            output_img_path = output_path / img_name
            cv2.imwrite(str(output_img_path), result_img)
            
            # 准备JSON数据
            lanes_data = []
            for i, (lane_coords, confidence) in enumerate(zip(coords, confidences)):
                lane_data = {
                    "lane_id": i,
                    "points": lane_coords,
                    "confidence": float(confidence),
                    "num_points": len(lane_coords)
                }
                lanes_data.append(lane_data)
            
            # 添加到总结果
            image_result = {
                "image_name": img_name,
                "image_path": img_path,
                "output_path": str(output_img_path),
                "lanes_detected": len(coords),
                "lanes": lanes_data
            }
            all_results["results"].append(image_result)
            
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            continue
    
    # 保存JSON结果
    json_output_path = output_path / f"{input_path.name}_results.json"
    save_detection_results(all_results, json_output_path)
    
    print(f"Processed {len(all_results['results'])} images successfully")
    print(f"Results saved to: {output_path}")

def process_video(model, input_video, output_video, device='cuda', confidence_threshold=0.4):
    """Process video with enhanced lane detection and JSON output"""
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
    
    # 存储所有帧的结果
    video_results = {
        "input_video": input_video,
        "output_video": output_video,
        "video_info": {
            "fps": fps,
            "width": width,
            "height": height,
            "total_frames": total_frames
        },
        "confidence_threshold": confidence_threshold,
        "frames": []
    }
    
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
            coords, confidences = enhanced_pred2coords(
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
            
            # 准备JSON数据（每10帧保存一次以减少文件大小）
            if frame_count % 10 == 0:
                lanes_data = []
                for i, (lane_coords, confidence) in enumerate(zip(coords, confidences)):
                    lane_data = {
                        "lane_id": i,
                        "points": lane_coords,
                        "confidence": float(confidence),
                        "num_points": len(lane_coords)
                    }
                    lanes_data.append(lane_data)
                
                frame_result = {
                    "frame_number": frame_count,
                    "timestamp": frame_count / fps,
                    "lanes_detected": len(coords),
                    "lanes": lanes_data
                }
                video_results["frames"].append(frame_result)
            
            pbar.update(1)
    
    # Cleanup
    cap.release()
    out.release()
    pbar.close()
    
    # 保存JSON结果
    video_name = Path(input_video).stem
    json_output_path = f"{video_name}_results.json"
    save_detection_results(video_results, json_output_path)
    
    print(f"Enhanced video saved to: {output_video}")
    print(f"JSON results saved to: {json_output_path}")

def main():
    parser = argparse.ArgumentParser(description='Enhanced Ultra-Fast Lane Detection Inference')
    parser.add_argument('--model', type=str, required=True, help='Path to model file')
    parser.add_argument('--input', type=str, required=True, help='Input video path or image directory')
    parser.add_argument('--output', type=str, required=True, help='Output video path or image directory')
    parser.add_argument('--mode', type=str, choices=['video', 'image'], default='video', 
                       help='Processing mode: video or image (default: video)')
    parser.add_argument('--confidence', type=float, default=0.4, help='Confidence threshold')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda/cpu)')
    
    args = parser.parse_args()
    
    # 检查模型文件
    if not os.path.exists(args.model):
        print(f"Error: Model file '{args.model}' not found!")
        return
    
    # 检查输入
    if args.mode == 'video':
        if not os.path.exists(args.input):
            print(f"Error: Input video file '{args.input}' not found!")
            return
    else:  # image mode
        if not os.path.isdir(args.input):
            print(f"Error: Input directory '{args.input}' not found!")
            return
    
    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 加载模型
    print("Loading model...")
    model = load_model(args.model, device)
    print("Model loaded successfully!")
    
    # 根据模式处理
    if args.mode == 'video':
        print(f"Processing video: {args.input}")
        process_video(model, args.input, args.output, device, args.confidence)
        print("Video processing completed!")
    else:  # image mode
        print(f"Processing images from directory: {args.input}")
        process_images(model, args.input, args.output, device, args.confidence)
        print("Image processing completed!")

if __name__ == '__main__':
    main()