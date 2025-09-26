#!/usr/bin/env python3
"""
Simple Lane Detection Inference Script (Fixed Version)
Performs lane detection on single images using pretrained model
"""

import torch
import cv2
import numpy as np
import argparse
from torchvision import transforms
from PIL import Image
from model.model_culane import parsingNet

def pred2coords(pred, row_anchor, col_anchor, local_width=1, original_image_width=1640, original_image_height=590):
    """Convert model predictions to lane coordinates (Fixed version from demo.py)"""
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

    # Process row-based lanes (vertical lanes)
    for i in row_lane_idx:
        tmp = []
        if valid_row[0, :, i].sum() > num_cls_row / 2:
            for k in range(valid_row.shape[1]):
                if valid_row[0, k, i]:
                    all_ind = torch.tensor(list(range(max(0, max_indices_row[0, k, i] - local_width), 
                                                    min(num_grid_row-1, max_indices_row[0, k, i] + local_width) + 1)))
                    
                    out_tmp = (pred['loc_row'][0, all_ind, k, i].softmax(0) * all_ind.float()).sum() + 0.5
                    out_tmp = out_tmp / (num_grid_row-1) * original_image_width
                    tmp.append((int(out_tmp), int(row_anchor[k] * original_image_height)))
            coords.append(tmp)

    # Process column-based lanes (horizontal lanes)
    for i in col_lane_idx:
        tmp = []
        if valid_col[0, :, i].sum() > num_cls_col / 4:
            for k in range(valid_col.shape[1]):
                if valid_col[0, k, i]:
                    all_ind = torch.tensor(list(range(max(0, max_indices_col[0, k, i] - local_width), 
                                                    min(num_grid_col-1, max_indices_col[0, k, i] + local_width) + 1)))
                    
                    out_tmp = (pred['loc_col'][0, all_ind, k, i].softmax(0) * all_ind.float()).sum() + 0.5
                    out_tmp = out_tmp / (num_grid_col-1) * original_image_height
                    tmp.append((int(col_anchor[k] * original_image_width), int(out_tmp)))
            coords.append(tmp)

    return coords

def load_model(model_path, device='cuda'):
    """Load the pretrained model"""
    print(f"Loading model from {model_path}")
    
    # Model configuration for CULane ResNet34 (from config file)
    net = parsingNet(pretrained=False, backbone='34', 
                    num_grid_row=200, num_cls_row=72, num_grid_col=100, num_cls_col=81,
                    num_lane_on_row=4, num_lane_on_col=4, use_aux=False,
                    input_height=320, input_width=1600, fc_norm=True)
    
    # Load the pretrained weights
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

def draw_lanes(img, coords, colors=None):
    """Draw lane lines on image"""
    if colors is None:
        colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0)]
    
    result_img = img.copy()
    
    for i, lane in enumerate(coords):
        if len(lane) > 1:  # Need at least 2 points to draw a line
            color = colors[i % len(colors)]
            
            # Sort points by y coordinate for proper line drawing
            lane_sorted = sorted(lane, key=lambda x: x[1])
            
            # Draw circles for each point
            for point in lane_sorted:
                x, y = point
                if 0 <= x < img.shape[1] and 0 <= y < img.shape[0]:
                    cv2.circle(result_img, (x, y), 4, color, -1)
            
            # Draw lines connecting the points
            for j in range(len(lane_sorted) - 1):
                pt1 = lane_sorted[j]
                pt2 = lane_sorted[j + 1]
                if (0 <= pt1[0] < img.shape[1] and 0 <= pt1[1] < img.shape[0] and
                    0 <= pt2[0] < img.shape[1] and 0 <= pt2[1] < img.shape[0]):
                    cv2.line(result_img, pt1, pt2, color, 3)
    
    return result_img

def process_image(model, image_path, output_path, device='cuda'):
    """Process a single image"""
    
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Cannot load image: {image_path}")
    
    height, width = img.shape[:2]
    print(f"Image size: {width}x{height}")
    
    # Image transforms (same as training with crop_ratio)
    crop_ratio = 0.6
    train_height = 320
    resize_height = int(train_height / crop_ratio)  # 533
    
    img_transforms = transforms.Compose([
        transforms.Resize((resize_height, 1600)),  # Resize to (533, 1600)
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    
    # Prepare input
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    img_tensor = img_transforms(img_pil)
    
    # Apply bottom crop (same as LaneTestDataset)
    img_tensor = img_tensor[:, -train_height:, :]  # Keep bottom 320 pixels
    img_tensor = img_tensor.unsqueeze(0).to(device)
    
    # Row and column anchors (from CULane config)
    row_anchor = np.linspace(0.42, 1, 72)  # 72 row anchors
    col_anchor = np.linspace(0, 1, 81)     # 81 col anchors
    
    # Inference
    with torch.no_grad():
        out_net = model(img_tensor)
    
    # Convert predictions to coordinates
    coords = pred2coords(out_net, row_anchor, col_anchor, 
                        original_image_width=width, original_image_height=height)
    
    # Draw lanes on image
    result_img = draw_lanes(img, coords)
    
    # Save result
    cv2.imwrite(output_path, result_img)
    
    print(f"Result saved to {output_path}")
    print(f"Detected {len(coords)} lanes")
    for i, lane in enumerate(coords):
        print(f"Lane {i+1}: {len(lane)} points")
    
    return coords

def main():
    parser = argparse.ArgumentParser(description='Simple Lane Detection Inference (Fixed)')
    parser.add_argument('--model', required=True, help='Path to pretrained model')
    parser.add_argument('--image', required=True, help='Input image path')
    parser.add_argument('--output', required=True, help='Output image path')
    parser.add_argument('--device', default='cuda', help='Device to use (cuda/cpu)')
    
    args = parser.parse_args()
    
    # Check if CUDA is available
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = 'cpu'
    
    print(f"Running inference on {args.image}")
    
    # Load model
    model = load_model(args.model, args.device)
    
    # Process image
    coords = process_image(model, args.image, args.output, args.device)

if __name__ == '__main__':
    main()