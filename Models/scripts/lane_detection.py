"""
Lane Detection Script - Refactored Version

This script preserves the core lane detection logic while using proper utilities
and removing hardcoded paths. It focuses on the essential Bézier curve extraction
for Blender rendering.
"""

import torch
import torchvision
import cv2
import argparse
import numpy as np
import torch.nn as nn
import glob
import os
import pickle
import time
import json
from pathlib import Path

from PIL import Image
from torchvision.transforms import transforms as transforms
from lane_utils import LaneProcessor, validate_lane_data, format_for_blender

# Define class names directly
INSTANCE_CATEGORY_NAMES = [
    '__background__',  # 0
    'solid-line',      # 1 - Solid lane lines
    'dotted-line',     # 2 - Dotted lane lines  
    'divider-line',    # 3 - Divider lines
]


def get_outputs(image, model, threshold):
    """Get model outputs for lane detection using Mask R-CNN."""
    with torch.no_grad():
        outputs = model(image)
    
    # Extract outputs
    scores = outputs[0]['scores'].cpu().numpy()
    labels = outputs[0]['labels'].cpu().numpy()
    boxes = outputs[0]['boxes'].cpu().numpy()
    masks = outputs[0]['masks'].cpu().numpy()
    
    # Filter by threshold
    keep = scores >= threshold
    
    filtered_masks = masks[keep]
    filtered_boxes = boxes[keep]
    filtered_labels = labels[keep]
    
    # Convert labels to class names
    class_names = ['solid-line', 'dotted-line', 'divider-line']
    label_names = [class_names[label - 1] if label <= len(class_names) else f'class_{label}' 
                   for label in filtered_labels]
    
    return filtered_masks, filtered_boxes, label_names


def draw_segmentation_map(image, masks, boxes, labels, args):
    """Draw segmentation map with bounding boxes and labels."""
    import cv2
    
    # Convert PIL to numpy array
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    # Create a copy for drawing
    result_image = image.copy()
    
    # Color map for different classes
    colors = {
        'solid-line': (255, 0, 0),      # Red
        'dotted-line': (0, 255, 0),      # Green
        'divider-line': (0, 0, 255),    # Blue
    }
    
    # Draw masks and bounding boxes
    for i, (mask, box, label) in enumerate(zip(masks, boxes, labels)):
        # Get color for this class
        color = colors.get(label, (128, 128, 128))
        
        # Draw bounding box
        if not args.no_boxes:
            cv2.rectangle(result_image, 
                         (int(box[0]), int(box[1])), 
                         (int(box[2]), int(box[3])), 
                         color, 2)
        
        # Draw mask
        mask_3d = np.stack([mask[0]] * 3, axis=2)
        mask_3d = (mask_3d * 255).astype(np.uint8)
        
        # Apply mask with transparency
        mask_area = mask_3d > 0
        result_image[mask_area] = cv2.addWeighted(
            result_image[mask_area], 0.7, 
            mask_3d[mask_area], 0.3, 0
        )
        
        # Draw label
        if not args.no_boxes:
            cv2.putText(result_image, label, 
                       (int(box[0]), int(box[1]) - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    return result_image


def main():
    """
    Main lane detection processing function.
    """
    parser = argparse.ArgumentParser(description='Lane Detection using Mask R-CNN')
    
    parser.add_argument(
        '-i', '--input', 
        required=True, 
        help='path to the input data directory'
    )
    
    parser.add_argument(
        '-t', '--threshold', 
        default=0.5, 
        type=float,
        help='score threshold for discarding detection'
    )
    
    parser.add_argument(
        '-w', '--weights',
        default='../weights/lane_detection.pth',
        help='path to the trained weight file'
    )
    
    parser.add_argument(
        '-o', '--output',
        default='outputs/lane_detection',
        help='path to output directory'
    )
    
    parser.add_argument(
        '--show',
        action='store_true',
        help='whether to visualize the results in real-time on screen'
    )
    
    parser.add_argument(
        '--no-boxes',
        action='store_true',
        help='do not show bounding boxes, only show segmentation map'
    )
    
    args = parser.parse_args()
    
    # Create output directory
    OUT_DIR = Path(args.output)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {OUT_DIR}")
    
    # Initialize lane processor
    lane_processor = LaneProcessor()
    
    # Load and configure model
    model = torchvision.models.detection.maskrcnn_resnet50_fpn_v2(
        pretrained=False, 
        num_classes=91
    )
    
    model.roi_heads.box_predictor.cls_score = nn.Linear(
        in_features=1024, 
        out_features=len(class_names), 
        bias=True
    )
    
    model.roi_heads.box_predictor.bbox_pred = nn.Linear(
        in_features=1024, 
        out_features=len(class_names) * 4, 
        bias=True
    )
    
    model.roi_heads.mask_predictor.mask_fcn_logits = nn.Conv2d(
        256, 
        len(class_names), 
        kernel_size=(1, 1), 
        stride=(1, 1)
    )
    
    # Load weights
    if os.path.exists(args.weights):
        ckpt = torch.load(args.weights, map_location=torch.device('cpu'))
        model.load_state_dict(ckpt['model'])
        print(f"Loaded weights from: {args.weights}")
    else:
        print(f"Warning: Weights file not found: {args.weights}")
        print("Using untrained model - results may be poor")
    
    # Set device and eval mode
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device).eval()
    print(f"Using device: {device}")
    
    # Setup transforms
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    
    # Find image files
    image_paths = glob.glob(os.path.join(args.input, '*.jpg'))
    image_paths = sorted(image_paths)
    
    if not image_paths:
        print(f"No .jpg files found in {args.input}")
        return
    
    print(f"Found {len(image_paths)} images to process")
    
    # Process each image
    for j, image_path in enumerate(image_paths):
        print(f"Processing {j+1}/{len(image_paths)}: {os.path.basename(image_path)}")
        
        try:
            # Load image
            image = Image.open(image_path)
            orig_image = image.copy()
            
            # Transform image
            image_tensor = transform(image)
            image_tensor = image_tensor.unsqueeze(0).to(device)
            
            # Get model outputs
            masks, boxes, labels = get_outputs(image_tensor, model, args.threshold)
            
            # Process lane detections using utilities
            lane_data = lane_processor.process_lane_detections(masks, boxes, labels)
            
            # Validate data
            if not validate_lane_data(lane_data):
                print(f"Warning: Lane data validation failed for {image_path}")
            
            # Format for Blender
            blender_data = format_for_blender(lane_data)
            
            # Create visualization
            temp = image_tensor.clone()
            temp = temp.cpu()
            temp = torch.squeeze(temp)
            temp = temp.numpy().transpose((1, 2, 0)).copy()
            
            boxes = np.array(boxes)
            
            # Draw bounding boxes and centroids
            for i, coord in enumerate(boxes):
                cent = lane_processor.centroid(masks[i])
                cv2.rectangle(temp, coord[0], coord[1], [0, 0, 255], thickness=2)
                cv2.circle(temp, cent, radius=5, color=[255, 0, 0], thickness=2)
            
            # Draw Bézier points
            vis_image = lane_processor.create_visualization(
                orig_image, masks, boxes, labels, lane_data
            )
            
            cv2.imshow('plot_result', temp)
            
            # Create segmentation result
            result = draw_segmentation_map(orig_image, masks, boxes, labels, args)
            
            # Visualize the image
            if args.show:
                cv2.imshow('Segmented image', np.array(result))
                cv2.waitKey(0)
            
            # Save results with proper paths
            image_name = Path(image_path).stem
            
            # Save segmentation image
            result_path = OUT_DIR / f"{image_name}_segmentation.jpg"
            cv2.imwrite(str(result_path), result)
            
            # Save visualization
            vis_path = OUT_DIR / f"{image_name}_visualization.jpg"
            cv2.imwrite(str(vis_path), cv2.cvtColor(np.array(vis_image), cv2.COLOR_RGB2BGR))
            
            # Save pickle data
            pickle_path = OUT_DIR / f"{image_name}.pkl"
            with open(pickle_path, 'wb') as f:
                pickle.dump(blender_data, f)
            
            # Save debug data
            debug_path = OUT_DIR / f"{image_name}_debug.txt"
            with open(debug_path, "w") as f:
                f.write(f"Image: {image_path}\n")
                f.write(f"Detected lanes: {blender_data['metadata']}\n")
                f.write(f"Lane data: {json.dumps(blender_data, indent=2)}\n")
            
            print(f"Saved results for {image_name}")
            
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            continue
    
    print(f"Processing complete! Results saved to: {OUT_DIR}")


if __name__ == '__main__':
    main()
