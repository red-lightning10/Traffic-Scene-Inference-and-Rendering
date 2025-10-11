"""
Lane Detection Utilities

Core utilities for lane detection, Bézier curve extraction, and data processing.
This module contains the essential logic for converting lane masks into renderable data.
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class LaneProcessor:
    """
    Processes lane detection results and extracts Bézier curves for rendering.
    """
    
    def __init__(self, bezier_points_solid: int = 5, bezier_points_dotted: int = 2):
        """
        Initialize lane processor.
        
        Args:
            bezier_points_solid: Number of Bézier points for solid lines
            bezier_points_dotted: Number of Bézier points for dotted lines
        """
        self.bezier_points_solid = bezier_points_solid
        self.bezier_points_dotted = bezier_points_dotted
        
        # Lane type mappings
        self.lane_types = {
            'solid-line': self.bezier_points_solid,
            'divider-line': self.bezier_points_solid,
            'dotted-line': self.bezier_points_dotted
        }
    
    def centroid(self, mask: np.ndarray) -> Tuple[int, int]:
        """
        Calculate centroid of a binary mask.
        
        Args:
            mask: Binary mask array
            
        Returns:
            Tuple of (x, y) centroid coordinates
        """
        coords = np.where(mask)
        if len(coords[0]) == 0:
            return (0, 0)
        
        centroid_x = int(np.mean(coords[1]))
        centroid_y = int(np.mean(coords[0]))
        return (centroid_x, centroid_y)
    
    def extract_bezier_points(self, mask: np.ndarray, bbox: np.ndarray, 
                             lane_type: str) -> List[List[int]]:
        """
        Extract Bézier curve points from lane mask.
        
        Args:
            mask: Binary mask of the lane
            bbox: Bounding box coordinates [x1, y1, x2, y2]
            lane_type: Type of lane ('solid-line', 'dotted-line', 'divider-line')
            
        Returns:
            List of Bézier points [[x, y], ...]
        """
        num_points = self.lane_types.get(lane_type, self.bezier_points_solid)
        
        # Convert bbox to corner points
        if len(bbox) == 4:
            # bbox format: [x1, y1, x2, y2]
            x1, y1, x2, y2 = bbox
            top_left = (int(x1), int(y1))
            bottom_right = (int(x2), int(y2))
        else:
            # bbox format: [(x1, y1), (x2, y2)]
            top_left = (int(bbox[0][0]), int(bbox[0][1]))
            bottom_right = (int(bbox[1][0]), int(bbox[1][1]))
        
        box_height = abs(bottom_right[1] - top_left[1])
        bottom_y_line = top_left[1] + box_height * 0.05
        
        bezier_points = []
        
        for i in range(num_points):
            temp_y = int(abs(bottom_y_line + (0.9 / (num_points - 1)) * i * box_height))
            
            # Ensure y coordinate is within mask bounds
            if temp_y >= mask.shape[0]:
                temp_y = mask.shape[0] - 1
            if temp_y < 0:
                temp_y = 0
            
            # Find lane pixels at this y level
            indices = np.where(mask[temp_y, :])[0]
            
            if indices.size > 0:
                temp_x = int(np.mean(indices))
                bezier_points.append([temp_x, temp_y])
            else:
                # If no pixels found, use previous point or skip
                if bezier_points:
                    bezier_points.append(bezier_points[-1])
        
        return bezier_points
    
    def process_lane_detections(self, masks: List[np.ndarray], boxes: List[np.ndarray], 
                              labels: List[str]) -> Dict[str, List]:
        """
        Process lane detection results and organize by lane type.
        
        Args:
            masks: List of lane masks
            boxes: List of bounding boxes
            labels: List of lane labels
            
        Returns:
            Dictionary with organized lane data
        """
        solid_beziers = []
        dotted_beziers = []
        divider_beziers = []
        
        for i, (mask, box, label) in enumerate(zip(masks, boxes, labels)):
            try:
                bezier_points = self.extract_bezier_points(mask, box, label)
                
                if label == 'solid-line':
                    solid_beziers.append(bezier_points)
                elif label == 'dotted-line':
                    dotted_beziers.append(bezier_points)
                elif label == 'divider-line':
                    divider_beziers.append(bezier_points)
                else:
                    logger.warning(f"Unknown lane type: {label}")
                    
            except Exception as e:
                logger.error(f"Error processing lane {i}: {e}")
                continue
        
        return {
            'solid-line': solid_beziers,
            'dotted-line': dotted_beziers,
            'divider-line': divider_beziers,
            'all-lines': solid_beziers + dotted_beziers + divider_beziers
        }
    
    def create_visualization(self, image: np.ndarray, masks: List[np.ndarray], 
                           boxes: List[np.ndarray], labels: List[str],
                           bezier_data: Dict[str, List]) -> np.ndarray:
        """
        Create visualization image with lane detections and Bézier points.
        
        Args:
            image: Original image
            masks: List of lane masks
            boxes: List of bounding boxes
            labels: List of lane labels
            bezier_data: Processed Bézier data
            
        Returns:
            Visualization image
        """
        vis_image = image.copy()
        
        # Color mapping for different lane types
        colors = {
            'solid-line': (255, 0, 0),      # Red
            'dotted-line': (0, 255, 0),      # Green
            'divider-line': (0, 0, 255),    # Blue
        }
        
        # Draw bounding boxes and centroids
        for i, (mask, box, label) in enumerate(zip(masks, boxes, labels)):
            color = colors.get(label, (128, 128, 128))
            
            # Draw bounding box
            cv2.rectangle(vis_image, 
                         (int(box[0]), int(box[1])), 
                         (int(box[2]), int(box[3])), 
                         color, 2)
            
            # Draw centroid
            centroid = self.centroid(mask)
            cv2.circle(vis_image, centroid, radius=5, color=color, thickness=2)
        
        # Draw Bézier points
        for lane_type, bezier_list in bezier_data.items():
            color = colors.get(lane_type, (128, 128, 128))
            
            for bezier_points in bezier_list:
                for point in bezier_points:
                    cv2.circle(vis_image, tuple(point), radius=3, 
                             color=color, thickness=2)
        
        return vis_image


def validate_lane_data(lane_data: Dict[str, List]) -> bool:
    """
    Validate lane detection data.
    
    Args:
        lane_data: Dictionary containing lane data
        
    Returns:
        True if data is valid, False otherwise
    """
    required_keys = ['solid-line', 'dotted-line', 'divider-line']
    
    for key in required_keys:
        if key not in lane_data:
            logger.error(f"Missing required lane type: {key}")
            return False
        
        if not isinstance(lane_data[key], list):
            logger.error(f"Invalid data type for {key}: expected list")
            return False
    
    return True


def format_for_blender(lane_data: Dict[str, List]) -> Dict[str, any]:
    """
    Format lane data for Blender integration.
    
    Args:
        lane_data: Processed lane data
        
    Returns:
        Formatted data for Blender
    """
    blender_data = {
        'lanes': {
            'solid_lines': lane_data['solid-line'],
            'dotted_lines': lane_data['dotted-line'],
            'divider_lines': lane_data['divider-line']
        },
        'metadata': {
            'total_solid': len(lane_data['solid-line']),
            'total_dotted': len(lane_data['dotted-line']),
            'total_divider': len(lane_data['divider-line']),
            'total_lanes': len(lane_data['all-lines'])
        }
    }
    
    return blender_data
