import sys
import os

if '__file__' in globals():
    raft_core_path = os.path.join(os.path.dirname(__file__), 'RAFT', 'core')
else:
    raft_core_path = os.path.join(os.getcwd(), 'Models', 'RAFT', 'core')

sys.path.append(raft_core_path)

import argparse
import cv2
import glob
import numpy as np
import torch
from PIL import Image

from raft import RAFT
from utils import flow_viz
from utils.utils import InputPadder

DEVICE = 'cpu'

def load_txt(args):
    input_dir = args.input_dir
    output_dir = args.output_dir
    start_frame = args.start_frame
    end_frame = args.end_frame
    
    for i in range(start_frame, end_frame):
        input_file = os.path.join(input_dir, f'frame{i}.txt')
        
        if os.path.exists(input_file):
            bb = np.genfromtxt(input_file)
            bb = np.vstack((bb, [0, 0, 0, 0, 0]))
            bb = np.vstack((bb, [0, 0, 0, 0, 0]))
            
            output_file = os.path.join(output_dir, f'frame{i}.txt')
            np.savetxt(output_file, bb)

def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)


def getF(image1_path, image2_path):
    sift = cv2.SIFT_create()
    image1 = cv2.imread(image1_path)
    image2 = cv2.imread(image2_path)
    
    keypoints1, descriptors1 = sift.detectAndCompute(image1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(image2, None)

    matcher = cv2.BFMatcher()
    matches = matcher.knnMatch(descriptors1, descriptors2, k=2)

    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    points1 = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    points2 = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    F, mask = cv2.findFundamentalMat(points1, points2, cv2.FM_RANSAC, 
                                    ransacReprojThreshold=1.0, confidence=0.99)
    return F


def viz(flo):
    flo = flo[0].permute(1, 2, 0).cpu().numpy()
    flo = flow_viz.flow_to_image(flo)
    return flo


def demo(args):
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model, map_location=torch.device('cpu')))

    model = model.module
    model.to(DEVICE)
    model.eval()
    
    with torch.no_grad():
        images = sorted(glob.glob(os.path.join(args.path, '*.png')) + 
                       glob.glob(os.path.join(args.path, '*.jpg')))

        images = sorted(images, key=lambda x: int(x.split('/')[-1][5:-4]))
        txt = sorted(glob.glob(os.path.join(args.detection_dir, '*.txt')))
        txt = sorted(txt, key=lambda x: int(x.split('/')[-1][5:-4]))
        
        i = 0
        for imfile1, imfile2 in zip(images[:-1], images[1:]):
            image1 = load_image(imfile1)
            image2 = load_image(imfile2)
            
            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)
            
            flow_low, flow_up = model(image1, image2, iters=20, test_mode=True)
            flow = viz(flow_up)
            
            bb = np.loadtxt(txt[i])
            i = i + 1
            
            bounding_box = []
            for b in bb:
                if b[0] == 2 or b[0] == 7:        
                    bounding_box.append(b[1:5])
            
            F = getF(imfile1, imfile2)
            sampson_distance(flow, bounding_box, imfile1, imfile2, F, i, args)
 


def calculate_movement(image1, image2, bbox, flow, F):
    flow_subset = flow[bbox[1]:bbox[3], bbox[0]:bbox[2]]
    height, width = flow_subset.shape[:2]
    
    corners = cv2.goodFeaturesToTrack(image1[bbox[1]:bbox[3], bbox[0]:bbox[2]], 
                                      maxCorners=100, 
                                      qualityLevel=0.01,
                                      minDistance=10)
    
    if corners is None:
        return False
    
    points1 = np.int0(corners).reshape(-1, 2)
    sampson_distances = []
    for point1 in points1:
        x1_local, y1_local = point1
        # Convert to global image coordinates
        x1_global = x1_local + bbox[0]
        y1_global = y1_local + bbox[1]
        
        # Get flow vector at this point
        flow_vec = flow_subset[y1_local, x1_local]
        expected_displacement = flow_vec
        
        # Convert to homogeneous coordinates
        x1_homog = np.array([x1_global, y1_global, 1])
        
        # Compute epipolar line: l = F * x1
        epipolar_line = np.dot(F, x1_homog)
        
        # Skip if epipolar line is degenerate
        if abs(epipolar_line[0]) < 1e-6 and abs(epipolar_line[1]) < 1e-6:
            continue
            
        # For epipolar line ax + by + c = 0, find intersection with image boundaries
        # This is a simplified approach - in practice you'd find the actual epipolar line intersection
        # For now, we'll use the flow vector directly for motion validation
        
        # Calculate actual displacement from flow
        actual_displacement = flow_vec
        
        # Simple motion validation: check if flow magnitude is reasonable
        flow_magnitude = np.linalg.norm(actual_displacement)
        if flow_magnitude > 0.1:  # Minimum flow threshold
            sampson_distances.append(flow_magnitude)
    threshold = 2.0  # Flow magnitude threshold
    
    if len(sampson_distances) == 0:
        return False  # No motion detected
	
    avg_flow_magnitude = np.mean(sampson_distances)
    
    if avg_flow_magnitude > threshold:
        return True  # Object is moving
    else:
        return False  # Object is stationary 

def sampson_distance(flow, bounding_boxes, image1_path, image2_path, F, frame_id, args):
    
    image1 = cv2.imread(image1_path)
    image2 = cv2.imread(image2_path)
    
    img1_gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    
    output = []
    
    for i in range(len(bounding_boxes)):
        x = int(bounding_boxes[i][0] * 1280)
        y = int(bounding_boxes[i][1] * 960)
        
        w = int(bounding_boxes[i][2] * 1280)
        h = int(bounding_boxes[i][3] * 960)
        
        x_min = int(x - w // 2)
        y_min = int(y - h // 2)
        x_max = int(x + w // 2)
        y_max = int(y + h // 2)
        
        bbox = (x_min, y_min, x_max, y_max)
        
        is_moving = calculate_movement(img1_gray, img2_gray, bbox, flow, F)
        
        if is_moving:
            label_moving = "Moving"
            label = 1
        else:
            label_moving = "Stationary"
            label = 0
        
        output.append(label)
        
        # Draw bounding box and label
        cv2.rectangle(image2, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        cv2.putText(image2, label_moving, (x_min, y_min - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
    
    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    cv2.imwrite(os.path.join(args.output_dir, str(frame_id) + '.jpg'), image2)
    np.savetxt(os.path.join(args.output_dir, str(frame_id) + '.txt'), output)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint")
    parser.add_argument('--path', help="dataset for evaluation")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--input_dir', required=True, help='input directory for detections')
    parser.add_argument('--output_dir', required=True, help='output directory for results')
    parser.add_argument('--detection_dir', required=True, help='directory for detection files')
    parser.add_argument('--start_frame', type=int, required=True, help='start frame number')
    parser.add_argument('--end_frame', type=int, required=True, help='end frame number')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    args = parser.parse_args()

    #load_txt(args)
    demo(args)
