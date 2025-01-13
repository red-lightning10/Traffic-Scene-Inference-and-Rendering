import sys

sys.path.append('core')

import argparse
import os
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
    i=5
    for i in range(1,439):
        bb = np.genfromtxt('/home/adhi/Downloads/yolo_13/frame'+ str(i)+'.txt')
        bb = np.vstack((bb,[0,0,0,0,0]))
        bb = np.vstack((bb,[0,0,0,0,0]))
        np.savetxt('/home/adhi/Downloads/yolo_13/frame'+ str(i)+'.txt',bb)

def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)


def getF(imfi1e1, imfile2):
    sift = cv2.SIFT_create()
    image1 = cv2.imread(imfi1e1)
    image2 = cv2.imread(imfile2)
    keypoints1, descriptors1 = sift.detectAndCompute(image1,None)
    keypoints2, descriptors2 = sift.detectAndCompute(image2, None)

    matcher = cv2.BFMatcher()
    matches = matcher.knnMatch(descriptors1, descriptors2, k=2)

    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    points1 = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    points2 = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    F, mask = cv2.findFundamentalMat(points1, points2, cv2.FM_RANSAC, ransacReprojThreshold=1.0, confidence=0.99)
    print("Fundamental matrix:")
    return F


def viz(flo):
    flo = flo[0].permute(1, 2, 0).cpu().numpy()
    flo = flow_viz.flow_to_image(flo)
    print(np.shape(flo))
    return flo


def demo(args):
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model, map_location=torch.device('cpu')))

    model = model.module
    model.to(DEVICE)
    model.eval()
    with torch.no_grad():
        images = sorted(glob.glob(os.path.join(args.path, '*.png')) + \
                 glob.glob(os.path.join(args.path, '*.jpg')))

        images = sorted(images, key=lambda x: int(x.split('/')[-1][5:-4]))
        txt = sorted(glob.glob(os.path.join('/home/adhi/Downloads/yolo_13', '*.txt')))
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
            #x,y =np.shape(bb)
            i=i+1
            bounding_box=[]
            for b in bb:
                print(b)
                if b[0]==2 or b[0]==7:        
                    bounding_box.append(b[1:5])
            print(bounding_box)
            F = getF(imfile1, imfile2)
            sampson_distance(flow, bounding_box, imfile1,imfile2, F,i)
 


def calculate_movement(image1, image2, bbox, flow, F):

    flow_subset = flow[bbox[1]:bbox[3], bbox[0]:bbox[2]]
    height, width = flow_subset.shape[:2]
    corners = cv2.goodFeaturesToTrack(image1[bbox[1]:bbox[3], bbox[0]:bbox[2]], maxCorners=100, qualityLevel=0.01,
                                      minDistance=10)
    if corners is None:
        return False
    
    points1 = np.int0(corners).reshape(-1, 2)
    sampson_distances = []
    for point1 in points1:
        x1, y1 = point1
        flow_vec = flow_subset[y1, x1]
        expected_displacement = flow_vec
        x1_homog = np.array([x1, y1, 1])
        epipolar_line = np.dot(F, x1_homog)
        x2_expected = int(-epipolar_line[1]/epipolar_line[0])
        y2_expected = int(-epipolar_line[2]/epipolar_line[0])

        if x2_expected is not None and 0 <= x2_expected < width and 0 <= y2_expected < height:
            actual_displacement = [image2[y2_expected, x2_expected] - image1[y1, x1]]
            sampson_distance = np.linalg.norm(expected_displacement - actual_displacement) * 2 / (
                        expected_displacement[0] * 2)
            sampson_distances.append(sampson_distance)
    threshold = 1.5
    if len(sampson_distances) == 0:
        return True
	
    avg_sampson_distance = np.mean(sampson_distances)
    print("Average Sampson Distance: ", avg_sampson_distance)
    if avg_sampson_distance < threshold :
        return True 
    else:
        return False 

def sampson_distance(flow, bounding_boxes, imfile1,imfile2,F,j):

        image1 = cv2.imread(imfile1)
        image2 = cv2.imread(imfile2)
        img1_gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
        img2_gray = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
        output =[]
        for i in range(len(bounding_boxes)):
           x = int(bounding_boxes[i][0]*1280)
           y = int(bounding_boxes[i][1]*960)
           w = int(bounding_boxes[i][2]*1280)
           h = int(bounding_boxes[i][3]*960)
           x_min = int(x-w//2)
           y_min = int(y-h//2)
           x_max = int(x+w//2)
           y_max = int(y+h//2)
           bbox = (x_min, y_min, x_max, y_max)
           flow_field = flow
           out = calculate_movement(img1_gray, img2_gray, bbox, flow_field, F)
           #print("Sampson Distance: ", sampson_distance_out)
           threshold =2
           if out== True:
               label_moving = "Moving"
               label = 1
           else:
               label_moving = "Stationary"
               label = 0
           output.append(label)
           cv2.rectangle(image2, (int(x-w//2), int(y-h//2)), (int(x+w//2), int(y+h//2)), (0, 255, 0), 2)
           cv2.putText(image2, label_moving,  (int(x-w//2), int(y-h//2)), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
        cv2.imwrite('/home/adhi/RAFT/scene13/'+str(j)+'.jpg',image2)
        np.savetxt('/home/adhi/RAFT/scene13/'+str(j)+'.txt',output)


def compute_intersection_point(epipolar_line, width, height):
    intersection_points = []
    for line in epipolar_lines:
        point1, point2 = line[0]
        x1, y1 = point1
        x2, y2 = point2
        if y1 < 0 and y2 >= 0:
            intersect_x = x1 + (x2 - x1) * (-y1) / (y2 - y1)
            if 0 <= intersect_x <= width:
                intersection_points.append((intersect_x, 0))
        if y1 >= height and y2 < height:
            intersect_x = x1 + (x2 - x1) * (height - y1) / (y2 - y1)
            if 0 <= intersect_x <= width:
                intersection_points.append((intersect_x, height))
        if x1 < 0 and x2 >= 0:
            intersect_y = y1 + (y2 - y1) * (-x1) / (x2 - x1)
            if 0 <= intersect_y <= height:
                intersection_points.append((0, intersect_y))
        if x1 >= width and x2 < width:
            if 0 <= intersect_y <= height:
                intersectiframe1

    cv2.imshow('Result', frame)
    cv2.waitKey(0)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint")
    parser.add_argument('--path', help="dataset for evaluation")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    args = parser.parse_args()

    #load_txt(args)
    demo(args)
