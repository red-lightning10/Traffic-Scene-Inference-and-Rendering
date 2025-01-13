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
from PIL import Image
from infer_utils import draw_segmentation_map, get_outputs
from torchvision.transforms import transforms as transforms
from class_names import INSTANCE_CATEGORY_NAMES as class_names


def centroid(mask):
    coords = np.where(mask)
    centroid = int(np.mean(coords[1])), int(np.mean(coords[0]))
    return centroid


def solid_lane_bezier(mask, bound_box, num):
    '''
    Takes mask and bounding box for solid lane, and then returns 'num' bezier points that lie on the solid lane.
    '''
    box_height = abs(bound_box[1][1] - bound_box[0][1])

    bottom_y_line = bound_box[0][1] + box_height * 0.05

    bezier = []
    for i in range(num):
        temp_y = int(abs(bottom_y_line + (0.9 / (num - 1)) * i * box_height))
        # print('\ny = ', temp_y)
        
        indices = np.where(mask[temp_y, :])[0]  # Getting indices where the condition is True
        if indices.size > 0:  # Checking if any indices were found
            temp_x = int(np.mean(indices))
            bezier.append([temp_x, temp_y])

    return bezier

    

parser = argparse.ArgumentParser()
parser.add_argument(
    '-i', 
    '--input', 
    required=True, 
    help='path to the input data'
)
parser.add_argument(
    '-t', 
    '--threshold', 
    default=0.5, 
    type=float,
    help='score threshold for discarding detection'
)
parser.add_argument(
    '-w',
    '--weights',
    default='out/checkpoint.pth',
    help='path to the trained wieght file'
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

OUT_DIR = os.path.join('outputs', 'inference')
print(OUT_DIR)
os.makedirs(OUT_DIR, exist_ok=True)

model = torchvision.models.detection.maskrcnn_resnet50_fpn_v2(
    pretrained=False, num_classes=91
)

model.roi_heads.box_predictor.cls_score = nn.Linear(in_features=1024, out_features=len(class_names), bias=True)
model.roi_heads.box_predictor.bbox_pred = nn.Linear(in_features=1024, out_features=len(class_names)*4, bias=True)
model.roi_heads.mask_predictor.mask_fcn_logits = nn.Conv2d(256, len(class_names), kernel_size=(1, 1), stride=(1, 1))

# initialize the model
ckpt = torch.load(args.weights,map_location=torch.device('cpu'))
model.load_state_dict(ckpt['model'])
# set the computation device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# load the modle on to the computation device and set to eval mode
model.to(device).eval()
#print(model)

# transform to convert the image to tensor
transform = transforms.Compose([
    transforms.ToTensor()
])




image_paths = glob.glob(os.path.join(args.input, '*.jpg'))
image_paths = sorted(image_paths)
for j, image_path in enumerate(image_paths):
    data = {}
    solid_beziers = []
    dotted_beziers = []
    lines = []
    image = Image.open(image_path)
    # keep a copy of the original image for OpenCV functions and applying masks
    orig_image = image.copy()
    
    # transform the image
    image = transform(image)
    # add a batch dimension
    image = image.unsqueeze(0).to(device)
    
    masks, boxes, labels = get_outputs(image, model, args.threshold)

    # print('\nMask shape ')
    # print(masks, '\nBos shape ', boxes.shape, '\n Labels shape ', labels.shape)

    temp = image.clone()
    temp = temp.cpu()
    temp = torch.squeeze(temp)
    # temp = temp.reshape(temp.shape[1], temp.shape[2], temp.shape[0])

    temp = temp.numpy().transpose((1,2,0)).copy()

    # temp = np.array(temp)
    boxes = np.array(boxes)
    # boxes = boxes.numpy()

    for i, coord in enumerate(boxes):
        cent = centroid(masks[i])
        #print('cent', cent)
        cv2.rectangle(temp, coord[0], coord[1], [0, 0, 255], thickness = 2)
        cv2.circle(temp, cent, radius = 5, color = [255, 0, 0], thickness = 2)

        if labels[i] == 'solid-line' or labels[i] == 'divider-line' :
            num = 5
        elif labels[i] == 'dotted-line':
            num = 2
        else:
            continue

        bezier = solid_lane_bezier(masks[i], coord, num = num)
        for bez in bezier:
            cv2.circle(temp, bez, radius = 5, color = [0, 255, 0], thickness = 2)
   

        if labels[i] == 'solid-line' or labels[i] == 'divider-line' :
            solid_beziers.append(bezier)

        elif labels[i] == 'dotted-line':
            dotted_beziers.append(bezier)

        else:
            print(' another class ', labels[i])    

    data['dotted-line'] = dotted_beziers
    data['solid-line'] = solid_beziers
    lines.append(solid_beziers)
    lines.append(dotted_beziers)

    # plot_result = cv2.rectangle(temp, boxes[0], boxes[1], [0, 0, 255], thickness = 5)
    cv2.imshow('plot_result', temp)

    result = draw_segmentation_map(orig_image, masks, boxes, labels, args)
    
    # visualize the image
    if args.show:
        cv2.imshow('Segmented image', np.array(result))
        # time.sleep(3)
        cv2.waitKey(0)
    

    # set the save path
    save_path = "/home/adhi/20230814_Lane_Detection_using_Mask_RCNN_An_Instance_Segmentation_Approach/outputs/inference/"
    cv2.imwrite(save_path+str(j+750)+'.jpg', result)

    if not os.path.exists(f'./data/{args.input.split("/")[-1]}'):
        os.makedirs(f'./data/{args.input.split("/")[-1]}')

    with open(f'./data/{args.input.split("/")[-1]}/{image_path.split("/")[-1]}.pkl', 'wb') as f:
        #f = open('/home/adhi/20230814_Lane_Detection_using_Mask_RCNN_An_Instance_Segmentation_Approach/outputs/inference/', 'wb')
        pickle.dump(data,f)
    with open("/home/adhi/file.txt", "w") as output:
        output.write(str(data))
    # with open('f') as f:
    #     pickle.load(f)
