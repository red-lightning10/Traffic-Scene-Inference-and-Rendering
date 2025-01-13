import numpy as np
import os
from utils.LoadCalibrationData import pixel2world_pipeline
import cv2
import matplotlib.pyplot as plt
import pickle

def get_lane_data(filepath):

    with open(filepath) as f:
        lines = f.readlines()
    lines = [x.strip() for x in lines] 
    for i in range(len(lines)):
        lines[i] = lines[i].split()
    point_pairs = []
    line_type = []
    curve_type = []
    for i in range(len(lines)):
        line_type.append(lines[i][0])
        curve_type.append(lines[i][1])
        print(lines[i][2:], filepath)
        if not lines[i][2:] == "['[]']":
            points = lines[i][2:]
            #remove brackets and , and ' from array
            # points = [int(x.replace('[','').replace(']','').replace(',','').replace("'",'')) for x in points]
            points = [int(x.replace('[','').replace(']','').replace(',','').replace("'",'')) for x in points if x.replace('[','').replace(']','').replace(',','').replace("'",'') != '']
            points = np.array(points)
            point_pair = []
            for j in range(0,len(points),2):
                point_pair.append((points[j],points[j+1]))
                
            point_pairs.append(point_pair)
    return point_pairs, line_type, curve_type

def pkl2txt(filepath):
    
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    print(data)
    print(filepath[:-5]+'.txt')
    for idx,i in enumerate(data):
        
        with open(filepath[:-5]+'_'+str(idx)+'.txt', 'w') as f:
            for j in i.keys():
                print(j)
                line_type = j
                # line_type = data
                curve_type = "beziers"
                print(i.get(j))
                #check if lane is empty
                if i.get(j):
                    f.write(line_type + ' ' + curve_type + ' ' + str(i.get(j)[0]) + '\n')


def get_world_coords_lane(filepath, depth_path):
    
    point_pairs, line_type, curve_type = get_lane_data(filepath)
    world_points = []
    for i in range(len(point_pairs)):
        world_pair = []
        for x,y in point_pairs[i]:
            Y, X, Z = pixel2world_pipeline(y, x, depth_path, False)
            world_pair.append((Y,Z,0))
        world_points.append(world_pair)
    
    # fig = plt.figure()
    # ax = fig.add_subplot(projection = '3d')
    # for i in range(len(world_points)):
    #     for x,y,z in world_points[i]:
    #         #3d plot
    #         ax.plot(x, y, z, '.-')
            
    # plt.savefig(filepath[:-4]+'_world.png')
            
    with open(filepath[:-4]+'_world.txt', 'w') as f:
        for i in range(len(world_points)):
            if world_points[i]:
                f.write(line_type[i] + ' ' + curve_type[i] + ' ' + str(world_points[i]) + '\n')
    # print(world_points)
def plot_lane_data(image_path, filepath):
    img = cv2.imread(image_path)
    point_pairs, line_type, curve_type = get_lane_data(filepath)
    for i in range(len(point_pairs)):
        for x,y in point_pairs[i]:
            cv2.circle(img, (y,x), 5, (0,0,255), -1)
        # cv2.polylines(img, [np.array(point_pairs[i])], False, (0,255,0), 2)
    cv2.imwrite(filepath[:-4]+'.jpg', img)
    
def main():
    filepath = "Output/1450_lane.txt"
    image_path = "Output/1450.jpg"
    plot_lane_data(image_path, filepath)
    get_world_coords_lane(filepath)
    # get_world_coords_lane(filepath)
    # array = np.load(filepath[:-4]+'.npy')
    # print(array)


if __name__ == "__main__":
    # for i in range(750, 901):
    #     filename = "Output/lane_scene2/" + str(i) + ".jpg.pkl"
    #     pkl2txt(filename)
    # # main()
    # for i in range(750, 901):
    #     filename = "Output/lane_scene2/" + str(i) + "_lane.txt"
    #     get_world_coords_lane(filename)
    pkl2txt("Output/lane_scene10/scene10_lanes.pkl")
    root = "Output/lane_scene10/"
    depth_root = "Output/depth_npy_scene10/"
    # print(root[-7:-1])
    num_files = len([f for f in os.listdir(root) if f.endswith('.txt')])
    for i in range(num_files):
        depth_filename = depth_root + "frame" + str(i+1) + "_pred.npy"
        print(depth_filename)
        filename = root + root[-8:-1] + '_lane_' + str(i) + ".txt"
        print(filename)
        get_world_coords_lane(filename, depth_filename)
        # if not i:
        #     plot_lane_data(root+"0.jpg", filename)

    # filename = "Output/lane_scene5/1755.jpg.pkl"
    # pkl2txt(filename)
    
