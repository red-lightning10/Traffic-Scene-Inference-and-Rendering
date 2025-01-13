import csv
import numpy as np
import os
import glob

def load_cam_calibration_data(filepath="Data/Calib/K_front.csv"):
    
    with open(filepath, 'r') as f:
        data = csv.reader(f)
        K = np.zeros((3,3))
        for i, row in enumerate(data):
            K[i, 0], K[i, 1], K[i, 2] = float(row[0]), float(row[1]), float(row[2])
        return K

def get_depth_pixel(filepath="Output/1000.csv"):
    
    with open(filepath, 'r') as f:
        data = csv.reader(f)
        #remove first row
        data = list(data)[1:]
        #convert to float
        data = [[float(j) for j in i] for i in data]
        for i, row in enumerate(data):
            if not i:
                depth = np.array(row)
            else:
                depth = np.vstack((depth, row))
    return depth

def pixel2world(u, v, K, depth):
    # K is the camera calibration matrix
    # depth is the depth value at pixel (u, v)
    # returns the 3D coordinates of the pixel (u, v)
    # print(K)

    # print(type(depth))
    # print(u)
    # print(v)
    # print(K[0, 0])
    # print(K[1, 1])
    # print(K[0, 2])
    # print(K[1, 2])
    # print(depth)
    # scale = 1.5
    #for 1000 image
    # depth = depth * 5
    # x = scale * (u - K[0, 2]) * depth / K[0, 0]
    # y = scale * (v - K[1, 2]) * depth / K[1, 1]
    # z = depth * 1.2 - 25
    # scale = 2.5
    # depth = depth*1.5
    # x = scale * (u - K[0, 2]) * depth / K[0, 0]
    # y = scale * (v - K[1, 2]) * depth / K[1, 1]
    # z = depth - 15
    # scale = 3.5
    # depth = depth * 7.5
    # x = scale * (u - K[0, 2]) * depth / K[0, 0]
    # y = scale * (v - K[1, 2]) * depth / K[1, 1]
    # z = depth*2 - 20
    # scale = 2.5
    # depth = depth * 1.5
    # x = scale * (u - K[0, 2]) * depth / K[0, 0]
    # y = scale * (v - K[1, 2]) * depth / K[1, 1]
    # z = depth*1.5 - 20
    # scale = 3
    # depth = depth * 2.5
    # x = scale * (u - K[0, 2]) * depth / K[0, 0]
    # y = scale * (v - K[1, 2]) * depth / K[1, 1]
    # z = depth*2.2
    # scale = 3
    # depth = depth
    # x = scale * (u - K[0, 2]) * depth / K[0, 0]
    # y = scale * (v - K[1, 2]) * depth / K[1, 1]
    # z = depth
    #rotation matrix guess
    theta = np.deg2rad(10)
    scale = 66
    R = np.array([[1, 0, 0], [0, np.cos(theta), -np.sin(theta)], [0, np.sin(theta), np.cos(theta)]])
    world_coords = R.T @ np.linalg.inv(K) @ np.array([u, v, 1]) * depth * scale
    x = world_coords[0]
    y = world_coords[1] 
    z = world_coords[2]
    print(world_coords)
    #66 scene1
    #66 scene 5
    #70 scene 2
    #75 scene 3
    

    return x, y, z

def pixel2world_pipeline(u, v, depth_filepath, normalized=True):
    calib_path = "Data/Calib"
    #get file names in directory
    K = []
    # for i in glob.glob(calib_path + "/*.csv"):
    #     print(i)
    #     K.append(load_cam_calibration_data(i))
    K = load_cam_calibration_data(os.path.join(calib_path, "K_front.csv"))
    # print((K[2]))
    # depth_data = get_depth_pixel(depth_filepath)
    depth_data = np.load(depth_filepath)
    print(depth_data)
    
    if normalized:
        frame_width = depth_data.shape[1]
        frame_height = depth_data.shape[0]
        print(frame_width, frame_height)
        u = int(u * frame_width)
        v = int(v * frame_height)
    print(u, v)
    # print(depth_data[v, u])
    x, y, z = pixel2world(u, v, K, depth_data[int(v), int(u)])

    return x, y, z
    # theta = np.deg2rad(10)
    # R = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
    # K[1] = K[1] @ R
    
    # print(pixel2world(u, v, K[1], depth_data[u, v]))
    # cameras = os.path


    # load_car_calibration_data(os.path.joincalib_path)
# if __name__ == '__main__':
#     pixel2world_pipeline(0, 0)