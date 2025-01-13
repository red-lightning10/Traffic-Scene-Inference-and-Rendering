import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
import json
from utils.LoadCalibrationData import pixel2world_pipeline

def get_pose_data(pose_file):
    with open(pose_file, 'r') as f:
        lines = f.readlines()

    pose_datas = {}
    pose_data = {}
    keypoints = []
    num_humans = len(lines) // 18
    print(num_humans)
    keypoints = np.zeros((num_humans, 18, 2))
    kp = []
    temp = np.zeros((num_humans, 2))
    for i in range(len(lines)):
        lines[i] = lines[i].split()
        kp.append((float(lines[i][0]), float(lines[i][1])) )
    com = np.zeros((num_humans, 2))
    for i in range(num_humans):
        com[i] = kp[i][0], kp[i][1]
    print(com)
    for i in range(len(lines)//num_humans):
        for j in range(num_humans):
            temp[j, :] = float(lines[i*num_humans + j][0]), float(lines[i*num_humans + j][1])
            #add temp to whichever com mean it is closest to 
            # print(temp)
            # print(com)
        # for j in range(num_humans):
        #     dist_array = np.linalg.norm(temp - com[j], axis=1)
        #     sorted_indices = np.argsort(dist_array)
        #     print(dist_array)
        #     keypoints[j, i, :] = temp[sorted_indices[0]]
        #     com[j] = np.mean(keypoints[j], axis=0)
        # dist_array = np.linalg.norm(temp - com, axis=1)
        # sorted_indices = np.argsort(dist_array)
        # print(sorted_indices)
        # if np.linalg.norm(temp[0] - com[0]) < np.linalg.norm(temp[1] - com[0]):
        #     keypoints[0, i, :] = temp[0]
        #     keypoints[1, i, :] = temp[1]
        # else:
        #     keypoints[0, i, :] = temp[1]
        #     keypoints[1, i, :] = temp[0]
        
        # com[0] = np.mean(keypoints[0], axis=0)
        # com[1] = np.mean(keypoints[1], axis=0)
        dist_array = np.linalg.norm(temp - com, axis=1)
        sorted_indices = np.argsort(dist_array)
        print(sorted_indices)

        for j in range(num_humans):
            keypoints[j, i, :] = temp[sorted_indices[j]]
            com[j] = np.mean(keypoints[j], axis=0)
           
        # slope = keypoints[0, i, :] - keypoints[1, i, :]
        # print(slope)
            
    
    # for i in range(num_humans):
    #     for j in range(18):
    #         keypoints[i, j, :] = float(lines[
    with open('Output/pose_blender.txt', 'w') as f:
        for i in range(num_humans):
            # keypoints[j] = keypoints[j] - np.mean(keypoints[j], axis=0)
            # keypoints = keypoints[j] / np.std(keypoints[j], axis=0)
            f.write('Nose ' + str(keypoints[i, 0, :]) + '\n')
            f.write('Chest ' + str(keypoints[i, 1, :]) + '\n')
            f.write('Right Shoulder ' + str(keypoints[i, 2, :]) + '\n')
            f.write('Right Elbow ' + str(keypoints[i, 3, :]) + '\n')
            f.write('Right Wrist ' + str(keypoints[i, 4, :]) + '\n')
            f.write('Left Shoulder ' + str(keypoints[i, 5, :]) + '\n')
            f.write('Left Elbow ' + str(keypoints[i, 6, :]) + '\n')
            f.write('Left Wrist ' + str(keypoints[i, 7, :]) + '\n')
            f.write('Right Hip ' + str(keypoints[i, 8, :]) + '\n')
            f.write('Right Knee ' + str(keypoints[i, 9, :]) + '\n')
            f.write('Right Ankle ' + str(keypoints[i, 10, :]) + '\n')
            f.write('Left Hip ' + str(keypoints[i, 11, :]) + '\n')
            f.write('Left Knee ' + str(keypoints[i, 12, :]) + '\n')
            f.write('Left Ankle ' + str(keypoints[i, 13, :]) + '\n')
            # pose_data['nose'] = keypoints[i, 0, :]
            # pose_data['chest'] = keypoints[i, 1, :]
            # pose_data['right_shoulder'] = keypoints[i, 2, :]
            # pose_data['right_elbow'] = keypoints[i, 3, :]
            # pose_data['right_wrist'] = keypoints[i, 4, :]
            # pose_data['left_shoulder'] = keypoints[i, 5, :]
            # pose_data['left_elbow'] = keypoints[i, 6, :]
            # pose_data['left_wrist'] = keypoints[i, 7, :]
            # pose_data['right_hip'] = keypoints[i, 8, :]
            # pose_data['right_knee'] = keypoints[i, 9, :]
            # pose_data['right_ankle'] = keypoints[i, 10, :]
            # pose_data['left_hip'] = keypoints[i, 11, :]
            # pose_data['left_knee'] = keypoints[i, 12, :]
            # pose_data['left_ankle'] = keypoints[i, 13, :]
            # pose_datas[str(i+1)] = pose_data
    # print(pose_datas)

    #export as json
    # print(keypoints.shape)
    
    #get depth
    kp_3d = []
    for kp in keypoints[0]:
        X,Y,Z = pixel2world_pipeline(int(kp[0]), int(kp[1]), 'Output/1450_pred.npy', False)
        #3d plot
        
        kp_3d.append((X,Y,Z))
    print(kp_3d)
    kp_3d = np.array(kp_3d)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(kp_3d[:, 0], kp_3d[:, 1], kp_3d[:, 2], c='r', marker='o')
    #add text
    for i in range(len(kp_3d)):
        ax.text(kp_3d[i, 0], kp_3d[i, 1], kp_3d[i, 2], str(i))
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.show()

    #create bone structure plot
    # keypoints = keypoints[0]
    # keypoints = np.array(keypoints)
    # print(keypoints)
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # for i in range(len(keypoints)):
    #     x, y, z = keypoints[i]
    #     ax.scatter(x, y, z, c='r', marker='o')
    #     ax.text(x, y, z, str(i))
    

    # plt.show()
    #normalize
    # for i in range(1):
    #     # keypoints[i] = keypoints[i] - np.mean(keypoints[i], axis=0)
    #     # keypoints[i] = keypoints[i] / np.std(keypoints[i], axis=0)
    #     print(keypoints[i])
    #     #plot keypoints
    #     plt.plot(keypoints[i, :14, 0], -keypoints[i, :14, 1], 'bo')
    #     plt.show()
    # keypoints = np.array(keypoints)
    # keypoints = keypoints - np.mean(keypoints, axis=0)
    # keypoints = keypoints / np.std(keypoints, axis=0)
    # pose_data['keypoints'] = keypoints.tolist()
    #set keypoint 0 as origin
    # keypoints = [(x - keypoints[1][0], y - keypoints[1][1]) for x,y in keypoints]
    # print(keypoints)
    #plot
    # img = plt.imread("Output/1450.jpg")

    # for i in range(len(keypoints[0])):
    #     #plot circles on image using plt
    #     x, y = keypoints[0, i]
    #     plt.plot(x, y, 'ro')
    #     #attach number
    #     plt.text(x, y, str(i))
    #     plt.imshow(img)
        
    # plt.show()
#     armature = [[ 0.,         -1.07800359 , 1.10977237],
#  [ 0.        ,  1.11310824,  0.82015789],
#  [ 0.59210005 , 1.11310824,  0.75351553],
#  [ 1.33885298 , 0.88627848,  0.74117443],
#  [ 2.11653667 , 0.14908186,  0.70661905],
#  [ 0.31912514 ,-1.61042328, -0.35225271],
#  [ 0.3620843  ,-0.98664179, -0.98987934],
#  [ 0.37975893 , 0.43104417, -1.8241421 ],
#  [-0.59210005 , 1.11310824 , 0.75351553],
#  [-1.33885298 , 0.88627848,  0.74117443],
#  [-2.11653667 , 0.14908186,  0.70661905],
#  [-0.31912514, -1.61042328, -0.35225271],
#  [-0.3620843 , -0.98664179, -0.98987934],
#  [-0.37975893 , 0.43104417, -1.8241421 ]]

#     print(armature)

#     #plot
#     for i in range(len(armature)):
#         x, y = armature[i][0], armature[i][2]
#         plt.plot(x, y, 'ro')
#     plt.show()


def main():
    pose_file = "Output/pose.txt"
    get_pose_data(pose_file)

if __name__ == "__main__":
    main()