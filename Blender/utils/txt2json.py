import json
import pickle
import numpy as np
import os, sys

#add directory to path
sys.path.append("/home/redlightning/Workspace/Git_repos/I2L-MeshNet_RELEASE/demo")
class Object:
    def __init__(self, object_id, x, y, width, height, angle):
        self.object_id = object_id
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.angle = angle
    
    def get_name(self, name_list):
        self.name = name_list[self.object_id]

def YOLO_data_loader(filepath, name_list):
    with open (filepath, 'r') as f:
        data = f.readlines()
        for i in range(len(data)):
            data[i] = data[i].split()

    objects = []
    with open(filepath[:-4] + "_bb.txt", 'w') as f:
        for i in range(len(data)):
            object_id = int(float(data[i][0]))
            x = float(data[i][1])
            y = float(data[i][2])
            width = float(data[i][3])
            height = float(data[i][4])
            angle = float(data[i][5])
            # angle = 0
            #only consider lines that are not zero padded completely
            if x or y or width or height or angle:
                if object_id == 0:
                    
                        xmin = str((x-width/2)*1280 - 5)
                        ymin = str((y-height/2)*960 - 5)
                        w = str(1280*width + 10)
                        h = str(960*height + 10)
                        print(xmin, ymin, w, h)
                        f.write(str((x-width/2)*1280 - 5) + " " + str((y-height/2)*960 - 5) + " " + str(1280*width + 10) +  " " +  str(960*height + 10) + "\n")
                object = Object(object_id, x, y, width, height, angle)
                if object_id in name_list:
                    object.get_name(name_list)
                    objects.append(object)
    #move file to output/bbox/ if file not empty
    if os.path.exists(filepath[:-4] + "_bb.txt") and os.stat(filepath[:-4] + "_bb.txt").st_size > 0:
        os.system("mv " + filepath[:-4] + "_bb.txt Output/bbox/")
    return objects

def YOLO_list_loader(filepath):
    with open (filepath, 'r') as f:
        data = f.readlines()
        for i in range(len(data)):
            data[i] = data[i].split()
            # print(data[i])
        object_list = {}
        for i in range(len(data)):
            object_id = int(data[i][0])
            object_name = data[i][1]

            object_list[object_id] = object_name
        # print(object_list)
        return object_list
    
def convert_to_json(objects, filename):
    data = {}
    data["num_objects"] = len(objects)
    for i, object in enumerate(objects):
        # print(object.name)
        data[i+1] = {
            "name": object.name,
            "x": object.x,
            "y": object.y,
            "width": object.width,
            "height": object.height,   
            "angle": object.angle         
        }
    
    with open(filename[:-4] + ".json", 'w') as f:
        json.dump(data, f, indent=2)

def txt2json(filename="Output/1450.txt"):
    
    filepath = "Output/yolo_list.txt"
    object_names = YOLO_list_loader(filepath)
    detic_object_names = Detic_list_loader()
    objects = YOLO_data_loader(filename, object_names)
    objects.extend(pkl_data_loader_v2(filename[:-4] + ".pkl", detic_object_names))
    convert_to_json(objects, filename)
    return filename[:-4] + ".json"

def pkl_data_loader(filename="Output/1450.pkl"):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    labels = [tensor.cpu().numpy() for tensor in data['labels']]
    labels = labels[0]
    relevant_object_ids = list(data['glossary'].values()) #dictionary of relevant objects in model
    relevant_object_names = list(data['glossary'].keys())
    object_locations = np.array(data['bb_centres'][0])
    # print(object_locations)
    objects = []
    for i in range(len(labels)):
        # print(labels[i])
        #check if labels in glossary
        if labels[i] in relevant_object_ids:
            print(labels[i])
            print(relevant_object_names[relevant_object_ids.index(labels[i])])
            object = Object(labels[i], float(object_locations[i, 1]/1280), float(object_locations[i, 0]/960), 0, 0, 0)
            object.name = relevant_object_names[relevant_object_ids.index(labels[i])]
            objects.append(object)

    return objects

def pkl_data_loader_v2(filename="Output/1450.pkl", glossary={}):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    labels = data['labels']
    object_locations = np.array(data['bb'])
    # print(object_locations)
    relevant_object_ids = list(glossary.keys()) #dictionary of relevant objects in model
    objects = []
    # print(glossary)
   
    for i in range(len(labels)):
        # print(labels[i])
        #check if labels in glossary
        x = (object_locations[i][0] + object_locations[i][2])/2
        y = (object_locations[i][1] + object_locations[i][3])/2
        width = object_locations[i][2] - object_locations[i][0]
        height = object_locations[i][3] - object_locations[i][1]

        if str(labels[i]) in relevant_object_ids:
            object = Object(0, float(x/1280), float(y/960), float(width), float(height), 0)
            object.name = glossary[str(labels[i])]
            print(object.name)
            objects.append(object)

    return objects

def Detic_list_loader(filename="Output/detic_list.txt"):
   with open (filename, 'r') as f:
        data = f.readlines()
        data = [line.split() for line in data]
        object_list = {}
        for i in range(len(data)):
            object_id = data[i][0]
            #join the rest
            object_name = ' '.join(data[i][1:])

            object_list[object_id] = object_name
        return object_list
    
def pkl2txt(filepath):
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    print(data)

    # num_frames = len(data['labels'])
    # print(num_frames)

    # data['labels'][frame_id][num_bb]
    # data['bb'][frame_id][num_bb]

def main():
    # start_id = 0
    # final_id = 429
    # filepath = "Output/frames/labels/" + str(0 + start_id) + ".txt"
    # txt2json(filepath)

    root = "Output/yolo_scene13/"
    num_files = len([f for f in os.listdir(root) if f.endswith('.txt')])
    for i in range(num_files):
        filepath = root + str(i) + ".txt"
        # print(i)
        txt2json(filepath)

def change_filename(number):
    path = "Output/new_yoloo/"
    with open(path+str(number)+'.txt', 'r') as f:
        data = f.readlines()
        #remove line with all zeros
        #split new space
        # data = [line.split() for line in data]
        # #remove zero padding
        # data.pop(i )
        # #write to new file

        with open(path+str(1750 + number)+'.txt', 'w') as f:
            for i in range(len(data)):
                f.write(data[i])

if __name__ == '__main__':
    main()
    # pkl2txt("Output/yolo_scene1/5.pkl")
    # for i in range(151):
    #     change_filename(i)