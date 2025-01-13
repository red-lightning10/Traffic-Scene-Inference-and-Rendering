import os
import json
from utils.LoadCalibrationData import pixel2world_pipeline
from utils.txt2json import txt2json

class Model:
    def __init__(self, name=None, x=None, y=None, z=None, angle=0):
        self.name = name
        self.x = x
        self.y = y
        self.z = z
        self.yaw = angle

def get_location_from_json(filepath, depth_path):
    objects = []
    with open(filepath, 'r') as f:
        data = json.load(f)
    for i in range(data['num_objects']):
        # print(data[str(i+1)]['name'])
        x = data[str(i+1)]['x']
        y = data[str(i+1)]['y']
        yaw = data[str(i+1)]['angle']
        print('s', yaw)
        # print('d', x,y)
        X, Y, Z = pixel2world_pipeline(x, y, depth_path)
        # print(X,Y,Z)
        if data[str(i+1)]['name'] == 'car':
            car = Model("Car", X, Z, 0, yaw)
            print(car.x, car.y, car.z)
            objects.append(car)
        elif data[str(i+1)]['name'] == 'truck':
            truck = Model("Truck", X, Z, 0, yaw)
            objects.append(truck)
        elif data[str(i+1)]['name'] == 'bicycle':
            bicycle = Model("Bicycle", X, Z, 0, yaw)
            objects.append(bicycle)
        elif data[str(i+1)]['name'] == 'motorcycle':
            motorcycle = Model("Motorcycle", X, Z, 0, yaw)
            objects.append(motorcycle)
        elif data[str(i+1)]['name'] == 'person':
            pedestrian = Model("Pedestrian", X, Z, 0, yaw)
            objects.append(pedestrian)
        elif data[str(i+1)]['name'] == 'Traffic Light':
            traffic_light = Model("TrafficLight", X, Z, 0, yaw)
            objects.append(traffic_light)
        elif data[str(i+1)]['name'] == 'stop sign':
            stop_sign = Model("RoadSign", X, Z, 0, yaw)
            objects.append(stop_sign)
        elif data[str(i+1)]['name'] == 'Trash Bin':
            trashcan = Model("TrashCan", X, Z, 0, yaw)
            objects.append(trashcan)
        elif data[str(i+1)]['name'] == 'Drum':
            drum = Model("Drum", X, Z, 0, yaw)
            objects.append(drum)
        elif data[str(i+1)]['name'] == 'Traffic Cone':
            cone = Model("Cone", X, Z, 0, yaw)
            objects.append(cone)
        elif data[str(i+1)]['name'] == 'Speed Limit Sign' or data[str(i+1)]['name'] == 'Crosswalk Sign':
            street_sign = Model("StreetSign", X, Z, 0, yaw)
            objects.append(street_sign)
        elif data[str(i+1)]['name'] == 'Fire Hydrant':
            fire_hydrant = Model("FireHydrant", X, Z, 0, yaw)
            objects.append(fire_hydrant)

    return objects

def get_depth_from_txt(filepath):
    with open(filepath, 'r') as f:
        data = f.readlines()
    for i in range(len(data)):
        data[i] = data[i].split(',')

    
def export_data_to_blender(objects, blender_filename):
    with open(blender_filename, 'w') as f:
        for i in objects:
            if i.name == 'Car':
                f.write('Car ' + str(i.x) + ' ' + str(i.y) + ' ' + str(i.z) + ' ' + str(i.yaw) + '\n')
            elif i.name == 'Truck':
                f.write('Truck ' + str(i.x) + ' ' + str(i.y) + ' ' + str(i.z) + ' ' + str(i.yaw) + '\n')
            elif i.name == 'Bicycle':
                f.write('Bicycle ' + str(i.x) + ' ' + str(i.y) + ' ' + str(i.z) + ' ' + str(i.yaw) + '\n')
            elif i.name == 'Motorcycle':
                f.write('Motorcycle ' + str(i.x) + ' ' + str(i.y) + ' ' + str(i.z) + ' ' + str(i.yaw) + '\n')
            elif i.name == 'Pedestrian':
                f.write('Pedestrian ' + str(i.x) + ' ' + str(i.y) + ' ' + str(i.z) + ' ' + str(i.yaw) + '\n')
            elif i.name == 'RoadSign':
                f.write('RoadSign ' + str(i.x) + ' ' + str(i.y) + ' ' + str(i.z) + ' ' + str(i.yaw) + '\n')
            elif i.name == 'TrafficLight':
                f.write('TrafficLight ' + str(i.x) + ' ' + str(i.y) + ' ' + str(i.z) + ' ' + str(i.yaw) + '\n')
            elif i.name == 'TrashCan':
                f.write('TrashCan ' + str(i.x) + ' ' + str(i.y) + ' ' + str(i.z) + ' ' + str(i.yaw) + '\n')
            elif i.name == 'Cone':
                f.write('Cone ' + str(i.x) + ' ' + str(i.y) + ' ' + str(i.z) + ' ' + str(i.yaw) + '\n')
            elif i.name == 'StreetSign':
                f.write('StreetSign ' + str(i.x) + ' ' + str(i.y) + ' ' + str(i.z) + ' ' + str(i.yaw) + '\n')
            elif i.name == 'FireHydrant':
                f.write('FireHydrant ' + str(i.x) + ' ' + str(i.y) + ' ' + str(i.z) + ' ' + str(i.yaw) + '\n')
            elif i.name == 'Drum':
                f.write('Drum ' + str(i.x) + ' ' + str(i.y) + ' ' + str(i.z) + ' ' + str(i.yaw) + '\n')

    

def main():
    # start_id = 750
    # final_id = 900
    # for i in range(final_id - start_id + 1):
    #     json_path = "Output/frames/labels/" + str(i + start_id) + ".json"
    #     depth_path = "Output/frames/" + str(i + start_id) + "_pred.npy"
    #     blender_filename = "Output/frames/" + str(i + start_id) + "_blender.txt"
    #     objects = get_location_from_json(json_path, depth_path)
    #     export_data_to_blender(objects, blender_filename)

    root = "Output/yolo_scene13/"
    depth_root = "Output/depth_npy_scene13/"
    blender_root = "Output/blender_scene13/"
    num_files = len([f for f in os.listdir(root) if f.endswith('.json')])
    for i in range(num_files):
        json_path = root + str(i) + ".json"
        depth_path = depth_root + "frame" + str(i+1) + "_pred.npy"
        blender_filename = blender_root + str(i) + "_blender.txt"
        objects = get_location_from_json(json_path, depth_path)
        export_data_to_blender(objects, blender_filename)
    # json_path = txt2json("Output/1450.txt")
    # # depth_path = "Output/947.csv"
    # depth_path = "Output/1450_pred.npy"
    # blender_filename = "Output/1450_blender.txt"
    # objects = get_location_from_json(json_path, depth_path)
    # export_data_to_blender(objects, blender_filename)

if __name__ == '__main__':
    main()