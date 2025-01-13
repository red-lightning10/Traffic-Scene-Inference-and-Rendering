import bpy
import os
import glob
import sys
import math
import numpy as np
import bmesh
#Add the path to the sys.path list
#sys.path.append("/home/redlightning/Workspace/RBE549/nilampooranan_p3")
#from utils.GetObjectLocation import *

project_path = "/home/redlightning/Workspace/RBE549/nilampooranan_p3"

if 'Cube' in bpy.data.objects:
    bpy.data.objects['Cube'].select_set(True)
    bpy.ops.object.delete()
    
def create_road():
    size = 1000  # Adjust this value to your needs
    location = (0, 0, 0)  # Adjust this value to your needs

    # Create a new plane
    bpy.ops.mesh.primitive_plane_add(size=size, location=location)

    # Get the newly created plane
    plane = bpy.context.active_object

    # Create a new material
    material = bpy.data.materials.new(name="BlackMaterial")

    # Set the material's diffuse color to black
    material.diffuse_color = (0, 0, 0, 1)  # RGBA

    # Assign the material to the plane
    if len(plane.data.materials) > 0:
        # If the plane already has a material, replace it
        plane.data.materials[0] = material
    else:
        # If the plane has no materials, add the new material
        plane.data.materials.append(material)
#    bpy.data.worlds["World"].node_tree.nodes["Sky Texture"].sun_direction = (0.992806, 0.028777, 0.116227)

def setup_env():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()
    for material in bpy.data.materials:
    # Remove material
        bpy.data.materials.remove(material)
    bpy.ops.object.camera_add()
    camera = bpy.data.objects['Camera']
    camera.location = (0,-2,1.2)
    camera.rotation_euler = (math.radians(-90), math.radians(180), math.radians(180)) 
    camera.data.lens = 30
    print(bpy.data.cameras['Camera'].lens)
    


    # Set the new camera as the active camera for the scene
    bpy.context.scene.camera = camera
    bpy.ops.object.light_add(type='POINT', location=(-1, -1, 5))
    bpy.context.object.data.type = 'SUN'
    bpy.context.object.data.energy = 5
    create_road()
    
# cube = bpy.data.objects['Cube']

# # Move the cube to a new location
# cube.location = (0, 0, 0)  # (x, y, z)
#bpy.ops.preferences.addon_enable(module='io_scene_obj')

file_path = "/home/redlightning/Workspace/RBE549/nilampooranan_p3/Data/Assets/Vehicles"
data_path = "/home/redlightning/Workspace/RBE549/nilampooranan_p3/Output/1450_blender.txt"
lane_path = "/home/redlightning/Workspace/RBE549/nilampooranan_p3/Output/1450_lane_world.txt"
car_path = "/home/redlightning/Workspace/RBE549/nilampooranan_p3/Data/Assets/obj"

class Model:
    def __init__(self, name):
        name = name.split(".")
        self.name = name[0]
        if len(name) > 1:
            self.id = "."+name[1]
        else:
            self.id = ""

obj_dir = "/home/redlightning/Workspace/RBE549/nilampooranan_p3/Data/Assets/obj"

def setup_car(name, status, indicator):
    bpy.ops.wm.obj_import(filepath=os.path.join(obj_dir, "Car.obj"), directory=obj_dir, files=[{"name":"Car.obj", "name":"Car.obj"}])
    #rename object
    car = Model(name)
    if not status:
        bpy.data.materials["CarBody"+car.id].node_tree.nodes["Principled BSDF"].inputs[0].default_value = (0.00120701, 0.800529, 0, 1)
    if indicator:
        bpy.data.materials["Aux"+car.id].node_tree.nodes["Principled BSDF"].inputs[0].default_value = (0.801157, 0.000572751, 0, 1)
#    for i in car.materials:
#        bpy.data.materials[i + car.id].diffuse_color = car.materials[i]
        # bpy.context.object.active_material = bpy.data.materials[i + car.id]

def setup_pedestrian(name):

    bpy.ops.wm.obj_import(filepath=os.path.join(obj_dir+"/human_models", name+".obj"), directory=obj_dir+"/human_models", files=[{"name":name+".obj", "name":name+".obj"}])
    bpy.ops.object.origin_set(type='ORIGIN_CENTER_OF_MASS', center='MEDIAN')

def setup_stop_sign(name):
    bpy.ops.wm.obj_import(filepath=os.path.join(obj_dir, "StopSign.obj"), directory=obj_dir, files=[{"name":"StopSign.obj", "name":"StopSign.obj"}])

def setup_traffic_light(name, status):
    bpy.ops.wm.obj_import(filepath=os.path.join(obj_dir, "TrafficSignal.obj"), directory=obj_dir, files=[{"name":"TrafficSignal.obj", "name":"TrafficSignal.obj"}])
    traffic = Model(name)   

    if status == "Go":
        bpy.data.materials["Go"+traffic.id].node_tree.nodes["Principled BSDF"].inputs[0].default_value = (0.0112263, 0.438326, 0, 1)
    elif status == "Wait":
        bpy.data.materials["Wait"+traffic.id].node_tree.nodes["Principled BSDF"].inputs[0].default_value = (0.60575, 0.233333, 0.00631048, 1)
    elif status == "Stop":
        bpy.data.materials["Stop"+traffic.id].node_tree.nodes["Principled BSDF"].inputs[0].default_value = (1, 0.00259199, 0.00553812, 1)

#    for i in traffic.materials:
#        bpy.data.materials[i + traffic.id].diffuse_color = traffic.materials[i]
#    for i in traffic.materials:
#        bpy.context.object.active_material = bpy.data.materials[i + traffic.id]

#    if status == "Go":
#        bpy.data.materials["Go"+traffic.id].diffuse_color = (0.0112263, 0.438326, 0, 1)

#    elif status == "Wait":
#        bpy.data.materials["Wait"+traffic.id].diffuse_color = (0.60575, 0.233333, 0.00631048, 1)

#    elif status == "Stop":
#        bpy.data.materials["Stop"+traffic.id].diffuse_color = (1, 0.00259199, 0.00553812, 1)
#    
def setup_cone(name):
    bpy.ops.wm.obj_import(filepath=os.path.join(obj_dir, "Cone.obj"), directory=obj_dir, files=[{"name":"Cone.obj", "name":"Cone.obj"}])
    cone = Model(name)

def setup_trashcan(name):
    bpy.ops.wm.obj_import(filepath="/home/redlightning/Workspace/RBE549/nilampooranan_p3/Data/Assets/obj/Dustbin.obj", directory="/home/redlightning/Workspace/RBE549/nilampooranan_p3/Data/Assets/obj/", files=[{"name":"Dustbin.obj", "name":"Dustbin.obj"}])
    trashcan = Model(name)

def setup_speedsign(name):
    bpy.ops.wm.obj_import(filepath=os.path.join(obj_dir, "SpeedLimitSign.obj"), directory=obj_dir, files=[{"name":"SpeedLimitSign.obj", "name":"SpeedLimitSign.obj"}])
    speed_sign = Model(name)
    
def setup_truck(name, status, indicator):
    bpy.ops.wm.obj_import(filepath=os.path.join(obj_dir, "Truck.obj"), directory=obj_dir, files=[{"name":"Truck.obj", "name":"Truck.obj"}])
    truck = Model(name)
    if not status:
        bpy.data.materials["TruckBody"+truck.id].node_tree.nodes["Principled BSDF"].inputs[0].default_value = (0.00120701, 0.800529, 0, 1)
    if indicator:
        bpy.data.materials["IndicatorT"+truck.id].node_tree.nodes["Principled BSDF"].inputs[0].default_value = (0.801157, 0.000572751, 0, 1)

def setup_bicycle(name):
    bpy.ops.wm.obj_import(filepath=os.path.join(obj_dir, "Bicycle.obj"), directory=obj_dir, files=[{"name":"Bicycle.obj", "name":"Bicycle.obj"}])
    bicycle = Model(name)
    
def setup_bike(name):
    bpy.ops.wm.obj_import(filepath=os.path.join(obj_dir, "Motorcycle.obj"), directory=obj_dir, files=[{"name":"Motorcycle.obj", "name":"Motorcycle.obj"}])
    bike = Model(name)

def setup_drums(name):
    bpy.ops.wm.obj_import(filepath=os.path.join(obj_dir, "Drum.obj"), directory=obj_dir, files=[{"name":"Drum.obj", "name":"Drum.obj"}])
    drum = Model(name)

def setup_hydrant(name):
    bpy.ops.wm.obj_import(filepath=os.path.join(obj_dir, "FireHydrant.obj"), directory=obj_dir, files=[{"name":"FireHydrant.obj", "name":"FireHydrant.obj"}])
    hydrant = Model(name)

def get_object_id(num):
    if num < 10:
        return '00'+str(num)
    else:
        return '0'+str(num)

def read_lane_points(filepath):
    with open(filepath, 'r') as f:
        data = f.readlines()
        for i in range(len(data)):
            data[i] = data[i].split()
        lane_points = []
        lane_types = []
        for i in range(len(data)):
            line_type = data[i][0]
            curve_type = data[i][1]
            points = data[i][2:]
            points = [float(float(x.replace('(','').replace(')','').replace(',','').replace('[','').replace(']',''))) for x in points]
            points = np.array(points)
            point_pair = []
            for j in range(0,len(points),3):
                point_pair.append((points[j],points[j+1], points[j+2]))
            lane_points.append(point_pair)
            lane_types.append(line_type)

        return lane_points, lane_types

#def plot_lane_points_blender(lane_points):
#    # Create a new curve data object
#    curve_data = bpy.data.curves.new('my_curve', type='CURVE')

#    # Create a new polyline in the curve
#    polyline = curve_data.splines.new('POLY')

#    # Set the number of points in the polyline
#    polyline.points.add(len(lane_points) - 1)

#    # Set the coordinates of the points
#    for i, xyz in enumerate(lane_points):
#        x, y, z = xyz
#        polyline.points[i].co = (x, y, z, 1)  # The fourth element is the weight of the point

#    # Create a new object with the curve
#    curve_object = bpy.data.objects.new('my_curve', curve_data)

#    # Link the curve object to the current collection
#    bpy.context.collection.objects.link(curve_object)

#    # Make the curve object the active object
#    bpy.context.view_layer.objects.active = curve_object
#    curve_object.select_set(True)

#    # Increase the thickness of the curve
#    curve_data.bevel_depth = 0.02  # Adjust this value to your needs

#def plot_lane_points_blender(lane_points):
#    # Create a new curve data object
#    curve_data = bpy.data.curves.new('my_curve', type='CURVE')

#    # Create a new NURBS curve in the curve
#    nurbs_curve = curve_data.splines.new('NURBS')

#    # Set the number of points in the curve
#    nurbs_curve.points.add(len(lane_points) - 1)

#    # Set the coordinates of the points
#    for i, xyz in enumerate(lane_points):
#        x, y, z = xyz
#        nurbs_curve.points[i].co = (x, y, z, 1)  # The fourth element is the weight of the point

#    # Set the order of the NURBS curve
#    nurbs_curve.order_u = len(nurbs_curve.points)

#    # Make the NURBS curve cyclic
#    nurbs_curve.use_cyclic_u = False

#    # Create a new object with the curve
#    curve_object = bpy.data.objects.new('my_curve', curve_data)

#    # Link the curve object to the current collection
#    bpy.context.collection.objects.link(curve_object)

#    # Make the curve object the active object
#    bpy.context.view_layer.objects.active = curve_object
#    curve_object.select_set(True)

#    # Increase the thickness of the curve
#    curve_data.bevel_depth = 0.02  # Adjust this value to your needs
#    

def create_bezier_curve(control_points):

    curve_data = bpy.data.curves.new('my_curve', type='CURVE')
    bezier_curve = curve_data.splines.new('BEZIER')

    bezier_curve.bezier_points.add(len(control_points) - 1)
    control_points = control_points[::-1]

    for i, xyz in enumerate(control_points):
        bezier_curve.bezier_points[i].co = xyz

    for point in bezier_curve.bezier_points:
        point.handle_left_type = 'AUTO'
        point.handle_right_type = 'AUTO'

    curve_object = bpy.data.objects.new('my_curve', curve_data)

    bpy.context.collection.objects.link(curve_object)
    bpy.context.view_layer.objects.active = curve_object
    curve_object.select_set(True)

    return curve_object

def create_plane_and_distort_along_curve(curve_object, lane_type, control_points):

    bpy.ops.mesh.primitive_plane_add(size=1, enter_editmode=False, align='WORLD', location=(0, 0, 0))
    plane = bpy.context.object
    plane.scale = (1, 0.1, 0.1)
    if lane_type == 'solid-line':

        array_modifier = plane.modifiers.new(name="Array", type='ARRAY')
        array_modifier.fit_type = 'FIT_CURVE'
        array_modifier.curve = curve_object
        curve_modifier = plane.modifiers.new(name="Curve", type='CURVE')
        curve_modifier.object = curve_object

    elif lane_type == 'dotted-line':
        print('dotted')
        
        array_modifier = plane.modifiers.new(name="Array", type='ARRAY')
        array_modifier.fit_type = 'FIXED_COUNT'
        array_modifier.count = len(curve_object.data.splines[0].bezier_points) + 2
        bpy.context.object.modifiers["Array"].use_constant_offset = True
        

        array_modifier.constant_offset_displace[0] = 1
        curve_modifier = plane.modifiers.new(name="Curve", type='CURVE')
        curve_modifier.object = curve_object
    
    elif lane_type == "arrow":
        vertices = control_points
        mesh_data = bpy.data.meshes.new("arrow_mesh")

        # Create a new object with the mesh data
        mesh_object = bpy.data.objects.new("Mark", mesh_data)

        # Link the mesh object to the current collection
        bpy.context.collection.objects.link(mesh_object)

        # Set the mesh object as the active object
        bpy.context.view_layer.objects.active = mesh_object
        mesh_object.select_set(True)

        # Create a bmesh object
        bm = bmesh.new()

        # Add the vertices to the bmesh object
        for v in vertices:
            bm.verts.new(v)

        # Update the bmesh to the mesh data
        bm.to_mesh(mesh_data)
        bm.free()

                # Switch back to object mode
        bpy.ops.object.mode_set(mode='OBJECT')

def create_arrow(yaw, length):
    # Define the vertices of the arrow pointing along the X-axis
    vertices = [(0, 0, 0), (1, 0, 0), (1, 0.2, 0), (1.5, 0, 0), (1, -0.2, 0), (1, 0, 0), (0, 0, 0)]

    # Define the faces of the arrow
    faces = [(0, 1, 2, 3, 4, 5, 6)]

    # Create a new mesh data object
    mesh_data = bpy.data.meshes.new("arrow_mesh")

    # Create a new object with the mesh data
    mesh_object = bpy.data.objects.new("Arrow", mesh_data)

    # Link the mesh object to the current collection
    bpy.context.collection.objects.link(mesh_object)

    # Set the mesh object as the active object
    bpy.context.view_layer.objects.active = mesh_object
    mesh_object.select_set(True)

    # Fill the mesh data with the vertices and faces
    mesh_data.from_pydata(vertices, [], faces)
    mesh_object.scale = (length, length, length)

start_id = 310
final_id = 1900
scene = str(9)
frame_count = 0
hold_frames = 1
plotLanes = 1
blender_root = "/home/redlightning/Workspace/RBE549/nilampooranan_p3/Output/blender_scene"+scene+"/"
num_files = len(os.listdir(blender_root))


for j in range(start_id, start_id + 500):
    data_path = blender_root+str(j)+"_blender.txt"
    optical_flow_path = "/home/redlightning/Workspace/RBE549/nilampooranan_p3/Output/Optical_flow/"+scene+"/"+str(j)+".txt"
    status = []
    if os.path.exists(optical_flow_path):
        with open(optical_flow_path, 'r') as f:
            data = f.readlines()
            if not data:
                status = None
            else:
                for i in range(len(data)):
                    status.append(int(float(data[i])))
    else:
        status = None
    print(status)
    indicator = 0
    print(data_path)
    print(str(j))
#    lane_path = "/home/redlightning/Workspace/RBE549/nilampooranan_p3/Output/"+str(start_id+i)+"_lane_world.txt"
    print(data_path)
#    print(lane_path)
    
#    bpy.context.scene.render.filepath = "/home/redlightning/Workspace/RBE549/nilampooranan_p3/Output/animation_output.mp4"
#    bpy.context.scene.render.image_settings.file_format = 'FFMPEG'
#    bpy.ops.render.render(animation=True)
    
    setup_env()

    cars = 0
    humans = 0
    signs = 0
    signals = 0
    cones = 0
    trashcans = 0
    speedsigns = 0
    trucks = 0
    hydrants = 0
    bikes = 0
    cycles = 0
    drums = 0
    with open(data_path, 'r') as f:
        data = f.readlines()
        if status == None:
            status = [1]*len(data)
        for i in range(len(data)):
            data[i] = data[i].split()
        s = 0
        for i in range(len(data)):
            object_name = data[i][0]
            x = float(data[i][1])
            y = float(data[i][2])
            z = float(data[i][3])
            yaw = float(data[i][4])
            status.append(1)
            print(object_name)
            if object_name == "Car":
                cars += 1
                if cars == 1:
                    setup_car("Car", status[s], indicator)
                    car = bpy.data.objects['Car']
                else:
                    setup_car("Car."+get_object_id(cars-1), status[s], indicator)
                    car = bpy.data.objects['Car.'+get_object_id(cars-1)]
                
                car.location = (x,y,z)
                if yaw == 0:
                    yaw = math.pi/2
                yaw = -yaw
                car.rotation_euler = (math.pi/2, 0, yaw )
                create_arrow(yaw, 2)
                arrow = bpy.context.active_object
                arrow.location = (x, y-1, car.dimensions.z + 0.2)
                arrow.rotation_euler = (0, 0, yaw)
                s += 1
                
            elif object_name == "Pedestrian":
                humans += 1
#                if i > 1:
#                    setup_pedestrian("output_mesh_lixel."+get_object_id(humans-1))
#                    human = bpy.data.objects['output_mesh_lixel.'+get_object_id(humans-1)]
#                setup_pedestrian(str(j)+"_"+str(humans))
#                human = bpy.data.objects[str(j)+"_"+str(humans)]
#                human.dimensions.y = 1.8
#                human.scale = (human.scale[1], human.scale[1], human.scale[1])
#                human.location = (x, y, 0.9792)
#                human.rotation_euler = (-math.pi/2,0,0)
                
            elif object_name == "RoadSign":
                signs += 1
                setup_stop_sign("StopSign_Geo."+get_object_id(signs-1))
                sign = bpy.data.objects['StopSign_Geo.'+get_object_id(signs)]
                sign.location = (x,y,z)
            
            elif object_name == "TrafficLight":
                signals += 1
                if signals == 1:
        
                    setup_traffic_light("Traffic_signal1", "Go")
                    signal = bpy.data.objects['Traffic_signal1']
                else:
                    setup_traffic_light("Traffic_signal1."+get_object_id(signals-1), "Go")
                    signal = bpy.data.objects['Traffic_signal1.'+get_object_id(signals-1)]
                signal.location = (x,y,z+10)

            elif object_name == "Cone":
                setup_cone("Cone")
                cones += 1
                if cones == 1:
                    cone = bpy.data.objects['absperrhut']
                else:
                    cone = bpy.data.objects['absperrhut.'+get_object_id(cones-1)]
                cone.location = (x, y, z)
            
            elif object_name == "TrashCan":
                print("Trashcan")
                trashcans += 1
                setup_trashcan("Dustbin")
                if trashcans == 1:
                    trashcan = bpy.data.objects['Dustbin']
                else:
                    trashcan = bpy.data.objects['Dustbin.'+get_object_id(trashcans-1)]
                trashcan.location = (x, y, z)
            
            elif object_name == "StreetSign":
                speedsigns += 1
                setup_speedsign("SpeedLimitSign")
                if speedsigns == 1:
                    speed_sign = bpy.data.objects['SpeedLimitSign']
                else:
                    speed_sign = bpy.data.objects['SpeedLimitSign.'+get_object_id(speedsigns-1)]
                speed_sign.location = (x, y, z)
            
            elif object_name == "Truck":
                trucks += 1
                if trucks == 1:
                    setup_truck("Truck", status[s], indicator)
                    truck = bpy.data.objects['Truck']
                else:
                    setup_truck("Truck."+get_object_id(trucks-1), status[s], indicator)
                    truck = bpy.data.objects['Truck.'+get_object_id(trucks-1)]
                truck.location = (x, y, z)
                if yaw == 0:
                    yaw = 0
                truck.rotation_euler = (math.pi/2, 0, yaw )
                create_arrow(yaw, 2)
                arrow = bpy.context.active_object
                arrow.location = (x, y, truck.dimensions.y + 0.2)
                arrow.rotation_euler = (0, 0, yaw + math.pi/2)
                s += 1
            
            elif object_name == "FireHydrant":
                hydrants += 1
                if hydrants == 1:
                    setup_hydrant("FireHydrant")
                    hydrant = bpy.data.objects['FireHydrant']
                else:
                    setup_hydrant("FireHydrant."+get_object_id(hydrants-1))
                    hydrant = bpy.data.objects['FireHydrant.'+get_object_id(hydrants-1)]
                hydrant.location = (x, y, z)
            elif object_name == "Bicycle":
                cycles += 1
                if cycles == 1:
                    setup_bicycle("Bicycle")
                    cycle = bpy.data.objects['Bicycle']
                else:
                    setup_bicycle("Bicycle."+get_object_id(cycles-1))
                    cycle = bpy.data.objects['Bicycle.'+get_object_id(cycles-1)]
                cycle.location = (x, y, z)
            elif object_name == "Motorcycle":
                bikes += 1
                if bikes == 1:
                    setup_bike("Motorcycle")
                    bike = bpy.data.objects['Motorcycle']
                else:
                    setup_bike("Motorcycle."+get_object_id(bikes-1))
                    bike = bpy.data.objects['Motorcycle.'+get_object_id(bikes-1)]
                bike.location = (x, y, z)
            elif object_name == "Drum":
                drums += 1
                if drums == 1:
                    setup_trashcan("Drum")
                    drum = bpy.data.objects['Drum']
                else:
                    setup_trashcan("Drum."+get_object_id(drums-1))
                    drum = bpy.data.objects['Drum.'+get_object_id(drums-1)]
                drum.location = (x, y, z)


    if plotLanes: 
        lane_path = "/home/redlightning/Workspace/RBE549/nilampooranan_p3/Output/lane_scene"+scene+"/scene"+scene+"_lane_"+str(j)+"_world.txt"       
        lane_points, lane_types = read_lane_points(lane_path)
    #    print(lane_points[0])
    #    
        for i in range(len(lane_points)):
    #        plot_lane_points_blender(lane_points[i])
            curve = create_bezier_curve(lane_points[i])
            print(lane_types[i])
            create_plane_and_distort_along_curve(curve, lane_types[i], lane_points[i])
    bpy.context.scene.render.filepath = "/home/redlightning/Workspace/RBE549/nilampooranan_p3/Output/render"+scene+"/"+str(j)+".png"
    bpy.ops.render.render(write_still=True)
    frame_count += hold_frames
    
    #bpy.context.scene.render.image_settings.file_format = 'FFMPEG'

    # Save animation
    
#lane_points, lane_types = read_lane_points(lane_path)
#print(lane_points[0])
#plot_lane_points_blender(lane_points, lane_types)



#        car_path = os.path.join(file_path, "SedanAndHatchback.blend")
        # Specify the object name in the .blend file
#        object_name = "Car"
#        # Append the object from the .blend file
#        bpy.ops.wm.append(directory=car_path + "/Object/", filename=str(i))
#        obj = bpy.data.objects[str(i)]
#        obj.location = (x, y, z)
        

        

# car_path = os.path.join(file_path, "SedanAndHatchback.blend")
# # Specify the object name in the .blend file
# object_name = "Car"
# # Append the object from the .blend file
# bpy.ops.wm.append(directory=car_path + "/Object/", filename=object_name)

    # print(i)
    # cube = cube = bpy.data.objects['Car']
    # cube.location = (0, 0, 5) 
    # camera.rotation_euler = (math.radians(90), math.radians(0), math.radians(0)) 
    # camera.location = (0, 1.25, 1.25)
#bpy.ops.import_scene.obj(filepath=file_path + "/" + "SUV"

#bpy.context.scene.frame_end = frame_count

## Set output path and file format
#bpy.context.scene.render.filepath = r"/home/redlightning/Workspace/RBE549/nilampooranan_p3/Output/animation_output.jpg"
##bpy.context.scene.render.image_settings.file_format = 'FFMPEG'

## Save animation
#bpy.ops.render.render(write_still=True)
#bpy.ops.render.render(animation=True)