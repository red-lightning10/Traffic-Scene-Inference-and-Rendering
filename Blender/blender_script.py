import bpy
import os
import glob
import sys
import math
import numpy as np
import bmesh
import argparse
from pathlib import Path


class BlenderSceneRenderer:
    """
    Main class for rendering traffic scenes in Blender.
    """
    
    def __init__(self, project_path=None, output_dir=None):
        """
        Initialize the Blender scene renderer.
        
        Args:
            project_path: Path to project root directory
            output_dir: Directory for output files
        """
        self.project_path = project_path or os.getcwd()
        self.output_dir = output_dir or os.path.join(self.project_path, 'outputs', 'blender')
        
        # Default asset paths
        self.vehicle_assets_path = os.path.join(self.project_path, 'Data', 'Assets', 'Vehicles')
        self.obj_assets_path = os.path.join(self.project_path, 'Data', 'Assets', 'obj')
        
        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Clean up default cube
        self._cleanup_default_objects()
    
    def _cleanup_default_objects(self):
        """Remove default Blender objects."""
        if 'Cube' in bpy.data.objects:
            bpy.data.objects['Cube'].select_set(True)
            bpy.ops.object.delete()
    
    def create_road(self, size=1000, location=(0, 0, 0)):
        """
        Create a road surface.
        
        Args:
            size: Size of the road plane
            location: Location of the road
        """
        # Create a new plane
        bpy.ops.mesh.primitive_plane_add(size=size, location=location)
        
        # Get the newly created plane
        plane = bpy.context.active_object
        
        # Create a new material
        material = bpy.data.materials.new(name="RoadMaterial")
        
        # Set the material's diffuse color to black
        material.diffuse_color = (0, 0, 0, 1)  # RGBA
        
        # Assign the material to the plane
        if len(plane.data.materials) > 0:
            plane.data.materials[0] = material
        else:
            plane.data.materials.append(material)
    
    def setup_environment(self):
        """Setup the basic Blender environment (camera, lighting, road)."""
        # Clear existing objects and materials
        bpy.ops.object.select_all(action='SELECT')
        bpy.ops.object.delete()
        
        for material in bpy.data.materials:
            bpy.data.materials.remove(material)
        
        # Add camera
        bpy.ops.object.camera_add()
        camera = bpy.data.objects['Camera']
        camera.location = (0, -2, 1.2)
        camera.rotation_euler = (math.radians(-90), math.radians(180), math.radians(180))
        camera.data.lens = 30
        
        # Set the new camera as the active camera for the scene
        bpy.context.scene.camera = camera
        
        # Add lighting
        bpy.ops.object.light_add(type='POINT', location=(-1, -1, 5))
        bpy.context.object.data.type = 'SUN'
        bpy.context.object.data.energy = 5
        
        # Create road
        self.create_road()
    
    def setup_car(self, name, status=True, indicator=False):
        """
        Setup a car object in the scene.
        
        Args:
            name: Name of the car
            status: Whether the car is moving (green) or stationary (red)
            indicator: Whether to show indicator lights
        """
        car_obj_path = os.path.join(self.obj_assets_path, "Car.obj")
        
        if not os.path.exists(car_obj_path):
            print(f"Warning: Car model not found at {car_obj_path}")
            return
        
        try:
            bpy.ops.wm.obj_import(
                filepath=car_obj_path, 
                directory=self.obj_assets_path, 
                files=[{"name": "Car.obj", "name": "Car.obj"}]
            )
            
            # Rename object
            car = Model(name)
            
            # Set car color based on status
            if not status:
                # Red for stationary
                bpy.data.materials["CarBody" + car.id].node_tree.nodes["Principled BSDF"].inputs[0].default_value = (0.00120701, 0.800529, 0, 1)
            
            # Set indicator lights
            if indicator:
                bpy.data.materials["Aux" + car.id].node_tree.nodes["Principled BSDF"].inputs[0].default_value = (0.801157, 0.000572751, 0, 1)
                
        except Exception as e:
            print(f"Error setting up car {name}: {e}")
    
    def setup_pedestrian(self, name):
        """
        Setup a pedestrian object in the scene.
        
        Args:
            name: Name of the pedestrian
        """
        pedestrian_obj_path = os.path.join(self.obj_assets_path, "Pedestrian.obj")
        
        if not os.path.exists(pedestrian_obj_path):
            print(f"Warning: Pedestrian model not found at {pedestrian_obj_path}")
            return
        
        try:
            bpy.ops.wm.obj_import(
                filepath=pedestrian_obj_path,
                directory=self.obj_assets_path,
                files=[{"name": "Pedestrian.obj", "name": "Pedestrian.obj"}]
            )
            
            # Rename object
            pedestrian = Model(name)
            
        except Exception as e:
            print(f"Error setting up pedestrian {name}: {e}")
    
    def load_detection_data(self, data_path):
        """
        Load detection data from file.
        
        Args:
            data_path: Path to detection data file
            
        Returns:
            List of detection data
        """
        if not os.path.exists(data_path):
            print(f"Warning: Detection data not found at {data_path}")
            return []
        
        try:
            with open(data_path, 'r') as f:
                data = f.readlines()
            return [line.strip().split() for line in data if line.strip()]
        except Exception as e:
            print(f"Error loading detection data: {e}")
            return []
    
    def load_lane_data(self, lane_path):
        """
        Load lane data from file.
        
        Args:
            lane_path: Path to lane data file
            
        Returns:
            List of lane data
        """
        if not os.path.exists(lane_path):
            print(f"Warning: Lane data not found at {lane_path}")
            return []
        
        try:
            with open(lane_path, 'r') as f:
                data = f.readlines()
            return [line.strip().split() for line in data if line.strip()]
        except Exception as e:
            print(f"Error loading lane data: {e}")
            return []
    
    def render_scene(self, scene_name="default", data_path=None, lane_path=None):
        """
        Render a complete traffic scene.
        
        Args:
            scene_name: Name of the scene
            data_path: Path to detection data
            lane_path: Path to lane data
        """
        print(f"Rendering scene: {scene_name}")
        
        # Setup environment
        self.setup_environment()
        
        # Load and process detection data
        if data_path:
            detection_data = self.load_detection_data(data_path)
            for detection in detection_data:
                if len(detection) >= 4:
                    obj_type = detection[0]
                    x, y, z = float(detection[1]), float(detection[2]), float(detection[3])
                    
                    if obj_type == "car":
                        self.setup_car(f"car_{len(detection_data)}", status=True)
                    elif obj_type == "pedestrian":
                        self.setup_pedestrian(f"pedestrian_{len(detection_data)}")
        
        # Load and process lane data
        if lane_path:
            lane_data = self.load_lane_data(lane_path)
            # Process lane data here
        
        # Save the scene
        scene_output_path = os.path.join(self.output_dir, f"{scene_name}.blend")
        bpy.ops.wm.save_as_mainfile(filepath=scene_output_path)
        
        # Render the scene
        render_output_path = os.path.join(self.output_dir, f"{scene_name}_render.png")
        bpy.context.scene.render.filepath = render_output_path
        bpy.ops.render.render(write_still=True)
        
        print(f"Scene saved to: {scene_output_path}")
        print(f"Render saved to: {render_output_path}")


class Model:
    """Helper class for model naming."""
    
    def __init__(self, name):
        name_parts = name.split(".")
        self.name = name_parts[0]
        if len(name_parts) > 1:
            self.id = "." + name_parts[1]
        else:
            self.id = ""


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description='Blender Traffic Scene Renderer')
    
    parser.add_argument(
        '--project_path', 
        type=str, 
        default=os.getcwd(),
        help='Path to project root directory'
    )
    
    parser.add_argument(
        '--output_dir', 
        type=str, 
        default=None,
        help='Output directory for rendered scenes'
    )
    
    parser.add_argument(
        '--scene_name', 
        type=str, 
        default='traffic_scene',
        help='Name of the scene to render'
    )
    
    parser.add_argument(
        '--data_path', 
        type=str, 
        help='Path to detection data file'
    )
    
    parser.add_argument(
        '--lane_path', 
        type=str, 
        help='Path to lane data file'
    )
    
    args = parser.parse_args()
    
    # Initialize renderer
    renderer = BlenderSceneRenderer(
        project_path=args.project_path,
        output_dir=args.output_dir
    )
    
    # Render scene
    renderer.render_scene(
        scene_name=args.scene_name,
        data_path=args.data_path,
        lane_path=args.lane_path
    )


# Legacy functions for backward compatibility
def create_road():
    """Legacy function - use BlenderSceneRenderer.create_road() instead."""
    renderer = BlenderSceneRenderer()
    renderer.create_road()

def setup_env():
    """Legacy function - use BlenderSceneRenderer.setup_environment() instead."""
    renderer = BlenderSceneRenderer()
    renderer.setup_environment()

def setup_car(name, status=True, indicator=False):
    """Legacy function - use BlenderSceneRenderer.setup_car() instead."""
    renderer = BlenderSceneRenderer()
    renderer.setup_car(name, status, indicator)

def setup_pedestrian(name):
    """Legacy function - use BlenderSceneRenderer.setup_pedestrian() instead."""
    renderer = BlenderSceneRenderer()
    renderer.setup_pedestrian(name)


if __name__ == '__main__':
    main()