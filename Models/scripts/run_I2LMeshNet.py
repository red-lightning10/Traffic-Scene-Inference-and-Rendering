"""
I2L-MeshNet Human Pose Estimation Script

This script uses the I2L-MeshNet submodule to estimate 3D human pose and generate mesh.
"""

import sys
import os
import os.path as osp
import argparse
import numpy as np
import cv2
import torch
import torchvision.transforms as transforms
from torch.nn.parallel.data_parallel import DataParallel
import torch.backends.cudnn as cudnn
from pathlib import Path

# Add I2L-MeshNet submodule to path
sys.path.insert(0, osp.join('..', 'I2LMeshNet', 'main'))
sys.path.insert(0, osp.join('..', 'I2LMeshNet', 'data'))
sys.path.insert(0, osp.join('..', 'I2LMeshNet', 'common'))

try:
    from config import cfg
    from model import get_model
    from utils.preprocessing import load_img, process_bbox, generate_patch_image
    from utils.transforms import pixel2cam, cam2pixel
    from utils.vis import vis_mesh, save_obj, render_mesh
    
    sys.path.insert(0, cfg.smpl_path)
    from smplpytorch.pytorch.smpl_layer import SMPL_Layer
except ImportError as e:
    print(f"Error importing I2L-MeshNet modules: {e}")
    print("Make sure the I2LMeshNet submodule is properly initialized")
    sys.exit(1)


def parse_args():
    parser = argparse.ArgumentParser(description='I2L-MeshNet Human Pose Estimation')
    parser.add_argument('--gpu', type=str, dest='gpu_ids', default='0', 
                       help='GPU device IDs (e.g., "0" or "0,1")')
    parser.add_argument('--stage', type=str, dest='stage', default='param', 
                       help='Stage: param or lixel')
    parser.add_argument('--test_epoch', type=str, dest='test_epoch', default='60', 
                       help='Test epoch number')
    parser.add_argument('--input', type=str, required=True, 
                       help='Input image path')
    parser.add_argument('--bbox', type=str, required=True, 
                       help='Bounding box coordinates (x1,y1,x2,y2)')
    parser.add_argument('--output', type=str, default='outputs/human_pose', 
                       help='Output directory')
    parser.add_argument('--model_path', type=str, default='../I2LMeshNet/snapshot_60.pth.tar',
                       help='Path to model weights')
    args = parser.parse_args()
    
    # Process GPU IDs
    if '-' in args.gpu_ids:
        gpus = args.gpu_ids.split('-')
        gpus[0] = int(gpus[0])
        gpus[1] = int(gpus[1]) + 1
        args.gpu_ids = ','.join(map(lambda x: str(x), list(range(*gpus))))
    
    return args


def process_human_pose(image_path, bbox_str, model, smpl_layer, output_dir):
    """
    Process a single image for human pose estimation.
    
    Args:
        image_path: Path to input image
        bbox_str: Bounding box string "x1,y1,x2,y2"
        model: Loaded I2L-MeshNet model
        smpl_layer: SMPL layer
        output_dir: Output directory
        
    Returns:
        Dictionary with pose results
    """
    # Parse bounding box
    bbox_coords = [float(x) for x in bbox_str.split(',')]
    bbox = np.array(bbox_coords)
    
    # Load and preprocess image
    img = load_img(image_path)
    img, img2bb_trans, bb2img_trans, rot, do_flip = process_bbox(img, bbox, 224)
    img_patch = generate_patch_image(img, img2bb_trans, 224)
    
    # Convert to tensor
    img_patch = torch.from_numpy(img_patch).float()
    img_patch = img_patch.unsqueeze(0).cuda()
    
    # Run inference
    with torch.no_grad():
        pose_3d, pose_3d_vis, mesh_3d, mesh_3d_vis = model(img_patch)
    
    # Convert to numpy
    pose_3d = pose_3d.cpu().numpy()[0]
    mesh_3d = mesh_3d.cpu().numpy()[0]
    
    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    image_name = Path(image_path).stem
    
    # Save pose data
    pose_file = output_path / f"{image_name}_pose.npy"
    np.save(pose_file, pose_3d)
    
    # Save mesh data
    mesh_file = output_path / f"{image_name}_mesh.npy"
    np.save(mesh_file, mesh_3d)
    
    # Save OBJ file
    obj_file = output_path / f"{image_name}_mesh.obj"
    save_obj(mesh_3d, smpl_layer.th_faces.numpy(), str(obj_file))
    
    return {
        'pose_3d': pose_3d,
        'mesh_3d': mesh_3d,
        'pose_file': str(pose_file),
        'mesh_file': str(mesh_file),
        'obj_file': str(obj_file)
    }


def main():
    """Main function for I2L-MeshNet human pose estimation."""
    args = parse_args()
    
    # Setup configuration
    cfg.set_args(args.gpu_ids, args.stage)
    cudnn.benchmark = True
    
    # SMPL joint set
    joint_num = 29
    vertex_num = 6890
    
    # Initialize SMPL layer
    smpl_layer = SMPL_Layer(gender='neutral', model_root=cfg.smpl_path + '/smplpytorch/native/models')
    
    # Load model
    if not osp.exists(args.model_path):
        print(f"Model not found at {args.model_path}")
        print("Please download the model weights or specify correct path with --model_path")
        return 1
    
    print(f'Loading checkpoint from {args.model_path}')
    model = get_model(vertex_num, joint_num, 'test')
    model = DataParallel(model).cuda()
    
    ckpt = torch.load(args.model_path)
    model.load_state_dict(ckpt['network'], strict=False)
    model.eval()
    
    # Process image
    try:
        results = process_human_pose(
            args.input, 
            args.bbox, 
            model, 
            smpl_layer, 
            args.output
        )
        
        print(f"Results saved to: {args.output}")
        print(f"Pose data: {results['pose_file']}")
        print(f"Mesh data: {results['mesh_file']}")
        print(f"OBJ file: {results['obj_file']}")
        
    except Exception as e:
        print(f"Error processing image: {e}")
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())