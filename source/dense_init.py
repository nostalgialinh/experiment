import sys

sys.path.append('../')
sys.path.append("../submodules")
sys.path.append('../submodules/RoMa')
sys.path.append('./submodules/gaussian-splatting/')

import torch
import numpy as np
from utils.sh_utils import RGB2SH
import cv2
import open3d as o3d
import argparse
from PIL import Image
import os

def dense_init_gaussians( 
                          fx, fy, cx, cy, w, h,
                          images,
                          pred_maps,
                          W2Cs):

    # 3) TSDF integration with Open3D

    with torch.no_grad():
        voxel_length = 0.001  # adjust resolution
        sdf_trunc = 0.04
        tsdf_volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=voxel_length,
            sdf_trunc=sdf_trunc,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
        )

        for i in range(len(images)):
            color_raw = o3d.geometry.Image(
                cv2.cvtColor(np.array(images[i]), cv2.COLOR_BGR2RGB)
            )
            depth_raw = o3d.geometry.Image((pred_maps[i]).astype(np.uint16))

            rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
                color_raw,
                depth_raw,
                depth_scale=1.0,
                depth_trunc=5.0,
                convert_rgb_to_intensity=False
            )

            intrinsic = o3d.camera.PinholeCameraIntrinsic(
                width=w, height=h,
                fx=fx, fy=fy,
                cx=cx, cy=cy
            )
            C2W = np.linalg.inv(W2Cs[i])

            tsdf_volume.integrate(rgbd_image, intrinsic, C2W)


        # mesh = tsdf_volume.extract_triangle_mesh()
        pcd_o3d = tsdf_volume.extract_point_cloud()


        points_np = np.asarray(pcd_o3d.points)
        colors_np = np.asarray(pcd_o3d.colors)

        all_new_xyz = torch.from_numpy(points_np).float()
        N = all_new_xyz.shape[0]
        print(f"Number of splats after dense init: {N}")

        all_new_rgb = torch.from_numpy(colors_np).float()
        all_new_features_dc = RGB2SH(all_new_rgb.detach().clone() / 255.).unsqueeze(1)

    return all_new_xyz, all_new_features_dc


def load_data_from_npz(images_path, npz_path):
    # Load the .npz file
    data = np.load(npz_path, allow_pickle=True)
    keys = list(data.keys())

    # Separate keys ending with '_W2C' and '_pred'
    w2c_keys = [k for k in keys if k.endswith('_W2C')]
    pred_keys = [k for k in keys if k.endswith('_pred')]

    # Extract image base names from keys by removing suffix
    image_names = [k[:-4]  for k in w2c_keys]  # remove '_W2C'


    W2Cs = []
    pred_maps = []
    images = []

    for name in image_names:
        # Load W2C and pred_maps from npz
        W2Cs.append(data[f"{name}_W2C"])
        pred_maps.append(data[f"{name}_pred"])

        # Load corresponding image
        image_path = os.path.join(images_path, name)
        image = Image.open(image_path).convert('RGB')
        images.append(image)

    return images, pred_maps, W2Cs

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--fx', type=float, required=True)
    parser.add_argument('--fy', type=float, required=True)
    parser.add_argument('--cx', type=float, required=True)
    parser.add_argument('--cy', type=float, required=True)
    parser.add_argument('--w', type=int, required=True)
    parser.add_argument('--h', type=int, required=True)
    parser.add_argument('--images_path', type=str, required=True)
    parser.add_argument('--data_path', type=str, required=True, help='Path to the folder containing pred_maps and W2Cs')
    parser.add_argument('--output_path', type=str, required=True)

    args = parser.parse_args()
    fx = args.fx
    fy = args.fy
    cx = args.cx
    cy = args.cy
    w = args.w
    h = args.h
    images_path = args.images_path
    data_path = args.data_path
    
    images, pred_maps, W2Cs = load_data_from_npz(images_path, data_path)

    all_new_xyz, all_new_features_dc = dense_init_gaussians(
        fx=fx, fy=fy, cx=cx, cy=cy, w=w, h=h,
        images=images,
        pred_maps=pred_maps,
        W2Cs=W2Cs
    )
    os.makedirs(args.output_path, exist_ok=True)
    torch.save(all_new_xyz, args.output_path + '/all_new_xyz.pt')
    torch.save(all_new_features_dc, args.output_path + '/all_new_features_dc.pt')