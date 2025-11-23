import numpy as np
import os

class TempCamera:
    def __init__(self, quaternion, translation, w, h, image_name, image_id, points2D):
        self.q = quaternion
        self.t = translation
        self.w = w
        self.h = h
        self.image_name = image_name
        self.image_id = image_id
        self.points2D = points2D
        
def create_points3D_text(output_path, pcd, dummy_error=1.0, dummy_track= (0, 0)):
    """
    Create a COLMAP-compatible points3D.txt file from a point cloud.

    Args:
        output_path (str): Path to save the points3D.txt file.
        pcd (np.ndarray): Point cloud data of shape (N, 6) where each row is (X, Y, Z, R, G, B).
    """

    with open(output_path, 'w') as f:
        f.write('# 3D point list with one line of data per point:\n')
        f.write('#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n')
        f.write('# Number of points: {}\n'.format(pcd.shape[0]))
        for i, point in enumerate(pcd):
            X, Y, Z, R, G, B = point
            dummy_track_str = ' '.join(map(str, dummy_track))
            f.write(f'{i} {X} {Y} {Z} {int(R)} {int(G)} {int(B)} {dummy_error} {dummy_track_str}\n')
    print(f'points3D.txt file saved to {output_path}')

def create_extrinsics_text(output_path, cameras):
    """
    Create a COLMAP-compatible extrinsics.txt file from camera poses.

    Args:
        output_path (str): Path to save the extrinsics.txt file.
        cameras (Camera)
    """
    with open(output_path, 'w') as f:
        f.write('# Image list with two lines of data per image:\n')
        f.write('#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n')
        f.write('#   POINTS2D[] as (X, Y, POINT3D_ID)\n')
        f.write('# Number of images: {}, mean observations per image: \n'.format(len(cameras)))
        for i, camera in enumerate(cameras):
            qx, qy, qz, qw = camera.q
            tx, ty, tz = camera.t
            f.write(f'{camera.image_id} {qw} {qx} {qy} {qz} {tx} {ty} {tz} {camera.image_id} {camera.image_name}\n')
            s = ''
            for point2D in camera.points2D:
                x, y, point3D_id = point2D
                s+= f'{x} {y} {point3D_id} '
            f.write(s.strip() + '\n')
    print(f'extrinsics.txt file saved to {output_path}')

def create_intrinsics_text(output_path, cameras, fx, fy, cx, cy, camera_model = 'PINHOLE'):
    with open(output_path, 'w') as f:
        f.write('# Camera list with one line of data per camera:\n')
        f.write('#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n')
        f.write('# Number of cameras: {}\n'.format(len(cameras)))
        for i, camera in enumerate(cameras):
            f.write(f'{camera.image_id} {camera_model} {camera.w} {camera.h} {fx} {fy} {cx} {cy}\n')
    print(f'intrinsics.txt file saved to {output_path}')