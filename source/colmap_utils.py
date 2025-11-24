import numpy as np
import os

class TempCamera:
    def __init__(self, quaternion, translation, image_name, image_id, points2D):
        self.q = quaternion
        self.t = translation
        self.image_name = image_name
        self.image_id = image_id
        self.points2D = points2D

class MapPoint:
    def __init__(self, point3D_id, X, Y, Z, R, G, B, error=None):
        self.point3D_id = point3D_id
        self.coord = (X, Y, Z)
        self.color = (R, G, B)
        self.error = error
        self.track = []

def create_points3D_text(output_path, map_points):
    """
    Create a COLMAP-compatible points3D.txt file from a point cloud.

    Args:
        output_path (str): Path to save the points3D.txt file.
        map_points (list of MapPoint): List of MapPoint objects representing the point cloud.
    """

    with open(output_path, 'w') as f:
        f.write('# 3D point list with one line of data per point:\n')
        f.write('#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n')
        f.write('# Number of points: {}\n'.format(len(map_points)))
        for id, point in map_points.items():
            X, Y, Z, = point.coord
            R, G, B = point.color
            dummy_error = 1.0 if point.error is None else point.error
            track_str = ' '.join(map(str, point.track))
            f.write(f'{id} {X} {Y} {Z} {int(R)} {int(G)} {int(B)} {dummy_error} {track_str}\n')
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
            qw, qx, qy, qz = camera.q
            tx, ty, tz = camera.t
            f.write(f'{camera.image_id} {qw} {qx} {qy} {qz} {tx} {ty} {tz} {camera.image_id} {camera.image_name}\n')
            s = ''
            for point2D in camera.points2D:
                x, y, point3D_id = point2D
                s+= f'{x} {y} {point3D_id} '
            f.write(s.strip() + '\n')
    print(f'images.txt file saved to {output_path}')

def create_intrinsics_text(output_path, N, w, h, fx, fy, cx, cy, camera_model = 'PINHOLE'):
    with open(output_path, 'w') as f:
        f.write('# Camera list with one line of data per camera:\n')
        f.write('#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n')
        f.write('# Number of cameras: {}\n'.format(N))
        for i in range(N):
            f.write(f'{i+1} {camera_model} {w} {h} {fx} {fy} {cx} {cy}\n')
    print(f'cameras.txt file saved to {output_path}')