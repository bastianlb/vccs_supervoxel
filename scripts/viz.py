import sys

import open3d as o3d
import numpy as np


if __name__ == "__main__":
    data = []
    with open(sys.argv[1], 'r+') as f:
        while True:
            line = f.readline()
            if not line:
                break
            data.append([float(x) for x in line.split(" ")])

    pts = np.array(data)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts[:, :3])
    pcd.colors = o3d.utility.Vector3dVector(pts[:, 3:] / 255.0)

    o3d.visualization.draw_geometries([pcd])



