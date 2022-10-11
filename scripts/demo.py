import sys
import numpy as np
import open3d as o3d
from vccs_supervoxel import segment


if __name__ == "__main__":
    if len(sys.argv) <= 1:
        print("Requires pointcloud as input argument")
        exit(1)
    data = []
    fname = sys.argv[1]
    if fname.endswith("npy"):
        pts = np.load(fname)
    else:
        # assumes xyz format. Do not currently support ply
        with open(fname, 'r+') as f:
            while True:
                line = f.readline()
                if not line:
                    break
                data.append([float(x) for x in line.split(" ")])

        pts = np.array(data)

    # for now the function calculates surface normals for the clustering
    # on it's own. Later we could even use scannet's surface normals
    out = segment(pts[:, :9], 0.1, 0.6)

    # out should now have an additional cluster id in pos 10
    # as well as random colors for the supervoxels to be optionally
    # visualized
    assert out.shape[1] == 10

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(out[:, :3])
    pcd.colors = o3d.utility.Vector3dVector(out[:, 6:9] / 255.0)
    # or original colors
    # pcd.colors = o3d.utility.Vector3dVector(out[:, 3:6] / 255.0)

    o3d.visualization.draw_geometries([pcd])
