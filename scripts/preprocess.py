import sys
from pathlib import Path
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from vccs_supervoxel import segment

import multiprocessing
from joblib import Parallel, delayed

viz = False

num_cores = multiprocessing.cpu_count()


def calculate_supervoxels(res):
    vals = []
    for npy in Path(dirname).rglob("*.npy"):
        pts = np.load(npy)
        print(f"Processing pointcloud {npy}")

        # for now the function calculates surface normals for the clustering
        # on it's own. Later we could even use scannet's surface normals
        out = segment(pts[:, :6], 0.1, res)

        # out should now have an additional cluster id in pos 10
        # as well as random colors for the supervoxels to be optionally
        # visualized
        assert out.shape[1] == 10

        if viz:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(out[:, :3])
            pcd.colors = o3d.utility.Vector3dVector(out[:, 6:9] / 255.0)
            # or original colors
            # pcd.colors = o3d.utility.Vector3dVector(out[:, 3:6] / 255.0)

            o3d.visualization.draw_geometries([pcd])

        catted = np.concatenate((pts, out[:, -1:]), axis=1)[:, -2:].astype(np.int16)

        supervoxels = np.unique(catted[:, 1])
        for spxl in supervoxels:
            x = catted[catted[:, 1] == spxl, 0]
            mx = np.argmax(np.bincount(x))
            intersection = np.sum(x == mx) / x.shape
            vals.append(intersection[0])

    plt.hist(vals, bins=100)
    plt.yscale('log')
    plt.title(f"Supervoxel (K={len(supervoxels)}) Intersection at {res} resolution. Std: {np.std(vals)}")
    plt.savefig(f"plots/{res}.png")


if __name__ == "__main__":
    if len(sys.argv) <= 1:
        print("Requires pointcloud as input argument")
        exit(1)
    data = []
    dirname = sys.argv[1]
    resolutions = np.linspace(0.1, 1, 10)
    Parallel(n_jobs=num_cores)(delayed(calculate_supervoxels(res)) for res in resolutions)
