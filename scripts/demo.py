import os
import sys
from math import sqrt, pi
import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
from vccs_supervoxel import segment
from skspatial.objects import Plane, Points
import skimage
import pandas as pd

from alphashape import alphashape

viz = True


def get_voxel_hulls(points, spvx_id, param=1.4):
    # supervoxel 0
    spvx_points = points[:, -1] == spvx_id
    spvx = points[spvx_points, 0:3]
    mesh = alphashape(spvx, param)
    # easy measure of convexity:
    # https://users.cs.cf.ac.uk/Paul.Rosin/resources/papers/convexity-CVPR.pdf
    convexity = mesh.volume / mesh.convex_hull.volume
    # measure planar_dist
    pts = Points(spvx)
    plane = Plane.best_fit(spvx)
    dist = 0
    for pt in pts:
        dist += plane.distance_point(pt)

    planar_dist = dist / len(pts)
    std_color = np.std(skimage.color.rgb2lab(points[spvx_points, 3:6])) / len(pts)
    # color with first point, they should be the same color
    # calculate the compactness with isoperimetric quotient
    # comparison of mesh volume to sphere volume with same surface area
    # surface area to volume harder to bound
    R = sqrt(mesh.area / (4 * pi))
    V = 4 / 3 * pi * R ** 3
    # compactness is tricky, because often our supervoxels are very planar
    # the bigger ones get penalized although very flat
    compactness = mesh.volume / V
    # convexity and compactness should be high, planar_dist and mean color should be low
    cost = (0.5 * (1 - convexity) ** 2 + 0.5 * (planar_dist ** 2) + (0.1 / (1 / compactness) ** 2) * (std_color ** 2) + (0.5 / 4))
    print(f"Convexity: {convexity:.2f} planar_dist: {planar_dist:.2f} std_color: {std_color:.2f} compactness: {compactness:.2f} cost: {sqrt(cost):.2f}")  # noqa
    bad = sqrt(cost) > 0.7
    if bad:
        print("Low quality supervoxel!")
        color = np.array([210, 4, 45]) / 255.0
    else:
        color = points[spvx_points, 6:9][0] / 255.0
    mesh = mesh.as_open3d
    mesh.paint_uniform_color(color.reshape(3, 1))
    return mesh, bad, np.array([convexity, planar_dist, std_color, cost])


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
    out = segment(pts[:, :9], 0.1, 0.8)

    # out should now have an additional cluster id in pos 10
    # as well as random colors for the supervoxels to be optionally
    # visualized
    assert out.shape[1] == 10

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(out[:, :3])
    pcd.colors = o3d.utility.Vector3dVector(out[:, 6:9] / 255.0)
    # or original colors
    # pcd.colors = o3d.utility.Vector3dVector(out[:, 3:6] / 255.0)

    vis = None
    if viz:
        app = gui.Application.instance
        app.initialize()
        vis = o3d.visualization.O3DVisualizer("Open3D - 3D Text", 1024, 768)
        vis.show_settings = True
        vis.add_geometry("pcd", pcd)

    first_n = 80

    data = []
    for spvx_id in np.unique(out[:, -1])[:first_n]:
        # TODO: our measure is very adhoc. Can we calculate this
        # for all and throw out the bottom 20% of supervoxels?
        hull, bad, stats = get_voxel_hulls(out, spvx_id)
        data.append(stats)
        if viz and bad:
            vis.add_geometry(f"trimesh_{spvx_id}", hull)
    if viz:
        app.add_window(vis)
        app.run()

    df = pd.DataFrame(data, columns=["convexity", "planar_dist", "std_color", "cost"])
    prefix = os.path.split(fname.split(".")[0])[-1]
    df.to_csv(f"./out/{prefix}.csv")
    # can we get 3D boundaries with alpha shape?
    # https://alphashape.readthedocs.io/en/latest/readme.html?highlight=concave%20hull#generate-an-alpha-shape-alpha-2-0-concave-hull
