import rasterio
import rasterio.features
from rasterio.plot import show
import geopandas as gpd
import numpy as np
import glob
from numpy import linalg as LA
from scipy.spatial import Delaunay, ConvexHull
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import AgglomerativeClustering
from scipy import stats
from scipy.spatial import distance
from sympy import Point, Segment
from sklearn.cluster import MeanShift, KMeans
import scipy.optimize
import functools
import math
from scipy.optimize import leastsq
import numpy.ma as ma
from sklearn.cluster import AffinityPropagation
from sklearn import metrics


file = "./data/train_val/train/tile_9_mask.tif"
src = rasterio.open(file)
train_roof = gpd.read_file(
    f'filtred_buildings/test.shp', bbox=src.bounds)


global point_plane_distances
point_plane_distances = []


class PlaneProcessor:
    def __init__(self):
        pass

    def plane(self, x, y, params):
        a = params[0]
        b = params[1]
        c = params[2]
        z = a*x + b*y + c
        return z
    def error(self, params, points):
        result = 0
        for (x, y, z) in points:
            plane_z = self.plane(x, y, params)
            diff = (plane_z - z)
            result += diff**2
        return result

    def cross(self, a, b):
        return [a[1]*b[2] - a[2]*b[1],
                a[2]*b[0] - a[0]*b[2],
                a[0]*b[1] - a[1]*b[0]]

    def f_min(self, X, p):
        plane_xyz = p[0:3]
        distance = (plane_xyz*X.T).sum(axis=1) + p[3]
        return distance / np.linalg.norm(plane_xyz)

    def residuals(self, params, signal, X):
        return self.f_min(X, params)

    def plane_equation(self, points, params=[0.1, 0.2, 0.1, 0]):
        sol = leastsq(self.residuals, params, args=(
            None, np.array(points).T))[0]
        return sol

    def plot_plane(self, points, params, ax):
        fun = functools.partial(self.error, points=points)
        params0 = params
        res = scipy.optimize.minimize(fun, params0)

        a = res.x[0]
        b = res.x[1]
        c = res.x[2]

        point = np.array([0.0, 0.0, c])
        normal = np.array(self.cross([1, 0, a], [0, 1, b]))
        d = -point.dot(normal)
        xs, ys, zs = zip(*points)

        ax.scatter(xs, ys, zs)
        aux = np.array(points)
        max_x = np.max(aux[:, 0])
        min_x = np.min(aux[:, 0])

        max_y = np.max(aux[:, 1])
        min_y = np.min(aux[:, 1])

        xx, yy = np.meshgrid([min_x, max_x], [min_y, max_y])
        z = (-normal[0] * xx - normal[1] * yy - d) * 1. / normal[2]
        # ax.plot_surface(xx, yy, z, alpha=0.2, color=[0, 1, 0])
    def estimate_height(self, equation, p):
        return (-(equation[0] * p[0]) - (equation[1] * p[1]) - equation[3])/equation[2]
    def update_plane_equation_from_neighbor_points(self, neighbor, tri, used_points, equation_points, equation, norm):
        continue_adding = False
        if(neighbor != -1):
            filtered = tri.simplices[neighbor][np.logical_not(np.isin(
                tri.simplices[neighbor], used_points))]

            for idx, point in enumerate(tri.points[filtered]):
                # CALCULATE PLANE FITTING ERROR VALUE
                pfe = (
                    self.f_min(np.array([*tri.points[equation_points], point]).T, equation)**2).sum()

                height_diff = self.estimate_height(equation, point) - point[2]
                point_plane_distances.append(pfe)

                if(pfe*norm <= 0.2 and height_diff * norm <= 0.15):
                    continue_adding = True

                    used_points.append(filtered[idx])
                    equation_points = np.append(equation_points, filtered[idx])
                    equation = self.plane_equation(tri.points[equation_points])

            return used_points, equation_points, equation, continue_adding
        else:
            return used_points, equation_points, equation, False
    def coplanar_and_noncoplanar_extraction(self, non_ground):

        tri = Delaunay(non_ground)
        mask = []

        for _, p in enumerate(tri.points):
            # FIND TRIANGLE WHICH CONTAINS POINT P
            idx = tri.find_simplex(p)
            if(idx == -1):
                print("Non valid point")
                mask.append(False)
            else:
                # GET TRIANGLE NEIGHBOORS AND GET ALL THE NEIGHBOR POINTS
                neighbors = tri.neighbors[idx]
                simplices = [tri.simplices[n] for n in neighbors]

                n_points = []
                for simplice in simplices:
                    n_points.append(tri.points[simplice])

                n_points = np.array(n_points)
                n_points = n_points.reshape(
                    n_points.shape[0]*n_points.shape[1], 3)

                n_points = np.unique(n_points, axis=0)

                # NORMALIZE EIGEN VALUE
                eig = LA.norm(LA.eigvals(np.cov(n_points.T)))
                if(eig <= 0.016):
                    mask.append(True)
                else:
                    mask.append(False)

        return np.array(mask), tri
    def extract_lines(self, non_ground, norm):

        heights = non_ground[:, 2]
        hight_level = np.max(heights)
        min_h = np.min(np.where(heights == 0, 9999, heights))

        lines = []

        while hight_level >= min_h:
            points_in_range = []
            for i in range(non_ground.shape[0]):
                if(non_ground[i][2] >= hight_level-(0.15/norm) and non_ground[i][2] <= hight_level+(0.15/norm)):
                    points_in_range.append(non_ground[i])

            candidates = np.array(list(map(lambda x: x[0:2], points_in_range)))
            try:
                if(len(candidates) >= 2):
                    lines.append([candidates, hight_level, *stats.linregress([x[0]
                                                                              for x in candidates], [x[1] for x in candidates])])
            except Exception as e:
                print(e)
            hight_level = hight_level-(0.4/norm)

        return lines

    def filter_planes_by_std(self, planes):
        final_planes = []
        mean_std = np.mean(
            list(map(lambda x: np.std(np.array(x["points"])[:, 2]), planes)))

        for pl in planes:
            if(np.std(np.array(pl["points"])[:, 2]) <= mean_std and len(pl["points"]) > 10):
                final_planes.append(pl)

        return final_planes
    def get_mid_points(self, cluster_lines):
        midpoints = []

        for line in cluster_lines:
            x = np.array([point[0] for point in line[0]])
            y = line[3] + line[2]*x
            z = [line[1] for _ in range(len(x))]

            p1, p2 = Point(np.min(x), np.min(y), np.min(z)), Point(
                np.max(x), np.max(y), np.max(z))
            s = Segment(p1, p2)

            midpoints.append([float(s.midpoint[0]), float(
                s.midpoint[1]), float(s.midpoint[2])])

        return midpoints

    def create_rectangle(self, points):
        # find Xmax and Xmin
        Xmax = np.max(points[:, 0])
        Xmin = np.min(points[:, 0])

        # find Ymax and Ymin
        Ymax = np.max(points[:, 1])
        Ymin = np.min(points[:, 1])

        Zmax = np.max(points[:, 2])
        Zmin = np.min(points[:, 2])

        return [[Xmin, Xmax], [Ymin, Ymax], [Zmin, Zmax]]

    def is_inside(self, rectangle, point):
        isInside = False

        if(point[0] >= rectangle[0][0] and point[0] <= rectangle[0][1]):
            if(point[1] >= rectangle[1][0] and point[1] <= rectangle[1][1]):
                if(point[2] >= rectangle[2][0] and point[2] <= rectangle[2][1]):
                    isInside = True
        return isInside

    def used_point_ratio(self, planes, non_ground, norm):
        filtered = []
        for possible_plane in planes:
            points = possible_plane["points"]
            rectangle = self.create_rectangle(points)
            mask = np.apply_along_axis(functools.partial(
                self.is_inside, rectangle), 1, non_ground)
            count = np.sum(mask)
            width = (rectangle[0][1]-rectangle[0][0])*norm
            length = (rectangle[1][1]-rectangle[1][0])*norm

            area = length * width
            if(len(points)/count > 0.65):
                filtered.append(possible_plane)
            if(width > 1.5):
                filtered.append(possible_plane)
            if(area > 3.5):
                filtered.append(possible_plane)
        return filtered


np.seterr('raise')

for index, row in train_roof.iterrows():
    plane_processor = PlaneProcessor()

    window = rasterio.features.geometry_window(
        src, [row["geometry"]], pad_x=0, pad_y=0, north_up=True, rotated=False, pixel_precision=1)

    data = src.read(window=window)[0]

    non_ground = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if(data[i][j] > 3.0):
                non_ground.append([i, j, data[i][j]])

    # NORMALIZE DATA
    non_ground = np.array(non_ground)
    non_ground_max = int(np.max(non_ground))
    non_ground = non_ground/non_ground_max

    # STEP_1:   GET COPLANAR AND NON-COPLANAR SETS
    mask, tri = coplanar_and_noncoplanar_extraction(
        non_ground)

    # STEP_2:    EXTRACT SEGMENTS FROM ROOF BASED ON HEIGHT
    cluster_lines = extract_lines(non_ground, non_ground_max)

    # STEP_3:   GET MIDPOINTS FROM LINES
    midpoints = get_mid_points(cluster_lines)

    # STEP_4:   FOR EACH MID POINT
    used_points = []
    planes = []

    distances = distance.cdist(midpoints, tri.points, 'cityblock')

    while (np.any(mask) == True):

        current_plane = dict(points=[], equation=[0, 0, 0, 0])
        equation_points = []

        # GET THE CLOSEST COPLANAR POINT TO A NON COPLANAR
        idx = [math.inf, math.inf]
        for d in distances:
            if idx[0] >= np.min(d):
                idx = [np.where(d == np.min(d))[0][0], np.min(d)]

        # FIND THE TRIANGLE OF THIS POINT
        p = tri.points[idx[0]]
        triangle_idx = tri.find_simplex(p)

        # USE POINTS OF THS TRIANGLE TO GENERATE THE BASE PLANE EQUATION
        used_points = [*used_points, idx[0],
                       *tri.simplices[triangle_idx]]

        equation_points = [idx[0], *tri.simplices[triangle_idx]]
        equation_points = np.unique(equation_points)

        equation = plane_equation(tri.points[equation_points])
        # GET NEIGHBOOR POINTS OF THE POINT TO SEE IF THEY FIT IN THE PLANE EQUATION
        neighbors = tri.neighbors[triangle_idx]

        while len(neighbors) > 0:
            continue_adding = []
            for neighbor in neighbors:

                used_points, equation_points, equation, aux = update_plane_equation_from_neighbor_points(
                    neighbor, tri, used_points, equation_points, equation, non_ground_max)
                continue_adding.append(aux)

            if(np.any(continue_adding)):

                new_neighboors = []
                for neighbor in neighbors:
                    new_neighboors = [*new_neighboors,
                                      *tri.neighbors[neighbor]]
                neighbors = np.unique(
                    [*np.extract(np.logical_not(continue_adding), neighbors), *new_neighboors])

            else:
                neighbors = []

        print(
            "FINISHED NEIGHBOOR EXPLORATION, APPENDING EQUATION AND POINTS TO SEGMENTS SET")
        current_plane["equation"] = equation
        current_plane["points"] = tri.points[equation_points]
        planes.append(current_plane)

        # DELETE  USED POINTS

        mask[equation_points] = False
        for d in distances:
            d[equation_points] = math.inf

        print("points left ", len(np.extract(mask, tri.points)))

        if(len(point_plane_distances) > 0):
            print("mean distances: ", np.mean(point_plane_distances))
            print("min distances: ", np.min(point_plane_distances))
            print("max distances: ", np.max(point_plane_distances))

            point_plane_distances = []

    #  STEP 3.2 If the standard deviation of height of a plane is higher than the average standard deviation the plane is discarded.
    planes = filter_planes_by_std(planes)
    planes = used_point_ratio(planes, non_ground, non_ground_max)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for res in planes:
        plot_plane(res["points"], res["equation"], ax)
    plt.show()

    planes = height_filtering(planes, non_ground_max)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for res in planes:
        plot_plane(res["points"], res["equation"], ax)
    plt.show()
