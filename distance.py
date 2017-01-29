import numpy as np
np.set_printoptions(threshold=np.nan)
import math
import time

def euclidean_distance(p, q):
    return math.sqrt(np.power(p-q,2).sum())

# Maximum 100K points otherwise python process crashes
# At 1K points, computation takes ~2s
# At 6K points, computation takes ~2m, file size 150MB gzipped
# At 10K points, computation takes ~4m
# At 15K points, computation takes ~9m, file size 320MB gzipped
# At 22K points, computation takes ~20m, file size 5GB without gzipped!

def distance_matrix(points):
    n_points = points.shape[0]
    dm = np.zeros([n_points, n_points])
    for i in range(n_points):
        for j in range(n_points):
            if i > j:
                dm[i, j] = dm[j, i]
                continue
            if i < j:
                p = points[i, :]
                q = points[j, :]
                dm[i, j] = euclidean_distance(p, q)
    return dm

# def main():
#     pts = np.loadtxt("foreground.txt")
#     print("points:", pts.shape)
#     start_time = time.time()
#     distances = distance_matrix(pts)
#     print("Distance matrix %s (%ds)" % (str(distances.shape), time.time() - start_time))
#     np.savetxt("distance.txt.gz", distances, '%5.8f')
#
# main()
# print("Done!")