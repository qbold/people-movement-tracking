import numpy as np
import time
import random
import cv2

UNCLASSIFIED = 0
NOISE = -1


# return all points within P's eps-neighborhood (including P)

def region_query(point, eps, distances):
    neighbours = np.where(distances[:, point] <= eps)[0]
    return neighbours


def expand_cluster(point, neighbours, cluster_id, eps, min_points, classifications, distances):
    classifications[point] = cluster_id

    new_neighbours_found = True
    while new_neighbours_found:
        new_neighbours = None
        for ne in neighbours:
            if classifications[ne] == UNCLASSIFIED:
                classifications[ne] = cluster_id
                ne_neighbours = region_query(ne, eps, distances)
                if ne_neighbours.size >= min_points:
                    if (new_neighbours is None):
                        new_neighbours = np.array(ne_neighbours, copy=True)
                    else:
                        new_neighbours = np.append(new_neighbours, ne_neighbours)
                continue
            if classifications[ne] == NOISE:
                classifications[ne] = cluster_id
                continue
                # If neighbour already belongs to another cluster, do nothing
        if new_neighbours is None:
            new_neighbours_found = False
        else:
            neighbours = np.append(neighbours, new_neighbours)

    return classifications


def dbscan(distances, eps, min_points):
    cluster_id = 1
    n_points = distances.shape[0]
    classifications = np.zeros([n_points, 1])
    for p in range(n_points):
        if classifications[p] != UNCLASSIFIED:
            continue
        neighbours = region_query(p, eps, distances)
        if neighbours.size < min_points:
            classifications[p] = NOISE
            continue

        # Start cluster

        classifications = expand_cluster(p, neighbours, cluster_id, eps, min_points, classifications, distances)
        cluster_id += 1
    return classifications


def region_query_test(distances):
    p = [0, 1000, 1500, 2000, 3000, 4000, 2500, 3500, 4500, 5000]
    eps = 0.1
    for i in range(len(p)):
        neighbours = region_query(p[i], eps, distances)
        print("neightbours.shape = %s, size = %d" % (str(neighbours.shape), neighbours.size))
        np.savetxt("dbscan-neighbours.txt", neighbours, '%5.0f')


def show_clusters_on_frame(classifications):
    pts = np.loadtxt("foreground_unnormalized.txt", dtype=int)
    cap = cv2.VideoCapture("data/z-cam_point1.mp4")
    _, f = cap.read()
    clusters = np.unique(classifications)
    # colors = np.array([0, 0, 0])

    # for i in range(1, clusters.size):
    #     new_color = np.array([random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)])
    #     colors = np.vstack((colors, new_color))

    for i in range(classifications.size):
        c = classifications[i, 0]
        if c == NOISE:
            continue
        # color = tuple(colors[c])
        color = (128, 50, 123)
        x = pts[i, 0]
        y = pts[i, 1]
        cv2.circle(f, (y, x), 1, color, thickness=-1)

    for c in clusters:
        members = np.where(classifications == c)[0]
        foot = bottom_most_point(members, pts)
        yellow = (0, 255, 255)
        x = foot[0]
        y = foot[1]
        cv2.circle(f, (y, x), radius=10, color=yellow, thickness=1)

    cv2.imshow("DBSCAN clustering", f)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cap.release()


def dbscan_test(distances, eps, min_points):
    print("DBSCAN Clustering: eps=%f, min points=%d" % (eps, min_points))
    start_time = time.time()
    classifications = dbscan(distances, eps, min_points)
    print("Classifications=%s (%ds)" % (str(classifications.shape), time.time() - start_time))
    clusters = np.unique(classifications)
    n_classified_points = 0
    for c in clusters:
        members = np.where(classifications == c)[0]
        n_classified_points += members.size
        if (c == UNCLASSIFIED):
            print("\n*** ERROR! ***\nUnclassified: %d pts\n" % (members.size))
            continue
        if (c == NOISE):
            print("Noise: %d pts" % (members.size))
            continue
        print("Cluster %d: %d pts" % (c, members.size))

    print("Total classified points: %d" % (n_classified_points))
    show_clusters_on_frame(classifications)
    np.savetxt("dbscan.txt", classifications, '%5.8f')


def bottom_most_point(cluster, pts):
    cluster_pts = pts[cluster, 0]
    return pts[cluster[np.argmax(cluster_pts, axis=0)], :]


def main():
    distances = np.loadtxt("distance.txt.gz")
    print("distances:", distances.shape)
    # region_query_test(distances)
    for i in range(20, 21):
        j = i / 1000.
        dbscan_test(distances, eps=j, min_points=2)


main()
print("Done!")