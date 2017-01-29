import cv2
import argparse
import numpy as np
from conditions import *
from bgextraction import bg_path
from feature import feature_vector
from distance import distance_matrix
import time
import labeling

ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", required=True,
                help="path to the video file")
try:
    args = vars(ap.parse_args())
    path = args["video"]
except SystemExit:
    path = 'data/z-cam_point1.mp4'


cap = cv2.VideoCapture(path)
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

bg = cv2.imread(bg_path, cv2.IMREAD_COLOR)


def masking(img, background, poly_x, poly_y, offset_v=0, offset_h=0):
    n_rows = img.shape[0]
    n_cols = img.shape[1]
    bg_hsv = cv2.cvtColor(background, cv2.COLOR_BGR2HSV)
    mask = np.zeros([n_rows, n_cols], dtype=np.int)
    obj_x = []
    obj_y = []

    for i in range(offset_v - 1, n_rows - 1):
        for j in range(offset_h - 1, n_cols - 1):
            bg_hue = int(bg_hsv[i, j, 0])

            if not on_field_background(bg_hue) or not in_polygon(j, i, poly_x, poly_y) or not threshold(img[i, j]):
                continue

            mask[i, j] = 1
            obj_x.append(i)
            obj_y.append(j)
    return mask, np.array(obj_x, dtype=np.int), np.array(obj_y, dtype=np.int)


def mask_background(img, mask, mask_color=(0, 0, 0)):
    result = np.array(img, copy=True)
    result[mask == 0] = mask_color
    result[mask == 1] = (255, 255, 255)
    return result

while(1):
    ret, f = cap.read()
    if ret is False:
        break

    diff = cv2.absdiff(f, bg)

    # the boundary points of a football field (295, 95), (640, 115) ... Left cam
    # x = (295, 640, 640, 0)
    # y = (95, 115, 380, 250)

    # Right cam
    # x = (4, 345, 640, 0)
    # y = (129, 120, 295, 385)

    x = (0, 160, 465, 615)
    y = (330, 225, 225, 330)

    obj_mask, obj_x, obj_y = masking(diff, bg, x, y)
    fg = mask_background(diff, obj_mask)

    kernel = np.ones((3, 3), np.uint8)
    fg = cv2.dilate(fg, kernel, iterations=1)

    fvs, fvs_raw = feature_vector(fg, obj_x, obj_y, x_max=height, y_max=width, increment=4)
    distances = distance_matrix(fvs)
    start_time = time.time()
    print("Distance matrix %s (%ds)" % (str(distances.shape), time.time() - start_time))
    np.savetxt("distance.txt.gz", distances, '%5.8f')
    np.savetxt("foreground.txt", fvs, '%5.8f')
    np.savetxt("foreground_unnormalized.txt", fvs_raw, '%5.0f')
    exit()

    # labeling.show_label(fg)

    both = np.hstack((f, fg))
    cv2.imshow('Show clustering', both)

    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break


cap.release()
cv2.destroyAllWindows()