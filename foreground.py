import cv2
import argparse
import numpy as np
from conditions import *
from bgextraction import bg_filepath

ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", required=True,
                help="path to the video file")
try:
    args = vars(ap.parse_args())
    path = args["video"]
except:
    path = 'data/0001_L.mp4'

cap = cv2.VideoCapture(path)
fwidth = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
fheight = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

bg = cv2.imread(bg_filepath, cv2.IMREAD_COLOR)

def masking(img, bg, polyX, polyY, offset_v=0, offset_h=0):
    n_rows = img.shape[0]
    n_cols = img.shape[1]
    bg_hsv = cv2.cvtColor(bg, cv2.COLOR_BGR2HSV)
    mask = np.zeros([n_rows, n_cols], dtype=np.int)

    for i in range(offset_v - 1, n_rows - 1):
        for j in range(offset_h - 1, n_cols - 1):
            bg_hue = int(bg_hsv[i, j, 0])

            if not on_field_background(bg_hue) or not in_polygon(j, i, polyX, polyY) or not treshold(img[i, j]):
                continue

            mask[i, j] = 1
    return mask

def mask_background(img, mask, mask_color=(0, 0, 0)):
    result = np.array(img, copy=True)
    result[mask == 0] = mask_color
    result[mask == 1] = (255, 255, 255)
    return result

# Skip 100 frames
for i in range(100):
    __, f = cap.read()

while(1):
    ret, f = cap.read()
    if ret is False:
        break

    diff = cv2.absdiff(f, bg)

    # the boundary points of a football field (295, 95), (640, 115) ...
    x = (295, 640, 640, 0)
    y = (95, 115, 380, 250)

    obj_mask = masking(diff, bg, x, y)
    fg = mask_background(diff, obj_mask)

    cv2.imshow('absdiff', fg)

    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break


cap.release()
cv2.destroyAllWindows()