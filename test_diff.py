import cv2
import numpy as np
import imutils
import time
from skimage import measure
from skimage import filters
import matplotlib.pyplot as plt

cap = cv2.VideoCapture("data/0001_L.mp4")
fwidth = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
fheight = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

bg = cv2.imread("data/side-view.jpg", cv2.IMREAD_COLOR)

def inPolygon(x, y, xp, yp):
   c = 0
   for i in range(len(xp)):
       if (((yp[i] <= y and y < yp[i-1]) or (yp[i-1] <= y and y < yp[i])) and
                   (x > (xp[i-1] - xp[i]) * (y - yp[i]) / (yp[i-1] - yp[i]) + xp[i])):
           c = 1 - c
   if c == 1:
       return True
   return False

def tresh(pixel):
    if pixel[0] < 40 and pixel[1] < 40 and pixel[2] < 40:
        return False
    return True

def masking(img, bg, polyX, polyY, offset_v=0, offset_h=0):
    n_rows = img.shape[0]
    n_cols = img.shape[1]
    bg_hsv = cv2.cvtColor(bg, cv2.COLOR_BGR2HSV)
    m = np.zeros([n_rows, n_cols], dtype=np.int)
    # obj_x = []
    # obj_y = []

    for i in range(offset_v - 1, n_rows - 1):
        for j in range(offset_h - 1, n_cols - 1):
            bg_hue = int(bg_hsv[i, j, 0])

            if not on_field_background(bg_hue) or not inPolygon(j, i, polyX, polyY) or not tresh(img[i, j]):
                continue

            m[i, j] = 1
            # obj_x.append(i)
            # obj_y.append(j)

    return m #, np.array(obj_x, dtype=np.int), np.array(obj_y, dtype=np.int)

def mask_background(img, mask, mask_color=(0, 0, 0)):
    result = np.array(img, copy=True)
    result[mask == 0] = mask_color
    result[mask == 1] = (255, 255, 255)
    return result

def on_field_background(hue):
    if hue in range(35, 45):
        return True
    return False

def feature_vector(img, obj_x, obj_y, x_max, y_max, increment=1):
    n_features = 3
    result = np.zeros([1, n_features])
    result_norm = np.zeros([1, n_features])
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    n_points = obj_x.size
    for i in range(0, n_points, increment):
        x = obj_x[i]
        y = obj_y[i]
        hue = hsv[x, y, 0]
        # sat = hsv[x, y, 1]

        fv = np.array([x, y, hue])
        result = np.vstack((result, fv))

        # Normalize the values

        x = x / (x_max - 1)
        y = y / (y_max - 1)
        hue = hue / 179.
        # sat = sat / 255.

        fv = np.array([x, y, hue])
        result_norm = np.vstack((result_norm, fv))

    result_norm = np.delete(result_norm, 0, 0)
    result = np.delete(result, 0, 0)
    return result_norm, result

# for i in range(500):
#     __, f = cap.read()

while(1):
    ret, f = cap.read()
    if ret is False:
        break

    diff = cv2.absdiff(f, bg)

    x = (295, 640, 640, 0)
    y = (95, 115, 380, 250)

    obj_mask = masking(diff, bg, x, y)
    fg = mask_background(diff, obj_mask)

    # n = 2
    # l = 10
    #
    # im = filters.gaussian(fg, sigma=l / (4. * n), multichannel=False)
    # blobs = im > 0.7 * im.mean()
    # labels = measure.label(fg, neighbors=8, background=0)
    #
    # for label in np.unique(labels):
    #     if label == 0:
    #         continue
    #
    #     labelMask = np.zeros(fg.shape, dtype="uint8")
    #     labelMask[labels == label] = 255
    #     fg = cv2.add(fg, labelMask)


    cv2.imshow('absdiff', fg)

    # start_time = time.time()
    # fvs, fvs_raw = feature_vector(fg, obj_x, obj_y, x_max=fheight, y_max=fwidth, increment=4)
    # print("Feature vectors %s (%ds)" % (str(fvs.shape), (time.time() - start_time)))
    # np.savetxt("foreground.txt", fvs, '%5.8f')
    # np.savetxt("foreground_unnormalized.txt", fvs_raw, '%5.0f')
    # exit()

    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break


cap.release()
cv2.destroyAllWindows()