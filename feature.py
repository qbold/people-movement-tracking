import cv2
import numpy as np

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