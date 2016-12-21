import cv2
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Test tracking movement')
parser.add_argument('-v', '--video', help='Path to video file', required=True)
args = vars(parser.parse_args())

if args['video'] is False:
    exit()

cap = cv2.VideoCapture(args['video'])

while(1):

    # Take each frame
    ret, frame = cap.read()

    if ret is False:
        break

    # Convert BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # define range of blue color in HSV
    lower_blue = np.array([90,100,50])
    upper_blue = np.array([140, 255, 255])
    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
    # mask = blue_mask

    # define range of red color in HSV
    lower_red = np.array([8, 100, 50])
    upper_red = np.array([20, 255, 255])
    red_mask = cv2.inRange(hsv, lower_red, upper_red)  # I have the Green threshold image.
    # mask = red_mask

    # plus blue_mask if want
    mask = red_mask

    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(frame, frame, mask=mask)
    imgGray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)

    ret, thresh = cv2.threshold(imgGray, 127, 255, 0)
    im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(frame, contours, -1, (0, 255, 0), 3)

    cv2.imshow('frame', frame)

    k = cv2.waitKey(30) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()