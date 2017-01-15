import cv2
import time
import os
import argparse

output_path = r'output'
if not os.path.exists(output_path):
    os.makedirs(output_path)

bg_filepath = 'output/side-view.jpg'

def extract_background(video):
    """Receives a video filename(with extension) and returns the extracted background"""
    vid_cap = cv2.VideoCapture(video)

    if vid_cap.isOpened():
        frame_count = int(vid_cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print("Extracting background...")
        start_time = time.time()
        _, img = vid_cap.read()
        avg_img = img

        for frame in range(1, frame_count):
            _, img = vid_cap.read()
            frame_fl = float(frame)
            avg_img = (frame_fl * avg_img + img) / (frame_fl + 1)

        end_time = time.time()
        print(end_time - start_time)

        print("Saving background...")
        vid_cap.release()
        cv2.imwrite(bg_filepath, avg_img)
    else:
        raise IOError("Could not open video")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--video", required=True,
                    help="path to the video file")
    args = vars(ap.parse_args())
    path = args["video"]

    extract_background(path)
