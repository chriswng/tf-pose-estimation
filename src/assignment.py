import argparse
import logging
import time

import cv2
import pandas as pd
import numpy as np
import pandas as pd

import common
from estimator import TfPoseEstimator
from networks import get_graph_path, model_wh

import platform
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

logger = logging.getLogger('TfPoseEstimator-WebCam') 
logger.setLevel(logging.CRITICAL)
ch = logging.StreamHandler()
ch.setLevel(logging.CRITICAL)
formatter = logging.Formatter(
    '[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

fps_time = 0

#
POSE_COCO_BODY_PARTS = {
    0: "Nose",
    1: "Neck",
    2: "RShoulder",
    3: "RElbow",
    4: "RWrist",
    5: "LShoulder",
    6: "LElbow",
    7: "LWrist",
    8: "RHip",
    9: "RKnee",
    10: "RAnkle",
    11: "LHip",
    12: "LKnee",
    13: "LAnkle",
    14: "REye",
    15: "LEye",
    16: "REar",
    17: "LEar",
    18: "Background",
}

# call this when a taxi is being hailed!
def hail_taxi(img):
    print("Someone is hailing a taxi!")
    cv2.putText(img, "TAXI!",
                    (10, 500), cv2.FONT_HERSHEY_SIMPLEX, 18, (94, 218, 255), 2)
    cv2.putText(img, platform.uname().node,
                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

def get_body_part_coord(bodypart, coord):
    """
    This function returns the x or y coord of a specified body type.
    Enter bodypart and coord as strings.
    e.g. bodypart = "RAnkle", coord = "x"
    """
    df_row = df[df["Body Part"] == bodypart]
    
    if not df_row.empty:
        return float(df_row[coord])

if __name__ == '__main__':
    # arguements to your program
    parser = argparse.ArgumentParser(
        description='tf-pose-estimation realtime webcam')
    parser.add_argument('--camera', type=int, default=0)
    parser.add_argument('--zoom', type=float, default=1.0)
    parser.add_argument(
        '--resolution',
        type=str,
        default='432x368',
        help='network input resolution. default=432x368')
    parser.add_argument(
        '--model',
        type=str,
        default='mobilenet_thin',
        help='cmu / mobilenet_thin')
    parser.add_argument(
        '--show-process',
        type=bool,
        default=False,
        help='for debug purpose, if enabled, speed for inference is dropped.')
    args = parser.parse_args()

    w, h = model_wh(args.resolution)
    e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h))
    camera = cv2.VideoCapture(args.camera)
    ret_val, image = camera.read()
    
    print("**** CTRL+C to exit ****")
    while True:
        # get image form the camera
        ret_val, image = camera.read()
        # boilerplate
        canvas = np.zeros_like(image)
        img_scaled = cv2.resize(
            image,
            None,
            fx=args.zoom,
            fy=args.zoom,
            interpolation=cv2.INTER_LINEAR)
        dx = (canvas.shape[1] - img_scaled.shape[1]) // 2
        dy = (canvas.shape[0] - img_scaled.shape[0]) // 2
        canvas[dy:dy + img_scaled.shape[0], dx:
               dx + img_scaled.shape[1]] = img_scaled
        image = canvas
        # feed image into the neural network
        humans = e.inference(image)  # list of humans
        for id, human in enumerate(humans):

            # TODO ensure it only does this when someone is hailing a taxi.
            # That is, an arm is above their head.

            df = pd.DataFrame([(POSE_COCO_BODY_PARTS[k], v.x, v.y) for k,v in human.body_parts.items()])
            df.columns = ["Body Part", "x", "y"]
            
            try:
                if get_body_part_coord("RWrist", "y") < get_body_part_coord("Nose", "y") or \
                   get_body_part_coord("LWrist", "y") < get_body_part_coord("Nose", "y"):
                    hail_taxi(image)
            except:
                print("")

            
        # drawing lines on an image
        image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)

        # FPS counter
        cv2.putText(image, "FPS: {:.2f}".format(1.0 / (time.time() - fps_time)),
                    (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 0, 0), 2)
        cv2.imshow('tf-pose-estimation result', image)
        fps_time = time.time()
        if cv2.waitKey(1) == 27:
            break

    cv2.destroyAllWindows()

# test commit: elijah
# co-author test commit: elijah
# test commit: Christopher
# co-author testing: chris