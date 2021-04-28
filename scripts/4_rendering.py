import argparse
import glob
import gzip
import json
import os
import pickle
import re
import sys

import cv2
import h5py
import matplotlib.pyplot as plt
import numpy as np
import sh

POINT_PAIR = [
    # body
    [0, 1], 
    [1, 2], 
    [2, 3],
    [3, 4],
    [1, 5],
    [5, 6],
    [6, 7], 
    # left hand
    # [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14]
    # right hand
]


def load_h5py(fpath):
    _file = h5py.File(fpath, "r")
    return _file


def load_annotation(fpath):
    with gzip.open(fpath, "rb") as f:
        _file = pickle.load(f)
    return _file


def get_mode(mode):
    if mode == "dev": 
        mode = "dev_json_res"
    elif mode == "train": 
        mode = "train_json_res"
    else: 
        mode = "test_json_res"
    return mode


def draw_skeleton(parray):
    '''
    parray: 
        pose_dim(=150)
        value range [0,1]
    '''
    H, W = 256, 256
    img = np.zeros((H, W), np.uint8) + 255 # white blank page
    
    # corresponding coordinate in a H x W
    # (x(H - 1) + 1, y(W - 1) + 1)
    px = parray[0::3] * H + 1
    py = parray[1::3] * W + 1

    for i in range(len(px)):
        cv2.circle(img, (int(px[i]), int(py[i])), 2, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
    
    for pair in POINT_PAIR:
        start = (int(px[pair[0]]), int(py[pair[0]]))
        end = (int(px[pair[1]]), int(py[pair[1]]))
        cv2.line(img, start, end, (0, 0, 255), 2)

    return img


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_in", type=str, default="./data/phoenix14t.pose.dev")
    parser.add_argument("--save", type=str, default="./videos")
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--verbose", type=bool, default=False)
    parser.add_argument("--num_videos", type=int, default=10)
    return parser.parse_args()


def main(args):
    annotation = load_annotation(args.data_in)
    
    for vid_num, _data in enumerate(annotation):
        pose_array = _data["sign"]
        mode, name = os.path.split(_data["name"])
        fdir = os.path.join(args.save, _data["name"])

        if not(os.path.exists(fdir)): 
            os.makedirs(fdir)
        
        # save pose images        
        for i, parray in enumerate(pose_array):
            fig = draw_skeleton(parray)          
            cv2.imwrite(os.path.join(fdir, f"{name}-{i}.jpg"), fig)
        
        # save pose video
        pose_images = glob.glob(os.path.join(fdir, "*.jpg"))
        pose_images = sorted(pose_images, key=lambda x: int(re.findall(r'\d+', x)[-1]))
                
        frame_array = []
        for p_img in pose_images:
            img = cv2.imread(p_img)
            h, w, c = img.shape
            size = (w, h)
            frame_array.append(img)
        out = cv2.VideoWriter(os.path.join(args.save, mode, f"{name}.mp4"), cv2.VideoWriter_fourcc(*'DIVX'), args.fps, size)
        for frame in frame_array:
            out.write(frame)
        out.release()
        
        if not(args.verbose):
            sh.rm("-r", "-f", fdir)

        if vid_num == args.num_videos:
            break
        
    
if __name__=="__main__":
    args = get_args()
    main(args)
