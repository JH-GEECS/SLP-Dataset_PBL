import argparse
import glob
import json
import os
import re

import h5py
import numpy as np
import sh
from tqdm import tqdm


def load_json(fpath):
    with open(fpath, "r") as f:
       _file = json.load(f)
    return _file

def select_points(points, keep):
    new_points = []
    for k in keep:
        new_points.append(points[3 * k + 0])
        new_points.append(points[3 * k + 1])
        new_points.append(points[3 * k + 2])
    return new_points

def load_data(_dir):
    pose_idx = [0, 1, 2, 3, 4, 5, 6, 7]
    hand_idx = range(21)
    fpath_list = sorted(glob.glob(os.path.join(_dir, "*.json")), key=lambda x: int(re.findall(r'\d+', x)[-1]))
    
    frames = []
    for fpath in fpath_list:
        data = load_json(fpath)
        if len(data["people"]) == 0:
            continue
        
        joint_data = data["people"][0]
        pose = joint_data["pose_keypoints_2d"]
        left_hand = joint_data["hand_left_keypoints_2d"]
        right_hand = joint_data["hand_right_keypoints_2d"]

        pose = select_points(pose, pose_idx)
        left_hand = select_points(left_hand, hand_idx)
        right_hand = select_points(right_hand, hand_idx)
        
        # (x,y,c) 24 + 63 + 63 = 150
        points = pose + left_hand + right_hand
        frames.append(points)

    return np.array(frames) # T x 150

def main(args):
    print("1. json2h5.py")
    if args.delete:
        sh.rm("-r", "-f", args.data_out)
    
    hf = h5py.File(args.data_out, "w")
    for mode in args.modes:
        dir_path = os.path.join(args.data_in, mode)
        dir_list = glob.glob(os.path.join(dir_path, "*"))
        
        for _dir in tqdm(dir_list):
            key = f"{mode}/{os.path.split(_dir)[-1]}"
            data = load_data(_dir)
            hf.create_dataset(key, data=data, dtype="float32")
    
    hf.close()

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_in", type=str, default="./data")
    parser.add_argument("--modes", nargs="+", required=True)
    parser.add_argument("--data_out", type=str, default="./data/keypoints-01-raw_wb.h5")
    parser.add_argument("--delete", action="store_true")
    return parser.parse_args()


if __name__=="__main__":
    args = get_args()
    main(args)
