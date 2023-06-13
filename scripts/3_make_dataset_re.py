import argparse
import glob
import gzip
import json
import os
import pickle
import re

import h5py
import numpy as np
from tqdm import tqdm


def load_json(fpath):
    with open(fpath, "r") as f:
       _file = json.load(f)
    return _file

def save_annotation(_file, fpath):
    with gzip.open(fpath, "wb") as f:
        pickle.dump(_file, f)

def load_h5py(fpath):
    _file = h5py.File(fpath, "r")
    return _file

def load_annotation(fpath):
    with gzip.open(fpath, "rb") as f:
        _file = pickle.load(f)
    return _file

def load_pickle(fpath):
    print(f"load {fpath}")
    with open(fpath, "rb") as f:
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

def minmax_norm(parray):
    '''
    input array: 
        [T, dim]
    '''
    pmin = min(parray.min(axis=-1))
    pmax = max(parray.max(axis=-1))
    norm = (parray - pmin) / (pmax - pmin)
    return norm

def centering(parray):
    '''
    centering using neck coordinate
    input:
        [T, dim]
    '''
    t, dim = parray.shape
    neck_idx = 1

    new_neck_xs = parray[:, neck_idx] - 0.5
    new_neck_ys = parray[:, dim//2 + neck_idx] - 0.5

    xs = parray[:, :dim//2] - np.expand_dims(new_neck_xs, axis=-1)
    ys = parray[:, dim//2:] - np.expand_dims(new_neck_ys, axis=-1)

    return np.concatenate((xs, ys), axis=-1)

def relocate(landmark_coordinates, pose_coordinates):
    landmark_xs = landmark_coordinates[:, landmark_coordinates.shape[-1]//2:]
    landmark_ys = landmark_coordinates[:, :landmark_coordinates.shape[-1]//2]

    nose_idx = 30
    nose_xs = landmark_xs[:, nose_idx] - pose_coordinates[:, 0]
    nose_ys = landmark_ys[:, nose_idx] - pose_coordinates[:, 51]

    landmark_xs -= np.expand_dims(nose_xs, axis=-1)
    landmark_ys -= np.expand_dims(nose_ys, axis=-1)

    return np.concatenate((landmark_xs, landmark_ys), axis=-1)

def main(args):
    # 여기서 독일어의 경우에는 그냥 취득하면 될 것 같다.
    annotation = load_annotation(args.annotation)
    # 이 부분 처리함에 있어서, frame 수를 넣어 주는 작업 필요한데, normalize하고 centring한 다음에 visaulize하면 될 듯
    pose = load_h5py(args.joint)

    total = len(annotation)
    passed = 0
    for a_data in tqdm(annotation):
        fpath = a_data["name"]
        
        mode, name = os.path.split(fpath)
        mode = get_mode(mode)
        
        # load json file to get landmark information
        flist = glob.glob(os.path.join("./data", mode, name, "*.json"))
        
        landmarks = []
        for _json in flist:
            json_file = load_json(_json)
            l = json_file["people"][0]["face_keypoints_2d"]
            landmarks.append(l)
        
        # get landmark array
        landmark_array = np.array(landmarks) # [T, 210]
        
        # get pose array
        parray = np.array(pose[mode][name]) # [T, 150]

        # check length
        assert len(landmark_array) == len(parray)
        
        # get pose coordinates
        parray_xs = parray[:, ::3] # [T, 50]
        parray_ys = parray[:, 1::3]

        # [T, 100]
        pose_coordinates = np.concatenate((parray_xs, parray_ys), axis=-1)

        # normalizing 0 ~ 1
        pose_coordinates = minmax_norm(pose_coordinates)

        # centering
        pose_coordinates = centering(pose_coordinates)

        # take off uneccessary values
        landmark_xs = landmark_array[:, ::3] # [T, 70]
        landmark_ys = landmark_array[:, 1::3]

        landmark_coordinates = np.concatenate((landmark_xs, landmark_ys), axis=-1)
        
        # normalizing 0 ~ 1
        # landmark_coordinates = minmax_norm(landmark_coordinates)

        # relocate face
        # landmark_coordinates = relocate(landmark_coordinates, pose_coordinates)
        
        # a_data["sign"] = pose_coordinates
        a_data["sign"] = np.concatenate((pose_coordinates, landmark_coordinates), axis=-1)

    print(f"processing completed [{total-passed}/{total}].")

    if args.save is not None:
        save_annotation(annotation, args.save)
        print(f"saved at {args.save}")
        

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation", type=str, default="./data/phoenix14t.pami0.dev")
    parser.add_argument("--joint", type=str, default="./data/lifted_dev.h5")
    parser.add_argument("--save", type=str, default="./data/phoenix14t.pose.dev")
    
    parser.add_argument("--delete", action="store_true")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


if __name__=="__main__":
    args = get_args()
    main(args)
