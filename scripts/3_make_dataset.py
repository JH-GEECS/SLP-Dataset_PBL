import argparse
import glob
import gzip
import os
import pickle
import re

import h5py
import numpy as np
from tqdm import tqdm

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
        pose_dim(=150) x seq
    '''
    pmin = min(parray.min(axis=0))
    pmax = max(parray.max(axis=0))
    norm = (parray - pmin) / (pmax - pmin)
    return norm


def centering(parray):
    '''
    centering using neck coordinate
    neck 
    '''
    dim, seq = parray.shape
    neck_array = parray[3:6, :] # (x, y, z) x seq
    
    diff = neck_array - 0.5
    diff = np.expand_dims(diff, axis=0)
    diff = np.repeat(diff, 50, axis=0).reshape(-1, seq) # 150 x seq
    
    parray -= diff
    return parray    


def main(args):
    annotation = load_annotation(args.annotation)
    pose = load_h5py(args.joint)

    total = len(annotation)
    passed = 0
    for a_data in tqdm(annotation):
        name = a_data["name"]
        mode, name = os.path.split(name)
        mode = get_mode(mode)
        try:
            parray = np.array(pose[mode][name]).transpose(1, 0) # pose_dim x seq
            parray = minmax_norm(parray)
            parray = centering(parray)
            
            parray = parray.transpose(1, 0) # seq x pose_dim
        except KeyError:
            parray = None
            passed += 1

        a_data["sign"] = parray

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
