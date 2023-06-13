import argparse
import glob
import json
import os
import re

import h5py
import numpy as np
# import sh
from tqdm import tqdm
import ray


@ray.remote
class Counter(object):
    def __init__(self):
        self.n = 0

    def increment(self):
        self.n += 1

    def read(self):
        return self.n


def load_json(fpath):
    with open(fpath, "r") as f:
        _file = json.load(f)
    return _file

# 여기 실수 했음
def select_points(points, keep):
    new_points = []
    for k in keep:
        new_points.append(points[3*k+0])
        new_points.append(points[3*k+1])
        new_points.append(points[3*k+2])
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

    return np.array(frames)  # T x 150


@ray.remote(num_cpus=1)
def parallel_load_data(key, _dir, counter):
    pose_idx = [0, 1, 2, 3, 4, 5, 6, 7]
    hand_idx = range(21)
    fpath_list = sorted(glob.glob(os.path.join(_dir, "*.json")), key=lambda x: int(re.findall(r'\d+', x)[-1]))

    frames = []
    for fpath in fpath_list:
        data = load_json(fpath)
        if len(data["people"]) == 0:
            continue

        joint_data = data["people"]
        pose = joint_data["pose_keypoints_2d"]
        left_hand = joint_data["hand_left_keypoints_2d"]
        right_hand = joint_data["hand_right_keypoints_2d"]

        pose = select_points(pose, pose_idx)
        left_hand = select_points(left_hand, hand_idx)
        right_hand = select_points(right_hand, hand_idx)

        # (x,y,c) 24 + 63 + 63 = 150
        points = pose + left_hand + right_hand
        frames.append(points)

    counter.increment.remote()
    return key, np.array(frames)  # T x 150

def single_load_data(key, _dir):
    pose_idx = [0, 1, 2, 3, 4, 5, 6, 7]
    hand_idx = range(21)
    fpath_list = sorted(glob.glob(os.path.join(_dir, "*.json")), key=lambda x: int(re.findall(r'\d+', x)[-1]))

    frames = []
    for fpath in fpath_list:
        data = load_json(fpath)
        if len(data["people"]) == 0:
            continue

        joint_data = data["people"]
        pose = joint_data["pose_keypoints_2d"]
        left_hand = joint_data["hand_left_keypoints_2d"]
        right_hand = joint_data["hand_right_keypoints_2d"]

        pose = select_points(pose, pose_idx)
        left_hand = select_points(left_hand, hand_idx)
        right_hand = select_points(right_hand, hand_idx)

        # (x,y,c) 24 + 63 + 63 = 150
        points = pose + left_hand + right_hand
        frames.append(points)

    return key, np.array(frames)  # T x 150


def main(args):
    print("1. json2h5.py")
    # if args.delete:
    #    sh.rm("-r", "-f", args.data_out)

    # temp data store
    dataset_list = []

    # ray init and progress counter
    ray.init(num_cpus=40)

    # Create the glob pattern for subdirectories with '01' to '16'
    # 여깃 살짝 수정해서 1 ~ 16개 모두 수집할 수 있도록 code 변형
    pattern = os.path.join(args.data_in, '[0-1][0-9]', '*_F')
    # pattern = os.path.join(args.data_in, '01', '*_F')

    # 여깃 살짝 수정해서 1 ~ 16개 모두 수집할 수 있도록 code 변형
    dir_list = glob.glob(pattern)

    # 이 부분 parallelize 해야함
    # 참조 key = f"{mode}/{os.path.split(_dir)[-1]}"
    """
        for _dir in dir_list:
        key = f"{os.path.split(_dir)[-1]}"
        dataset_list.append(single_load_data(key, _dir))
    """

    counter = Counter.remote()
    distributed_processing = [
        parallel_load_data.remote(f"{os.path.split(_dir)[-1]}", _dir, counter) for _dir in dir_list
    ]

    with tqdm(total=len(dir_list), desc="Progress Indicator") as pbar:
        while pbar.n < pbar.total:
            n = ray.get(counter.read.remote())
            pbar.n = n
            pbar.refresh()

    result_list = ray.get(distributed_processing)
    dataset_list.extend(result_list)

    ray.shutdown()
    # h5 create and save
    hf = h5py.File(args.data_out, "w")
    for key, data in tqdm(dataset_list):
        hf.create_dataset(key, data=data, dtype="float32")
    hf.close()


def get_args():
    main_data_path = r'/data/2023_PBL/sign_data/temp_sign_data/수어 영상/1.Training'
    df_data_path = r'/data/2023_PBL/sign_data/temp_sign_data/수어 영상'
    # F에서 facing하는 것만을 떼어 내야 한다.
    # main_data_path = r'Z:\2023_PBL\sign_data\temp_sign_data\수어 영상\1.Training'
    # df_data_path = r'Z:\2023_PBL\sign_data\temp_sign_data\수어 영상'

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_in", type=str, default=main_data_path)
    # parser.add_argument("--modes", nargs="+", default=["train", "dev", "test"])
    parser.add_argument("--data_out", type=str, default=os.path.join(df_data_path, 'keypoints-01-raw_wb_korean.h5'))
    # parser.add_argument("--delete", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    """
    # json load test
        
    json_path = r'Z:\2023_PBL\sign_data\raw_data\PHOENIX-2014-T-release-v3\PHOENIX_processing\dev\01April_2010_Thursday_heute-6697\images0001.json'
    test_json = load_json(json_path)
    test = 1
    """

    args = get_args()
    main(args)
