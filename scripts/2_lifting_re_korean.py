import argparse

import h5py
import numpy as np
# import sh
from tqdm import tqdm

import pose2D
import pose2Dto3D
import skeletalModel
import os
import ray
import pose3D_torch
import torch
import pandas as pd

def convList2Array(lst):
    T, dim = lst[0].shape
    a = []
    for t in range(T):
        a_t = []
        for i in range(dim):
            for j in range(len(lst)):
                a_t.append(lst[j][t, i])
        a.append(a_t)
    return np.asarray(a)

@ray.remote(num_cpus=1)
class Counter(object):
    def __init__(self):
        self.n = 0

    def increment(self):
        self.n += 1

    def read(self):
        return self.n
    
@ray.remote(num_cpus=1)
def use_filter(x, randomNubersGenerator, key, counter):
    structure = skeletalModel.getSkeletalModelStructure()

    inputSequence_2D = x

    # Decomposition of the single matrix into three matrices: x, y, w (=likelihood)
    X = inputSequence_2D
    Xx = X[0:X.shape[0], 0:(X.shape[1]):3]
    Xy = X[0:X.shape[0], 1:(X.shape[1]):3]
    Xw = X[0:X.shape[0], 2:(X.shape[1]):3]

    # Normalization of the picture (x and y axis has the same scale)
    Xx, Xy = pose2D.normalization(Xx, Xy)

    # Delete all skeletal models which have a lot of missing parts.
    dtype = "float32"
    Xx, Xy, Xw = pose2D.prune(Xx, Xy, Xw, (0, 1, 2, 3, 4, 5, 6, 7), 0.3, dtype)

    # Preliminary filtering: weighted linear interpolation of missing points.
    Xx, Xy, Xw = pose2D.interpolation(Xx, Xy, Xw, 0.99, dtype)

    # Initial 3D pose estimation
    lines0, rootsx0, rootsy0, rootsz0, anglesx0, anglesy0, anglesz0, Yx0, Yy0, Yz0 = pose2Dto3D.initialization(
        Xx,
        Xy,
        Xw,
        structure,
        0.001,  # weight for adding noise
        randomNubersGenerator,
        dtype,
        percentil=0.5,
    )
    print("initialization done")
    
    # Backpropagation-based instance
    dtype = torch.float32
    opt_model = pose3D_torch.BackpropagationBasedFiltering(
        lines0,
        rootsx0,
        rootsy0,
        rootsz0,
        anglesx0,
        anglesy0,
        anglesz0,
        structure,
        dtype,
        learning_rate=0.1,
        n_cycles=10,
    )

    Xx_tensor = torch.tensor(Xx, dtype=dtype)
    Xy_tensor = torch.tensor(Xy, dtype=dtype)
    Xw_tensor = torch.tensor(Xw, dtype=dtype)

    Yx, Yy, Yz = opt_model.train_model(Xx_tensor, Xy_tensor, Xw_tensor)
    print("backpropagation done")
    counter.increment.remote()
    return key, convList2Array([Yx.detach().cpu().numpy(),
                                Yy.detach().cpu().numpy(),
                                Yz.detach().cpu().numpy()])


def main(args):
    print("2. lifting.py")
    # if args.delete:
    #     sh.rm("-r", "-f", args.data_out)

    randomNubersGenerator = np.random.RandomState(1234)

    # temp data store
    dataset_list = []

    # temp ray subprocess list
    distributed_processing = []

    # ray init and progress counter
    context = ray.init(include_dashboard=True)

    # 여기서 병목, h5py 직렬화 시켜야함 parallelize해야 함
    data_in = h5py.File(args.data_in, "r")

    counter = Counter.remote()

    mode = '/'

    temp_keys = list(data_in.keys())
    temp_data = list(map(lambda x: np.asarray(data_in.get(x)), temp_keys))
    each_h5_df = pd.DataFrame({
        'name': temp_keys,
        'data': temp_data
    })
    each_h5_df = each_h5_df.set_index('name')

    for index, row in each_h5_df.iterrows():
        # print(f"\n[info] processing {key}\n")
        key = index
        x = row['data']
        # x = np.array(x)
        result_id = use_filter.remote(x, randomNubersGenerator, key, counter)
        distributed_processing.append(result_id)

    with tqdm(total=len(data_in.get(mode)), desc="Progress Indicator") as pbar:
        while pbar.n < pbar.total:
            n = ray.get(counter.read.remote())
            pbar.n = n
            pbar.refresh()

    distributed_result = ray.get(distributed_processing)
    dataset_list.extend(distributed_result)

    ray.shutdown()

    data_out = h5py.File(args.data_out, "w")
    for key, y in dataset_list:
        data_out.create_dataset(key, data=y, dtype=y.dtype)

    data_in.close()
    data_out.close()


def get_args():
    parser = argparse.ArgumentParser()
    # main_data_path = r'/data/2023_PBL/sign_data/raw_data/PHOENIX-2014-T-release-v3/PHOENIX_processing'
    main_data_path = r'/data/2023_PBL/sign_data/temp_sign_data/수어 영상'
    # main_data_path = r'C:\Users\Intern\Downloads\result'
    parser.add_argument("--data_in", type=str, default=os.path.join(main_data_path, "keypoints-01-raw_wb_korean.h5"))
    parser.add_argument("--data_out", type=str, default=os.path.join(main_data_path, "keypoints-01-raw_wb_3d_korean.h5"))

    # parser.add_argument("--modes", nargs="+", default=["train", "dev", "test"])
    # parser.add_argument("--delete", action="store_true")
    # parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    main(args)
