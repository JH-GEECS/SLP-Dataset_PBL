# 기본적으로 하나의 folder에만 적용되는 openpose를 recursive하게 적용하여 모든 folder에 대하여 json을 추출하기 위함이다.
# json 추출할 생각하니까 어지러워서

# From Python
# It requires OpenCV installed for Python
import os
import shutil
import subprocess
from tqdm import tqdm
import json
import pandas as pd
import re
import sys
import argparse
import cv2
import numpy as np
import ray

"""
내가 원하는 추출 구조

root
|-dev ---
|       |- image-folder-sentence_num_1
|       |- image-folder-sentence_num_2
|       |- ...
|
|-test ---
|       |- image-folder-sentence_num_1
|       |- image-folder-sentence_num_2
|       |- ...
|
|-train ---
|       |- image-folder-sentence_num_1
|       |- image-folder-sentence_num_2
|       |- ...
|

traversal을 거친 이후에는 똑같은 topology를 가진 형태로 json structure가 만들어 져야 한다.

"""


def json2dataframe(json_path, df_csv_path):
    with open(json_path, 'r') as f:
        raw_json_data = json.load(f)
    # 일단 최상위 dict에서 사진 경로만 뽑고 싶으니까 살짝 변경해준다.
    raw_json_data = raw_json_data[0]
    raw_json_data['name'] = ''

    # json_data를 dataframe으로 바꾸는 과정이 필요하다. 이후에는 csv로 저장한다.
    df_columns = {
        'type': str,
        'image_path': str,
        'image_name': str,
        'sentence_sequence': int,
        'frame_sequence': int
    }

    data_vector = []

    def extract_path(node, path=''):
        if 'type' in node and 'name' in node:
            # 여기 정규식에서 image만 걸러내자
            if node['type'] == 'file' and 'png' in node['name']:
                path_parsed = path.split('/')
                sentence_sequence = int(path_parsed[-1].split('-')[-1])
                frame_sequence = int(re.search(r'\d+', node['name']).group())
                data_vector.append([path_parsed[0],
                                    os.path.join(path, node['name']),
                                    node['name'],
                                    sentence_sequence,
                                    frame_sequence])
            elif node['type'] == 'directory' and 'contents' in node:
                if path == '':
                    current_path = node['name']
                else:
                    current_path = path + '/' + node['name']
                for child_node in node['contents']:
                    extract_path(child_node, current_path)

    extract_path(raw_json_data)

    df = pd.DataFrame(data_vector, columns=df_columns.keys()).astype(df_columns)
    df.to_csv(df_csv_path)
    return df


def openpose_python_applier(image_root_dir, dataframe, model_params, result_df_path, save_json=True):
    # dataframe을 받아서 openpose를 적용하고, 결과를 저장한다.

    # open pose python api 제작
    openpose_py_dir = model_params['openpose_py_dir']
    sys.path.append(openpose_py_dir)
    # noinspection PyUnresolvedReferences
    from openpose import pyopenpose as op

    parser = argparse.ArgumentParser()
    # parser.add_argument("--model_folder", default=model_params['model_dir'], help="Enable to disable the visual display.")
    parser.add_argument("--display", default=0, help="Enable to disable the visual display.")
    parser.add_argument("--render_pose", default=0, help="Enable to disable the visual display.")
    args = parser.parse_known_args()

    params = dict()

    for i in range(0, len(args[1])):
        curr_item = args[1][i]
        if i != len(args[1]) - 1:
            next_item = args[1][i + 1]
        else:
            next_item = "1"
        if "--" in curr_item and "--" in next_item:
            key = curr_item.replace('-', '')
            if key not in params:  params[key] = "1"
        elif "--" in curr_item and "--" not in next_item:
            key = curr_item.replace('-', '')
            if key not in params: params[key] = next_item

    params['model_folder'] = model_params['model_dir']
    params['face'] = model_params['face']
    params['hand'] = model_params['hand']
    opWrapper = op.WrapperPython()
    opWrapper.configure(params)
    opWrapper.start()

    data_pose_vector = []
    for index, row in tqdm(dataframe.iterrows(), total=dataframe.shape[0]):
        datum = op.Datum()
        # 여기 image 경로 수정해주기
        imageToProcess = cv2.imread(os.path.join(image_root_dir, row['image_path']))
        datum.cvInputData = imageToProcess
        opWrapper.emplaceAndPop(op.VectorDatum([datum]))

        pose_keypoint = datum.poseKeypoints[0]
        face_keypoint = datum.faceKeypoints[0]

        hand_left_keypoint = datum.handKeypoints[0][0]
        hand_right_keypoint = datum.handKeypoints[1][0]

        each_data_vector = [
            row['type'],
            row['image_path'],
            row['image_name'],
            row['sentence_sequence'],
            row['frame_sequence'],
            pose_keypoint,
            hand_left_keypoint,
            hand_right_keypoint,
            face_keypoint
        ]
        if save_json:
            each_data_dict = {
                'type': row['type'],
                'image_path': row['image_path'],
                'image_name': row['image_name'],
                'sentence_sequence': row['sentence_sequence'],
                'frame_sequence': row['frame_sequence'],
                'pose_keypoint': pose_keypoint.tolist(),  # Converting numpy array to list
                'hand_left_keypoint': hand_left_keypoint.tolist(),  # Converting numpy array to list
                'hand_right_keypoint': hand_right_keypoint.tolist(),  # Converting numpy array to list
                'face_keypoint': face_keypoint.tolist()
            }
            each_dir = os.path.join(os.path.dirname(result_df_path), os.path.dirname(row['image_path']))
            os.makedirs(each_dir, exist_ok=True)
            file_name = row['image_name'].split('.')[0] + '.json'
            with open(os.path.join(each_dir, file_name), 'w') as f:
                json.dump(each_data_dict, f)
        data_pose_vector.append(each_data_vector)

    processed_df = pd.DataFrame(data_pose_vector,
                                columns=dataframe.columns.tolist() + ['pose', 'hand_left', 'hand_right', 'face'])
    processed_df.to_csv(result_df_path)
    return processed_df


@ray.remote(num_gpus=1, num_cpus=4)
def parallelized_python_applier(image_root_dir, dataframe, model_params, result_df_path, save_json, counter):
    # dataframe을 받아서 openpose를 적용하고, 결과를 저장한다.

    # open pose python api 제작
    openpose_py_dir = model_params['openpose_py_dir']
    sys.path.append(openpose_py_dir)
    # noinspection PyUnresolvedReferences
    from openpose import pyopenpose as op

    params = dict()

    params['display'] = 0
    params['render_pose'] = 0
    params['model_folder'] = model_params['model_dir']
    params['face'] = model_params['face']
    params['hand'] = model_params['hand']
    opWrapper = op.WrapperPython()
    opWrapper.configure(params)
    opWrapper.start()

    data_pose_vector = []
    for index, row in tqdm(dataframe.iterrows(), total=dataframe.shape[0], disable=True):
        datum = op.Datum()
        # 여기 image 경로 수정해주기
        imageToProcess = cv2.imread(os.path.join(image_root_dir, row['image_path']))
        datum.cvInputData = imageToProcess
        opWrapper.emplaceAndPop(op.VectorDatum([datum]))

        pose_keypoint = datum.poseKeypoints[0]
        face_keypoint = datum.faceKeypoints[0]

        hand_left_keypoint = datum.handKeypoints[0][0]
        hand_right_keypoint = datum.handKeypoints[1][0]

        each_data_vector = [
            row['type'],
            row['image_path'],
            row['image_name'],
            row['sentence_sequence'],
            row['frame_sequence'],
            pose_keypoint,
            hand_left_keypoint,
            hand_right_keypoint,
            face_keypoint
        ]
        if save_json:
            each_data_dict = {
                'type': row['type'],
                'image_path': row['image_path'],
                'image_name': row['image_name'],
                'sentence_sequence': row['sentence_sequence'],
                'frame_sequence': row['frame_sequence'],
                'pose_keypoint': pose_keypoint.tolist(),  # Converting numpy array to list
                'hand_left_keypoint': hand_left_keypoint.tolist(),  # Converting numpy array to list
                'hand_right_keypoint': hand_right_keypoint.tolist(),  # Converting numpy array to list
                'face_keypoint': face_keypoint.tolist()
            }
            each_dir = os.path.join(os.path.dirname(result_df_path), os.path.dirname(row['image_path']))
            os.makedirs(each_dir, exist_ok=True)
            file_name = row['image_name'].split('.')[0] + '.json'
            counter.increment.remote()
            with open(os.path.join(each_dir, file_name), 'w') as f:
                json.dump(each_data_dict, f)
        data_pose_vector.append(each_data_vector)

    processed_df = pd.DataFrame(data_pose_vector,
                                columns=dataframe.columns.tolist() + ['pose', 'hand_left', 'hand_right', 'face'])
    # processed_df.to_csv(result_df_path)
    # 여기서 sharding된 dataframe을 꼭 나중에 뭉쳐줘야 한다.
    return processed_df

@ray.remote
class Counter(object):
    def __init__(self):
        self.n = 0

    def increment(self):
        self.n += 1

    def read(self):
        return self.n

def paralleization_wrapper(image_root_dir, dataframe, model_params, result_df_path, fuction_params):
    """
    다음에 짤 떄는 producer consumer task를 asynchronous하게 짜서, disk write에 의한 stall현상을 최대한 억제한다.


    :param image_root_dir:
    :param dataframe:
    :param model_params:
    :param result_df_path:
    :param fuction_params:
    :return:
    """

    # 사용할 GPU의 지정
    ray.init(num_gpus=fuction_params['num_gpus'])

    counter = Counter.remote()
    
    # dataframe을 sharding한다.
    sharded_dataframe = np.array_split(dataframe, fuction_params['num_gpus'])
    num_images = dataframe.shape[0]
    
    distributed_processes = [parallelized_python_applier.remote(image_root_dir,
                                                                each_dataframe,
                                                                model_params,
                                                                result_df_path,
                                                                fuction_params['save_json'],
                                                                counter)
                             for each_dataframe in sharded_dataframe]
    
    with tqdm(total=num_images, desc="Progress Indicator") as pbar:
        while pbar.n < pbar.total:
            n = ray.get(counter.read.remote())
            pbar.n = n
            pbar.refresh()
    
    # 여기서 sharding된 dataframe을 꼭 나중에 뭉쳐줘야 한다.
    result_df = ray.get(distributed_processes)
    result_df = pd.concat(result_df)
    result_df.to_csv(result_df_path)

    ray.shutdown()

    return result_df


def get_folder_structure(root_dir):
    #  file이 더럽게 많아서 걱정되기는 하는데 이렇게 진행한다.
    lowest_level_dirs = []
    for root, dirs, files in os.walk(root_dir):
        if not dirs:  # Check if the current directory has no subdirectories
            lowest_level_dirs.append(root)
            print(root)
    return lowest_level_dirs


def create_folder_structure(root_dir, structure):
    # 오로지 directory structure을 복제하기 위한 함수이다.
    # 물론 거대한 경우에 shell로 짜는게 맞을 것이다...
    # 하지만, openpose 실행만 해도 시간이 꽤 걸린다. overnight으로 할 것으로 생각한다.
    for folder_path, folder_type in structure.items():
        folder_path = os.path.join(root_dir, folder_path)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)


def openpose_json_extractor(image_folder_root_dir, dest_folder_root_dir, model_params):
    # 여기에서 parsing해서 file만을 가진 것이 그거다.
    structure = get_folder_structure(image_folder_root_dir)

    # terminal folder에 대하여 openpose를 실행한다.
    for terminal_folder in tqdm(structure):
        # cmd는 universial하게 사용할 수 있게 vector 형태로 빼자.
        openpose_cmd = [
            model_params['openpose_path'],
            "--image_dir", os.path.join(image_folder_root_dir, terminal_folder),
            "--write_json", os.path.join(dest_folder_root_dir, terminal_folder),
            "--model_folder", model_params['model_dir'],
            "--display", "0",
            "--render_pose", "0"]
        openpose_cmd.append("--hand") if model_params['hand'] else None
        openpose_cmd.append("--face") if model_params['face'] else None
        subprocess.run(openpose_cmd, check=True)

    # 음 구지 create_folder_structure를 쓸 필요 없이 추출하고, 새로운 root에 json 생성하면 되네,
    # phonenix dataset specific하게만 사용하는 중이다.
    # openpose 적용용으로 사용하는 code이므로 일단은 크게 신경쓰지 않는다.
    print("img to json job complete")


if __name__ == '__main__':
    """
        # recursive하게 path들만 획득하면 된다.
    image_folder_root_dir = r'/home/jovyan/data/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/features/fullFrame-210x260px/'
    dest_folder_root_dir = r'/home/jovyan/data2/PHOENIX_JSON'
    model_params = {
        'openpose_py_dir': r'/home/jovyan/data/openpose/build/python/',
        'model_dir': r'/home/jovyan/data/openpose/models/',
        'hand': True,
        'face': True}
    # openpose_json_extractor(image_folder_root_dir, dest_folder_root_dir, model_params)

    json_path = r'/home/jovyan/data2/PHOENIX_JSON/phoenix_tree.json'
    df_csv_path = r'/home/jovyan/data2/PHOENIX_JSON/phoenix_tree.csv'
    result_df_path = r'/home/jovyan/data2/PHOENIX_JSON/phoenix_tree_processed.csv'
    
    """

    # recursive하게 path들만 획득하면 된다.
    image_folder_root_dir = r'/data/2023_PBL/sign_data/raw_data/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/features/fullFrame-210x260px/'
    dest_folder_root_dir = r'/data/2023_PBL/sign_data/raw_data/PHOENIX-2014-T-release-v3/PHOENIX_processing_2'
    model_params = {
        'openpose_py_dir': r'/data/2023_PBL/sign_data/raw_data/PHOENIX-2014-T-release-v3/openpose/build/python',
        'model_dir': r'/data/2023_PBL/sign_data/raw_data/PHOENIX-2014-T-release-v3/openpose/models',
        'hand': True,
        'face': True}
    fuction_params = {
        'save_json': True,
        'num_gpus': 7
    }
    # openpose_json_extractor(image_folder_root_dir, dest_folder_root_dir, model_params)

    json_path = r'/data/2023_PBL/sign_data/raw_data/PHOENIX-2014-T-release-v3/phoenix_tree.json'
    df_csv_path = r'/data/2023_PBL/sign_data/raw_data/PHOENIX-2014-T-release-v3/phoenix_tree.csv'
    result_df_path = r'/data/2023_PBL/sign_data/raw_data/PHOENIX-2014-T-release-v3/PHOENIX_processing/phoenix_tree_processed.csv'
    # json2dataframe(json_path=json_path, df_csv_path=df_csv_path)
    paralleization_wrapper(image_root_dir=image_folder_root_dir,
                           dataframe=pd.read_csv(df_csv_path),
                           model_params=model_params,
                           result_df_path=result_df_path,
                           fuction_params=fuction_params)
    test = 1
