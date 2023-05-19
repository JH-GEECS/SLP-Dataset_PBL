# 기본적으로 하나의 folder에만 적용되는 openpose를 recursive하게 적용하여 모든 folder에 대하여 json을 추출하기 위함이다.
# json 추출할 생각하니까 어지러워서

# From Python
# It requires OpenCV installed for Python
import os
import shutil
import subprocess
from tqdm import tqdm

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


def get_folder_structure(root_dir):
    #  file이 더럽게 많아서 걱정되기는 하는데 이렇게 진행한다.
    structure = {}
    for dirpath, dirnames, filenames in os.walk(root_dir):
        folder_path = os.path.relpath(dirpath, root_dir)
        parent_folder = os.path.dirname(folder_path)
        if parent_folder not in structure:
            structure[parent_folder] = set()
        if filenames:
            structure[parent_folder].add('file')
        if dirnames:
            structure[parent_folder].add('directory')
    return structure


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

    # terminal folder만을 추출한다.
    structure_terminal = [k for k, v in structure.items() if 'file' in v and 'directory' not in v]

    # terminal folder에 대하여 openpose를 실행한다.
    for terminal_folder in tqdm(structure_terminal):
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
    # recursive하게 path들만 획득하면 된다.
    image_folder_root_dir = r'/home/jovyan/data/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/features/fullFrame-210x260px'
    dest_folder_root_dir = None
    model_params = {
        'openpose_path': r'/home/jovyan/data/openpose/build/examples/openpose/openpose.bin',
        'model_dir': r'/home/jovyan/data/openpose/models/',
        'hand': True,
        'face': True}
    openpose_json_extractor(image_folder_root_dir, dest_folder_root_dir, model_params)
