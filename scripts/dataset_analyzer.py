"""
해당 code의 목적은 취득한 원래 data를 model에 feed하기 위해서 어떤 식으로 변형해야하는 가를 보고자 함이다.

"""
import pandas as pd
import numpy as np
import os
import h5py
from tqdm import tqdm
import re

def original_loader(file_path):
    """
    1. files, gloss, skels, text의 순서로서 구성된다.
    2. 각각이 correspondings가 맞야야 한다.
    3. files, gloss, text의 경우에는 annotation 그대로 때서 붙이면 된다.
    4. 다만 skels에 대한 normalization 처리는 안된 상태이다.
        a. 그리고, 기존에 있던 data의 문제점은 frame에 대한 value가 입력이 안된 것이었다. data[150::151]
        b. 따라서 지금 해야할 것은 skels에 대한 처리이다.



    :param file_path:
    :return:
    """

    list_float = []

    with open(file_path, 'r') as f:
        list_raw = f.readlines()
        list_str_split = [x.split() for x in list_raw]
        list_float_split = [list(map(float, each_list)) for each_list in list_str_split]

        test = 1

def new_h5_to_text_writer(h5_read_path, write_dir, csv_dir):
    df_path_pre = "PHOENIX-2014-T."

    data_split = ['train', 'dev', 'test']
    write_suffix = ['files', 'gloss', 'text', 'skels']

    each_path = ["train.corpus.csv", "dev.corpus.csv", "test.corpus.csv"]

    text_file_list_list = []

    with h5py.File(h5_read_path, 'r') as file_to_read:
        # dataset split에 따라서 data의 정렬
        for each_split_h5, each_split_csv in zip(data_split, each_path):

            # 여기서 file을 열어주고 write해줄 필요가 있다. 아니다... 한번에 처리하는 것이 좋을 것 같다.
            text_file_list = []
            each_df = pd.read_csv(os.path.join(csv_dir, df_path_pre+each_split_csv), delimiter='|')
            # dataframe은 이미 sentence number 기준으로 정렬이 잘 되어 있다.
            each_h5 = file_to_read.get(each_split_h5)
            # data frame에서 key기준으로 값을 가저온다. 이후 이를 통해서 h5에 접근하여 수정한다.
            for index, row in tqdm(each_df.iterrows(), total=len(each_df)):
                text_vector = []
                text_vector.append(each_split_h5+'/'+row['name'])
                text_vector.append(row['orth'])  # gloss
                text_vector.append(row['translation'] +' .')  # text 여기서 구두점 추가 필요

                # h5에서 frame 별 float를 넣어 줘야 한다.
                each_np_skel = np.asarray(each_h5.get(row['name']))
                row_indices_float = np.arange(0, each_np_skel.shape[0], 1) / each_np_skel.shape[0]
                # 여기서 normalize 해줘야 할 수도 있다.. 저자는 3으로 나누기를 하였다.
                each_np_skel_with_frame = np.concatenate((each_np_skel, row_indices_float.reshape(-1, 1)), axis=1)
                # 여기서 string으로 변환 해줄 필요가 있다.
                # 이 부분에서 매우 연산이 많이 걸리는 것으로 생각된다.

                each_np_skel_with_frame_2str_raw = each_np_skel_with_frame.flatten().tolist()
                """
                numpy 자체 method에는 문제가 많다...
                each_np_skel_with_frame_2str_raw = np.array2string(each_np_skel_with_frame.flatten(),
                                               max_line_width=100000, precision=5,
                                               separator=' ', threshold=100000,
                                               suppress_small=True)[1:-1]  # [1:-1]은 앞뒤의 []를 제거하기 위함이다.
                """
                each_np_skel_with_frame_2str = ' '.join(str(element) for element in each_np_skel_with_frame_2str_raw)
                text_vector.append(each_np_skel_with_frame_2str)

                text_file_list.append(text_vector)

            text_file_list_list.append(text_file_list)

        # 이제 text_file_list_list를 write해주면 된다.
        for split_name, text_each_split in zip(data_split, text_file_list_list):

            f_descriptor_list = []
            for each_suffix in write_suffix:
                # files, gloss, text, skels 순서임
                f = open(os.path.join(write_dir, split_name + '.' + each_suffix), 'w')
                f_descriptor_list.append(f)

            for each_vector in tqdm(text_each_split, total=len(text_each_split), leave=False):
                for idx, each_f_descriptor in enumerate(f_descriptor_list):
                    encoded_str = each_vector[idx].encode('utf-8')
                    each_f_descriptor.write(encoded_str+'\n')

            # descriptor 꺼주기
            for each_f_descriptor in f_descriptor_list:
                each_f_descriptor.close()

    test = 1

if __name__ == '__main__':
    # file_path = r'Z:\2023_PBL\sign_data\raw_data\PHOENIX-2014-T-release-v3\example_data\test.skels'
    # original_loader(file_path)

    h5_read_path = r'Z:\2023_PBL\sign_data\raw_data\PHOENIX-2014-T-release-v3\PHOENIX_processing\keypoints-01-raw_3d.h5'
    write_dir = r'Z:\2023_PBL\sign_data\raw_data\PHOENIX-2014-T-release-v3\PHOENIX_processing\new_processed_data'
    os.makedirs(write_dir, exist_ok=True)
    csv_dir = r'Z:\2023_PBL\sign_data\raw_data\PHOENIX-2014-T-release-v3\PHOENIX-2014-T\annotations\manual'
    new_h5_to_text_writer(h5_read_path, write_dir, csv_dir)
