"""
해당 code의 목적은 취득한 원래 data를 model에 feed하기 위해서 어떤 식으로 변형해야하는 가를 보고자 함이다.

"""
import pandas as pd
import numpy as np
import os
import h5py
from tqdm import tqdm
import ray
import re
import glob
import json
import random

@ray.remote(num_cpus=1)
class Counter(object):
    def __init__(self):
        self.n = 0

    def increment(self):
        self.n += 1

    def read(self):
        return self.n

@ray.remote(num_cpus=1)
def process_row(row, each_h5, each_split_h5, counter):
    # dataframe으로 serliazable하게 변형된 자료구조이다. h5 -> dataframe
    text_vector = []
    text_vector.append(each_split_h5 + '/' + row['name'])
    text_vector.append(row['orth'])  # gloss
    text_vector.append(row['translation'] + ' .')  # text 여기서 구두점 추가 필요

    each_np_skel = each_h5.loc[row['name']][0]
    each_np_skel = each_np_skel / 3 # normalize to -1 ~ 1
    row_indices_float = np.arange(0, each_np_skel.shape[0], 1) / each_np_skel.shape[0]
    each_np_skel_with_frame = np.concatenate((each_np_skel, row_indices_float.reshape(-1, 1)), axis=1)

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
    counter.increment.remote()
    return text_vector

@ray.remote(num_cpus=1)
def process_row_korean(index, row, each_h5, each_split_h5, counter):
    # dataframe으로 serliazable하게 변형된 자료구조이다. h5 -> dataframe
    text_vector = []
    text_vector.append(each_split_h5 + '/' + index)
    text_vector.append(row['gloss'])  # gloss
    # 한국어의 경우에는 text vector가 없다.

    each_np_skel = each_h5.loc[index][0]
    each_np_skel = each_np_skel / 3 # normalize to -1 ~ 1 devide by 3, in here 5
    row_indices_float = np.arange(0, each_np_skel.shape[0], 1) / each_np_skel.shape[0]
    each_np_skel_with_frame = np.concatenate((each_np_skel, row_indices_float.reshape(-1, 1)), axis=1)

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
    counter.increment.remote()
    return text_vector

def new_h5_to_text_writer(h5_read_path, write_dir, csv_dir):
    df_path_pre = "PHOENIX-2014-T."

    data_split = ['train', 'dev', 'test']
    write_suffix = ['files', 'gloss', 'text', 'skels']

    each_path = ["train.corpus.csv", "dev.corpus.csv", "test.corpus.csv"]

    text_file_list_list = []

    # parallelize
    context = ray.init(num_cpus=40, include_dashboard=True)

    with h5py.File(h5_read_path, 'r') as file_to_read:
        # dataset split에 따라서 data의 정렬
        for each_split_h5, each_split_csv in zip(data_split, each_path):
            counter = Counter.remote()

            # 여기서 file을 열어주고 write해줄 필요가 있다. 아니다... 한번에 처리하는 것이 좋을 것 같다.
            each_df = pd.read_csv(os.path.join(csv_dir, df_path_pre+each_split_csv), delimiter='|')
            # dataframe은 이미 sentence number 기준으로 정렬이 잘 되어 있다.
            # 병렬화에 따라서 재 정렬을 해줘야만 한다!
            each_h5 = file_to_read.get(each_split_h5)

            temp_keys = list(each_h5.keys())
            temp_data = list(map(lambda x: np.asarray(each_h5.get(x)), temp_keys))
            each_h5_dflized = pd.DataFrame({
                'name': temp_keys,
                'data': temp_data
            })
            each_h5_dflized = each_h5_dflized.set_index('name')
            each_h5_dflized = ray.put(each_h5_dflized)  # ray serialize

            # data frame에서 key기준으로 값을 가저온다. 이후 이를 통해서 h5에 접근하여 수정한다.
        
            distributed_processing = []
            for index, row in tqdm(each_df.iterrows(), total=len(each_df)):
                distributed_processing.append(process_row.remote(row, each_h5_dflized, each_split_h5, counter))

            text_file_list = ray.get(distributed_processing)
            text_file_list_list.append(text_file_list)

            with tqdm(total=len(each_df), desc="Progress Indicator") as pbar:
                while pbar.n < pbar.total:
                    n = ray.get(counter.read.remote())
                    pbar.n = n
                    pbar.refresh()

    ray.shutdown()
    # 이제 text_file_list_list를 write해주면 된다.
    for split_name, text_each_split in zip(data_split, text_file_list_list):

        f_descriptor_list = []
        for each_suffix in write_suffix:
            # files, gloss, text, skels 순서임
            f = open(os.path.join(write_dir, split_name + '.' + each_suffix), 'w', encoding='utf-8')
            f_descriptor_list.append(f)

        for each_vector in tqdm(text_each_split, total=len(text_each_split), leave=False):
            for idx, each_f_descriptor in enumerate(f_descriptor_list):
                each_f_descriptor.write(each_vector[idx]+'\n')

        # descriptor 꺼주기
        for each_f_descriptor in f_descriptor_list:
            each_f_descriptor.write('\r\n')
            each_f_descriptor.close()

    test = 1

def sentence_num_splitter(number_sentence, random_seed, multiplier):
    """
    dataframe에서 specific sentence만을 추출하기 위한 부분이다.
    :param number_sentence:
    :param random_seed:
    :return:
    """

    random.seed(random_seed)
    sentence_seq_array = [f"{num:04}" for num in range(1, number_sentence+1)]
    random.shuffle(sentence_seq_array)

    train_length = number_sentence * 8 // 10
    val_length = number_sentence * 1 // 10
    test_length = number_sentence - (train_length + val_length)

    train_data = sentence_seq_array[:train_length]
    val_data = sentence_seq_array[train_length:train_length+val_length]
    test_data = sentence_seq_array[train_length+val_length:]

    return [train_data, val_data, test_data]


def new_h5_to_text_writer_korean(h5_read_path, write_dir, csv_dir):

    num_sentence = 2000
    random_seed = 42
    multiplier = 16
    split_arr = sentence_num_splitter(num_sentence, random_seed, multiplier)

    data_split = ['train', 'dev', 'test']
    write_suffix = ['files', 'gloss', 'skels']

    text_file_list_list = []
    # parallelize
    context = ray.init(include_dashboard=True)

    with h5py.File(h5_read_path, 'r') as file_to_read:
        # dataset을 sharding할 필요가 있음
        total_df = pd.read_csv(csv_dir, delimiter='|')
        total_df.set_index('name', inplace=True)
        sharded_df_list = [total_df[total_df.index.str.contains('|'.join(each_split_arr))]
                           for each_split_arr in split_arr]

        temp_keys = list(file_to_read.keys())
        temp_data = list(map(lambda x: np.asarray(file_to_read.get(x)), temp_keys))
        each_h5_dflized = pd.DataFrame({
            'name': temp_keys,
            'data': temp_data
        })
        each_h5_dflized = each_h5_dflized.set_index('name')
        each_h5_dflized = ray.put(each_h5_dflized)  # ray serialize

        for split_name, each_df in zip(data_split, sharded_df_list):
            counter = Counter.remote()

            # data frame에서 key기준으로 값을 가저온다. 이후 이를 통해서 h5에 접근하여 수정한다.
            distributed_processing = []
            for index, row in tqdm(each_df.iterrows(), total=len(each_df)):
                distributed_processing.append(process_row_korean.remote(index, row, each_h5_dflized, split_name, counter))

            text_file_list = ray.get(distributed_processing)
            text_file_list_list.append(text_file_list)

            with tqdm(total=len(each_df), desc="Progress Indicator") as pbar:
                while pbar.n < pbar.total:
                    n = ray.get(counter.read.remote())
                    pbar.n = n
                    pbar.refresh()

    ray.shutdown()
    # 이제 text_file_list_list를 write해주면 된다.
    for split_name, text_each_split in zip(data_split, text_file_list_list):

        f_descriptor_list = []
        for each_suffix in write_suffix:
            # files, gloss, skels 순서임
            f = open(os.path.join(write_dir, split_name + '.' + each_suffix), 'w', encoding='utf-8')
            f_descriptor_list.append(f)

        for each_vector in tqdm(text_each_split, total=len(text_each_split), leave=False):
            for idx, each_f_descriptor in enumerate(f_descriptor_list):
                each_f_descriptor.write(each_vector[idx] + '\n')

        # descriptor 꺼주기
        for each_f_descriptor in f_descriptor_list:
            each_f_descriptor.write('\r\n')
            each_f_descriptor.close()

def korean_anno_df_generator_single(korean_raw_anno_dir, write_dir):
    """
    F pose에 대하여,


    :return:
    """
    pattern = os.path.join(korean_raw_anno_dir, '[0-1][0-9]', '*_F_*')
    dir_list = glob.glob(pattern)
    dir_list.sort()

    temp_list = []

    ray.init(num_cpus=40)
    counter = Counter.remote()


    for each_json_file in tqdm(dir_list, total=len(dir_list)):
        temp_vector = []
        with open(each_json_file, 'r', encoding='utf-8') as f:
            temp_json = json.load(f)

            file_name = temp_json['metaData']['name'].split('.')[0]
            temp_vector.append(file_name)

            gloss_vector = []
            for each_gloss in temp_json['data']:
                gloss_vector.append(each_gloss['attributes'][0]['name'].strip('\n'))
            temp_vector.append(' '.join(gloss_vector))
        temp_list.append(temp_vector)

    df = pd.DataFrame(temp_list, columns=['name', 'gloss'])
    df.set_index('name', inplace=True)
    df.to_csv(os.path.join(write_dir, 'train.corpus.csv'), sep='|', encoding='utf-8')

@ray.remote(num_cpus=1)
def process_json_parallel(json_file, counter):
    temp_vector = []
    with open(json_file, 'r', encoding='utf-8') as f:
        temp_json = json.load(f)

        file_name = temp_json['metaData']['name'].split('.')[0]
        temp_vector.append(file_name)

        gloss_vector = []
        for each_gloss in temp_json['data']:
            word = each_gloss['attributes'][0]['name'].replace('\n', '')
            gloss_vector.append(word)
        temp_vector.append(' '.join(gloss_vector))
        
    counter.increment.remote()
    return temp_vector

def korean_anno_df_generator_parallel(korean_raw_anno_dir, write_dir):
    pattern = os.path.join(korean_raw_anno_dir, '[0-1][0-9]', '*_F_*')
    # pattern = os.path.join(korean_raw_anno_dir, '01', '*_F_*')
    dir_list = glob.glob(pattern)
    dir_list.sort()

    ray.init(num_cpus=40)
    counter = Counter.remote()

    # each_json_file in dir list
    distributed_process = [
        process_json_parallel.remote(each_json_file, counter) for each_json_file in dir_list
    ]
    result_list = ray.get(distributed_process)

    with tqdm(total=len(dir_list), desc="Progress Indicator") as pbar:
        while pbar.n < pbar.total:
            n = ray.get(counter.read.remote())
            pbar.n = n
            pbar.refresh()

    ray.shutdown()

    df = pd.DataFrame(result_list, columns=['name', 'gloss'])
    df.set_index('name', inplace=True)
    df.to_csv(os.path.join(write_dir, 'kroean_train.corpus.csv'), sep='|', encoding='utf-8')


if __name__ == '__main__':
    # file_path = r'Z:\2023_PBL\sign_data\raw_data\PHOENIX-2014-T-release-v3\example_data\test.skels'
    # original_loader(file_path)

    # korean_raw_anno_dir = r'Z:\2023_PBL\sign_data\temp_sign_data\수어 영상\1.Training\morpheme'
    korean_raw_anno_dir = r'/data/2023_PBL/sign_data/temp_sign_data/수어 영상/1.Training/morpheme'
    # korean_df_path = r'Z:\2023_PBL\sign_data\temp_sign_data\수어 영상\1.Training\morpheme'
    korean_df_path = r'/data/2023_PBL/sign_data/temp_sign_data/수어 영상'
    # korean_anno_df_generator_parallel(korean_raw_anno_dir, korean_df_path)

    # h5_read_path = r'Z:\2023_PBL\sign_data\raw_data\PHOENIX-2014-T-release-v3\PHOENIX_processing\keypoints-01-raw_3d.h5'
    # write_dir = r'Z:\2023_PBL\sign_data\raw_data\PHOENIX-2014-T-release-v3\PHOENIX_processing\new_processed_data'
    # csv_dir = r'Z:\2023_PBL\sign_data\raw_data\PHOENIX-2014-T-release-v3\PHOENIX-2014-T\annotations\manual'
    # h5_read_path = r'/data/2023_PBL/sign_data/raw_data/PHOENIX-2014-T-release-v3/PHOENIX_processing/keypoints-01-raw_3d.h5'
    # write_dir = r'/data/2023_PBL/sign_data/raw_data/PHOENIX-2014-T-release-v3/PHOENIX_processing/new_processed_data'
    # csv_dir = r'/data/2023_PBL/sign_data/raw_data/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/annotations/manual'
    # h5_read_path = r'Z:\2023_PBL\sign_data\temp_sign_data\수어 영상\keypoints-01-raw_wb_korean_3d.h5'
    h5_read_path = r'/data/2023_PBL/sign_data/temp_sign_data/수어 영상/keypoints-01-raw_wb_3d_korean.h5'
    # h5_read_path_2 = r'/data/2023_PBL/sign_data/temp_sign_data/수어 영상/keypoints-01-raw_wb_3d_korean_part01.h5'
    # write_dir = r'Z:\2023_PBL\sign_data\temp_sign_data\수어 영상'
    write_dir = r'/data/2023_PBL/sign_data/temp_sign_data/수어 영상/processed_dataset'
    # write_dir_2 = r'/data/2023_PBL/sign_data/temp_sign_data/수어 영상/processed_dataset01'
    # csv_dir = r'Z:\2023_PBL\sign_data\temp_sign_data\수어 영상\korean_train.corpus.csv'
    csv_dir = r'/data/2023_PBL/sign_data/temp_sign_data/수어 영상/korean_train.corpus.csv'
    # csv_dir_2 = r'/data/2023_PBL/sign_data/temp_sign_data/수어 영상/korean_train_part01.corpus.csv'
    os.makedirs(write_dir, exist_ok=True)

    new_h5_to_text_writer_korean(h5_read_path, write_dir, csv_dir)
