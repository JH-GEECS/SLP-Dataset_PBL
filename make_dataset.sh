#!/bin/bash

python /home/ejhwang/projects/SLP-Dataset/scripts/1_json2h5.py --delete --modes train_json_res dev_json_res test_json_res
python /home/ejhwang/projects/SLP-Dataset/scripts/2_lifting.py --delete
python /home/ejhwang/projects/SLP-Dataset/scripts/3_deltas.py --delete
python /home/ejhwang/projects/SLP-Dataset/scripts/4_norm.py --delete