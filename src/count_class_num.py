#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   count_class_num.py
@Time    :   2023/12/18 14:48:46
@Author  :   ZHU Zhengjie 
@Desc    :   None
'''

#count each class have how many samples from csv file. csv file second column is label.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
#count how many images in each class  from csv file. csv file second column is label.
def count_images_num_in_each_class (csv_file_path):
    data = pd.read_csv(csv_file_path)
    for range_num in range(7):
        print('class {} have {} samples'.format(range_num,len(data[data['label_test'] == range_num])))

print(count_images_num_in_each_class('/home/zhuzhengjie/root/CODE/AIAA5045/data/converted_label/labeled_20_split.csv'))

