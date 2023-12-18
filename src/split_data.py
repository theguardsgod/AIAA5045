#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   split_data.py
@Time    :   2023/12/18 14:24:57
@Author  :   ZHU Zhengjie 
@Desc    :   None
'''

import pandas as pd

# Read the CSV file
df = pd.read_csv('/home/zhuzhengjie/root/CODE/AIAA5045/data/converted_label/train.csv')

# Define the desired number of samples for each class
class_samples = {
    0: 355,
    1: 355,
    2: 354,
    3: 327,
    4: 355,
    5: 115,
    6: 142
}

# Initialize empty dataframes for the two parts
part1_df = pd.DataFrame()
part2_df = pd.DataFrame()

# Split the data into two parts
for class_label, num_samples in class_samples.items():
    # Filter rows for the current class
    class_rows = df[df['label_test'] == class_label]

    # Take the desired number of samples for part 1
    part1_samples = class_rows.sample(n=num_samples)

    # Take the remaining samples for part 2
    part2_samples = class_rows.drop(part1_samples.index)

    # Append the samples to the respective dataframes
    part1_df = pd.concat([part1_df, part1_samples])
    part2_df = pd.concat([part2_df, part2_samples])

# Save the two parts as separate CSV files
part1_df.to_csv('/home/zhuzhengjie/root/CODE/AIAA5045/data/converted_label/labeled_20_split.csv', index=False)
part2_df.to_csv('/home/zhuzhengjie/root/CODE/AIAA5045/data/converted_label/unlabeled_80_split.csv', index=False)
