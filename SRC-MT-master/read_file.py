import pandas as pd

# 替换为您的文件路径
file_path = '/home/boya/Documents/SRC-MT/SRC-MT-master/data/skin/training.csv'

# 读取 CSV 文件
data = pd.read_csv(file_path)

# 获取行数
num_rows = len(data)
print("行数:", num_rows)
