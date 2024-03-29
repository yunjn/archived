# coding=utf-8
import os
from shutil import copy, rmtree
import random


def mk_file(file_path: str):
    if os.path.exists(file_path):  # 如果文件夹存在，则先删除原文件夹在重新创建
        rmtree(file_path)
    os.makedirs(file_path)


random.seed(0)
split_rate = 0.1  # 分配总量10%为验证集
cwd = os.getcwd()
data_root = os.path.join(cwd, "fog_data")
origin_fog_path = os.path.join(data_root, "fog_img")
assert os.path.exists(origin_fog_path)
fog_class = [cla for cla in os.listdir(origin_fog_path)
             if os.path.isdir(os.path.join(origin_fog_path, cla))]

# 建立保存训练集的文件夹
train_root = os.path.join(data_root, "train")
mk_file(train_root)
for cla in fog_class:  # 建立每个类别对应的文件夹
    mk_file(os.path.join(train_root, cla))

# 建立保存验证集的文件夹
val_root = os.path.join(data_root, "val")
mk_file(val_root)
for cla in fog_class:  # 建立每个类别对应的文件夹
    mk_file(os.path.join(val_root, cla))

for cla in fog_class:
    cla_path = os.path.join(origin_fog_path, cla)
    images = os.listdir(cla_path)
    num = len(images)
    # 随机采样验证集的索引
    eval_index = random.sample(images, k=int(num*split_rate))
    for index, image in enumerate(images):
        if image in eval_index:
            # 将分配至验证集中的文件复制到相应目录
            image_path = os.path.join(cla_path, image)
            new_path = os.path.join(val_root, cla)
            copy(image_path, new_path)
        else:
            # 将分配至训练集中的文件复制到相应目录
            image_path = os.path.join(cla_path, image)
            new_path = os.path.join(train_root, cla)
            copy(image_path, new_path)
        print("\r[{}] processing [{}/{}]".format(cla, index+1, num), end="")
    print()
print("FINISHED!")
