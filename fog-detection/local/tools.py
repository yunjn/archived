# coding=utf-8
import os
import cv2
import json
import shutil
import requests
from time import strftime, time, localtime


# 信息打印
def LOG(msg, mark=1):
    if mark == 1:
        text = '[INFO]:'
    elif mark == 2:
        text = '[WARNING]:'
    else:
        text = '[ERROR]:'
    print(strftime('%Y-%m-%d %H:%M:%S', localtime(time())) + text + msg)


# video_name:视频名字，不包含路径 interval:帧率间隔
def get_frame_from_video(video_name, interval):
    video_ = video_name.split('/')[-1]
    video_ = video_[0:-4]
    save_path = '../resources/tmp/'  # + video_ + '/'
    is_exists = os.path.exists(save_path)

    if not is_exists:  # 路径
        os.makedirs(save_path)
        LOG(f'Create {save_path}')
    else:
        shutil.rmtree(save_path)
        os.makedirs(save_path)
        LOG(f'{save_path} exist')

    video_capture = cv2.VideoCapture(video_name)
    frame_num = 0
    img_num = 0

    while True:
        flag, frame = video_capture.read()
        frame_num += 1
        if frame_num % interval == 0 and frame is not None:
            img_num += 1
            save_name = save_path + video_ + '_' + str(img_num) + '.jpg'
            cv2.imwrite(save_name, frame)
            LOG(f'{video_}_{img_num} is saved')
        if not flag:
            if img_num == 0:
                LOG(f'{video_name} is not exist!', 0)
                return 0
            LOG('SUCCESS!!!')
            return img_num


if __name__ == '__main__':
    video_name = os.getcwd()+'/test.mp4'
    interval = 3
    get_frame_from_video(video_name, interval)
