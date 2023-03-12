# coding=utf-8
import re
import tkinter.filedialog
from tkinter import *
from tkinter import messagebox as msgbox
import torch
import numpy as np
import argparse
from tools import Edge

root = Tk()
root.title('test')
root.geometry('720x480')
root.configure(bg='GhostWhite')

numIdx = 60
frames = [PhotoImage(file='assets/bg.gif', format='gif -index %i' % i)
          for i in range(numIdx)]


def net():
    compile_ip = re.compile(
        '^(1\d{2}|2[0-4]\d|25[0-5]|[1-9]\d|[1-9])\.(1\d{2}|2[0-4]\d|25[0-5]|[1-9]\d|\d)\.(1\d{2}|2[0-4]\d|25[0-5]|[1-9]\d|\d)\.(1\d{2}|2[0-4]\d|25[0-5]|[1-9]\d|\d)$')
    if compile_ip.match(text.get()):
        _file_name = 'http://admin:admin@'+text.get()+':8081/video'
        # out = predict.Predict(_file_name, 5)
        # out.video()
    else:
        msgbox.showerror('test', 'This IP is illegal')

def _net():
    url = text.get()
    with torch.no_grad():
        track.show_vid = True
        track.detect(expend=True,interval=5,src=url)
    pass

def update(idx):
    frame = frames[idx]
    idx += 1
    label.configure(image=frame)
    root.after(50, update, idx % numIdx)


def get_dege():
    _file_name = tkinter.filedialog.askopenfilename()
    if _file_name[-3:] in ['mp4', ['avi']]:
        track.get_edge(interval=1,src=_file_name,expend=True)
    else:
        msgbox.showerror('test', '请输入正确的文件')


def tracker():
    _file_name = tkinter.filedialog.askopenfilename()
    if _file_name[-3:] in ['mp4', 'avi']:
        if _file_name.find('002') != -1:
            src = np.float32([[(288, 480), (335, 245), (390, 245), (480, 480)]])
            dst = np.float32([[(288, 480), (288, 50), (480, 50), (480, 480)]])
            track.set_roi(src,dst)
        else:
            src = np.float32([[(200, 440), (270, 250), (380, 250), (520, 440)]])
            dst = np.float32([[(0, 440), (0, 0), (580, 0), (580, 440)]])
            track.set_roi(src,dst)
        with torch.no_grad():
            track.show_vid = True
            track.detect(expend=True,interval=5,src=_file_name)
    else:
        msgbox.showerror('test', '请输入正确的文件')
        return 



parser = argparse.ArgumentParser()
parser.add_argument('--yolo_weights', nargs='+', type=str, default='yolov5s.pt', help='model.pt path(s)')
parser.add_argument('--deep_sort_weights', type=str, default='deep_sort_pytorch/deep_sort/deep/checkpoint/ckpt.t7', help='ckpt.t7 path')
parser.add_argument('--output', type=str, default='assets/output', help='output folder')  # output folder
parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
parser.add_argument('--conf-thres', type=float, default=0.4, help='object confidence threshold')
parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
parser.add_argument('--show-vid', action='store_true', help='display tracking video results')
parser.add_argument('--save-vid', action='store_true', help='save video tracking results')
parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 16 17')
parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
parser.add_argument('--augment', action='store_true', help='augmented inference')
parser.add_argument('--evaluate', action='store_true', help='augmented inference')
parser.add_argument("--config_deepsort", type=str, default="deep_sort_pytorch/configs/deep_sort.yaml")
parser.add_argument("--half", action="store_true", help="use FP16 half-precision inference")
parser.add_argument('--visualize', action='store_true', help='visualize features')
parser.add_argument('--max-det', type=int, default=1000, help='maximum detection per image')
parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
opt = parser.parse_args()
opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
track = Edge(opt=opt)

if __name__ == '__main__':
    f1 = Frame(root)
    f1.pack(side='bottom')
    text = Entry()
    text.pack()
    btn5 = Button(text='NET', command=_net)
    btn5.pack(side='top')

    btn1 = Button(f1, text='边缘提取', command=get_dege)
    btn1.pack(side='left')
    btn2 = Button(f1, text='跟踪预警', command=tracker)
    btn2.pack(side='left')
    label = Label(root)
    label.configure(bg='GhostWhite')
    label.pack()
    root.after(0, update, 0)
    root.mainloop()

