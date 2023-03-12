# coding=utf-8
import os
import shutil
import predict
import tkinter.filedialog
from tkinter import *
#from tools import get_frame_from_video
from tkinter import messagebox as msgbox

root = Tk()
root.title('公路浓雾预警')
root.geometry('400x225')
root.configure(bg='GhostWhite')

numIdx = 60
frames = [PhotoImage(file='../resources/images/bg.gif', format='gif -index %i' % i)
          for i in range(numIdx)]


def predict_video_from_net():
    compile_ip = re.compile(
        '^(1\d{2}|2[0-4]\d|25[0-5]|[1-9]\d|[1-9])\.(1\d{2}|2[0-4]\d|25[0-5]|[1-9]\d|\d)\.(1\d{2}|2[0-4]\d|25[0-5]|[1-9]\d|\d)\.(1\d{2}|2[0-4]\d|25[0-5]|[1-9]\d|\d)$')
    if compile_ip.match(text.get()):
        _file_name = 'http://admin:admin@'+text.get()+':8081/video'
        out = predict.Predict(_file_name, 5)
        out.video()
    else:
        msgbox.showerror('公路浓雾预警', 'This IP is illegal')


def select_file():
    _file_name = tkinter.filedialog.askopenfilename()
    if _file_name[-3:] == 'mp4' or _file_name[-3:] == 'avi':
        out = predict.Predict(_file_name, 5)
        out.video()
    elif _file_name[-3:] == 'jpg' or _file_name[-3:] == 'png':
        out = predict.Predict(file_name=_file_name)
        out.image()
    else:
        msgbox.showerror('公路浓雾预警', 'Please choose the correct file format')


def clean():
    global file_name
    if os.path.exists('../resources/tmp'):
        shutil.rmtree('../resources/tmp')
        file_name = 'none'
    else:
        msgbox.showinfo('公路浓雾预警', 'Clean~')


def update(idx):
    frame = frames[idx]
    idx += 1
    label.configure(image=frame)
    root.after(50, update, idx % numIdx)


f1 = Frame(root)
f1.pack(side='bottom')

text = Entry()
text.pack()
btn5 = Button(text='NET', command=predict_video_from_net)
btn5.pack(side='top')

btn1 = Button(f1, text='select', command=select_file)
btn1.pack(side='left')
btn2 = Button(f1, text='clean', command=clean)
btn2.pack(side='left')
label = Label(root)
label.configure(bg='GhostWhite')
label.pack()
root.after(0, update, 0)
root.mainloop()
