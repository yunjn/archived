from flask import Flask, render_template, request, jsonify
from config import *
from util import *
import torch
import argparse
import numpy as np
from PIL import Image
import cv2
from skimage.transform import resize
from network.e2e_resnet34_2b_gfm_tt import e2e_resnet34_2b_gfm_tt
import base64


def get_args():
    parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
    parser.add_argument('--cuda', action='store_true', help='use cuda?')
    parser.add_argument('--arch', type=str, required=False, default='e2e_resnet34_2b_gfm_tt',
                        choices=["e2e_resnet34_2b_gfm_tt"], help="net backbone")
    parser.add_argument('--resize', action='store_true',
                        help="resize testing: resize 1/2 for testing")
    parser.add_argument('--hybrid', action='store_true',
                        help="hybrid testing, 1/2 focus + 1/3 glance")
    args = parser.parse_args()
    return args


def inference_function(scale_img, scale_trimap=None):

    pred_list = []
    tensor_img = torch.from_numpy(scale_img.astype(
        np.float32)[np.newaxis, :, :, :]).permute(0, 3, 1, 2).cuda()
    input_t = tensor_img
    pred_global, pred_local, pred_fusion = model(input_t)

    if args.arch.rfind('tt') > 0:
        pred_global = pred_global.data.cpu().numpy()
        pred_global = gen_trimap_from_segmap_e2e(pred_global)
    else:
        pred_global = pred_global.data.cpu().numpy()
        # pred_global = gen_bw_from_segmap_e2e(pred_global)

    pred_local = pred_local.data.cpu().numpy()[0, 0, :, :]
    pred_fusion = pred_fusion.data.cpu().numpy()[0, 0, :, :]

    return pred_global, pred_local, pred_fusion


def inference_img_gfm(img, option):

    h, w, c = img.shape
    new_h = min(MAX_SIZE_H, h - (h % 32))
    new_w = min(MAX_SIZE_W, w - (w % 32))

    if args.hybrid:
        ####
        # Combine 1/3 glance+1/2 focus
        global_ratio = 1/3
        local_ratio = 1/2
        resize_h = int(h*global_ratio)
        resize_w = int(w*global_ratio)
        new_h = min(MAX_SIZE_H, resize_h - (resize_h % 32))
        new_w = min(MAX_SIZE_W, resize_w - (resize_w % 32))
        scale_img = resize(img, (new_h, new_w))*255.0
        pred_glance_1, pred_focus_1, pred_fusion_1 = inference_function(
            scale_img)
        pred_glance_1 = resize(pred_glance_1, (h, w))*255.0
        resize_h = int(h*local_ratio)
        resize_w = int(w*local_ratio)
        new_h = min(MAX_SIZE_H, resize_h - (resize_h % 32))
        new_w = min(MAX_SIZE_W, resize_w - (resize_w % 32))
        scale_img = resize(img, (new_h, new_w))*255.0
        pred_glance_2, pred_focus_2, pred_fusion_2 = inference_function(
            scale_img)
        pred_focus_2 = resize(pred_focus_2, (h, w))
        if option == 'tt':
            pred_fusion = get_masked_local_from_global_test(
                pred_glance_1, pred_focus_2)
        elif option == 'bt':
            pred_fusion = pred_glance_1/255.0 - pred_focus_2
            pred_fusion[pred_fusion < 0] = 0
        else:
            pred_fusion = pred_glance_1/255.0 + pred_focus_2
            pred_fusion[pred_fusion > 1] = 1
        return [pred_glance_1, pred_focus_2, pred_fusion]

    else:
        if args.resize:
            resize_h = int(h/2)
            resize_w = int(w/2)
            new_h = min(MAX_SIZE_H, resize_h - (resize_h % 32))
            new_w = min(MAX_SIZE_W, resize_w - (resize_w % 32))
        scale_img = resize(img, (new_h, new_w))*255.0
        pred_glance, pred_focus, pred_fusion = inference_function(
            args, model, scale_img)
        pred_focus = resize(pred_focus, (h, w))
        pred_glance = resize(pred_glance, (h, w))*255.0
        pred_fusion = resize(pred_fusion, (h, w))

        return [pred_glance, pred_focus, pred_fusion]


def exec(img_path: str):
    refresh_folder(SAMPLES_RESULT_ALPHA_PATH)
    refresh_folder(SAMPLES_RESULT_COLOR_PATH)

    img = np.array(Image.open(img_path))[:, :, :3]

    h, w, c = img.shape
    if min(h, w) > SHORTER_PATH_LIMITATION:
        if h >= w:
            new_w = SHORTER_PATH_LIMITATION
            new_h = int(SHORTER_PATH_LIMITATION*h/w)
            img = cv2.resize(img, (new_w, new_h),
                             interpolation=cv2.INTER_LINEAR)
        else:
            new_h = SHORTER_PATH_LIMITATION
            new_w = int(SHORTER_PATH_LIMITATION*w/h)
            img = cv2.resize(img, (new_w, new_h),
                             interpolation=cv2.INTER_LINEAR)

    with torch.no_grad():
        torch.cuda.empty_cache()
        if args.arch.rfind('tt') > 0:
            predict = inference_img_gfm(img, 'tt')[2]
        elif args.arch.rfind('ft') > 0:
            predict = inference_img_gfm(img, 'ft')[2]
        elif args.arch.rfind('bt') > 0:
            predict = inference_img_gfm(img, 'bt')[2]

    composite = generate_composite_img(img, predict)
    predict = predict*255.0

    cv2.imwrite(os.path.join(SAMPLES_RESULT_COLOR_PATH,
                extract_pure_name('cache_color')+'.png'), composite)
    cv2.imwrite(os.path.join(SAMPLES_RESULT_ALPHA_PATH, extract_pure_name(
        'cache_alpha')+'.png'), predict.astype(np.uint8))

    # return (composite, predict)


def add_alpha_channel(img):
    b_channel, g_channel, r_channel = cv2.split(img)
    alpha_channel = np.ones(
        b_channel.shape, dtype=b_channel.dtype) * 255

    img_new = cv2.merge(
        (b_channel, g_channel, r_channel, alpha_channel))
    return img_new


def merge_img(jpg_img, png_img, y1, y2, x1, x2):
    if jpg_img.shape[2] == 3:
        jpg_img = add_alpha_channel(jpg_img)

    yy1 = 0
    yy2 = png_img.shape[0]
    xx1 = 0
    xx2 = png_img.shape[1]

    if x1 < 0:
        xx1 = -x1
        x1 = 0
    if y1 < 0:
        yy1 = - y1
        y1 = 0
    if x2 > jpg_img.shape[1]:
        xx2 = png_img.shape[1] - (x2 - jpg_img.shape[1])
        x2 = jpg_img.shape[1]
    if y2 > jpg_img.shape[0]:
        yy2 = png_img.shape[0] - (y2 - jpg_img.shape[0])
        y2 = jpg_img.shape[0]

    alpha_png = png_img[yy1:yy2, xx1:xx2, 3] / 255.0
    alpha_jpg = 1 - alpha_png

    for c in range(0, 3):
        jpg_img[y1:y2, x1:x2, c] = (
            (alpha_jpg*jpg_img[y1:y2, x1:x2, c]) + (alpha_png*png_img[yy1:yy2, xx1:xx2, c]))

    return jpg_img


args = get_args()
if args.arch == 'e2e_resnet34_2b_gfm_tt':
    model = e2e_resnet34_2b_gfm_tt(args)

ckpt = torch.load('model/model_r34_2b_gfm_tt.pth')
model.load_state_dict(ckpt, strict=True)

if args.cuda:
    model = model.cuda()

model.eval()

try:
    exec('static/cache/cache_img.png')
    print('Done!')
except:
    print('Error!')

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template("index.html")


@app.route('/matting/', methods=['GET', 'POST'])
def matting():
    image = request.files["file0"]
    print(image.filename)
    image.save('static/cache/cache_img.png')
    exec('static/cache/cache_img.png')
    with open('static/cache/cache_color.png', 'rb') as img_f:
        img_stream = img_f.read()
        img_stream = str(base64.b64encode(img_stream))[2:]
        img_stream = img_stream[:-1]
    return jsonify({"result": img_stream})


@app.route('/change_bg/', methods=['GET', 'POST'])
def change_bg():
    image = request.files["file1"]
    print(image.filename)
    image.save('static/cache/cache_bg.jpg')

    img_jpg_path = 'static/cache/cache_bg.jpg'
    img_png_path = 'static/cache/cache_color.png'

    # 读取图像
    img_jpg = cv2.imread(img_jpg_path, cv2.IMREAD_UNCHANGED)
    img_png = cv2.imread(img_png_path, cv2.IMREAD_UNCHANGED)

    img_jpg = cv2.resize(img_jpg, (img_png.shape[1], img_png.shape[0]),
                         interpolation=cv2.INTER_NEAREST)

    # # 开始叠加
    res_img = merge_img(img_jpg, img_png, 0,
                        img_png.shape[0], 0, img_png.shape[1])

    cv2.imwrite('static/cache/cache_ret.png', res_img)

    with open('static/cache/cache_ret.png', 'rb') as img_f:
        img_stream = img_f.read()
        img_stream = str(base64.b64encode(img_stream))[2:]
        img_stream = img_stream[:-1]
    return jsonify({"result": img_stream})


if __name__ == "__main__":
    app.run(host='127.0.0.1', port=8080)
