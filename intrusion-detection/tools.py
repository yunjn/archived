import os
import cv2
import sys
import shutil
import torch
import numpy as np
from pathlib import Path
import torch.backends.cudnn as cudnn
from tkinter import messagebox as msgbox

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
sys.path.insert(0, './yolov5')

from yolov5.utils.downloads import attempt_download
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.datasets import LoadImages, LoadStreams
from yolov5.utils.general import check_img_size, non_max_suppression, scale_coords, check_imshow, xyxy2xywh
from yolov5.utils.torch_utils import select_device
from yolov5.utils.plots import Annotator, colors
from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort


class Edge:
    img_resolution = (720, 480)
    # (左下、左上、右上、右下)
    src = np.float32([[(200, 440), (270, 250), (380, 250), (520, 440)]])
    dst = np.float32([[(0, 440), (0, 0), (580, 0), (580, 440)]])
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    left_fit = []
    right_fit = []
    frame = ''

    # track
    out = source = yolo_weights = deep_sort_weights = show_vid = save_vid = save_txt = imgsz = evaluate = half = 0
    deepsort = device = half = model = stride = names = pt = jit = onnx = 0
    opt = 0

    # save
    vid_sobelx = vid_threshold = vid_add_lines = vid_contour = vid_remove_lines = vid_dilate = vid_warped = vid_fill_warp = vid_newwarp = vid_newwarp_with_expend = 0

    def __init__(self, opt, source='0'):
        # Initialize the parameters
        self.source = source
        self.opt = opt

        self.out, self.yolo_weights, self.deep_sort_weights, self.show_vid, self.save_vid, self.imgsz, self.evaluate, self.half = \
            opt.output, opt.yolo_weights, opt.deep_sort_weights, opt.show_vid, opt.save_vid, opt.imgsz, opt.evaluate, opt.half

        # initialize deepsort
        cfg = get_config()
        cfg.merge_from_file(opt.config_deepsort)
        attempt_download(self.deep_sort_weights,
                         repo='mikel-brostrom/Yolov5_DeepSort_Pytorch')
        self.deepsort = DeepSort(cfg.DEEPSORT.REID_CKPT,
                                 max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                                 max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                                 max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                                 use_cuda=True)

        # Initialize
        self.device = select_device(opt.device)
        self.half &= self.device.type != 'cpu'  # half precision only supported on CUDA

        if not self.evaluate:
            if os.path.exists(self.out):
                pass
                shutil.rmtree(self.out)  # delete output folder
            os.makedirs(self.out)  # make new output folder

        # Load model
        self.device = select_device(self.device)
        self.model = DetectMultiBackend(
            opt.yolo_weights, device=self.device, dnn=opt.dnn)
        self.stride, self.names, self.pt, self.jit, self.onnx = self.model.stride, self.model.names, self.model.pt, self.model.jit, self.model.onnx
        self.imgsz = check_img_size(
            self.imgsz, s=self.stride)  # check image size

        # Half
        # half precision only supported by PyTorch on CUDA
        self.half &= self.pt and self.device.type != 'cpu'
        if self.pt:
            self.model.model.half() if self.half else self.model.model.float()

    def set_img_resolution(self, img_resolution):
        self.img_resolution = img_resolution

    def set_roi(self, src, dst):
        self.M = cv2.getPerspectiveTransform(src, dst)
        self.Minv = cv2.getPerspectiveTransform(dst, src)

    def _get_edge_binary(self, img):
        sobelx = cv2.Sobel(img, cv2.CV_8UC1, 1, 0)  # 图像求导
        # sobely = cv2.Sobel(img, cv2.CV_8UC1, 0, 1, ksize=3)
        # sobelXY = cv2.Sobel(img, cv2.CV_8UC1, 1, 1, ksize=3)

        # self.vid_sobelx.write(sobelx)

        # 过滤低亮度(220,250)(start,end)
        _, threshold = cv2.threshold(sobelx, 220, 255, cv2.THRESH_TOZERO)

        # self.vid_threshold.write(threshold)

        img_gray = cv2.cvtColor(threshold, cv2.COLOR_BGR2GRAY)  # 转灰度图
        img = img_gray.copy()

        # 上面的线(y,x)
        cv2.line(img, (300, int(img.shape[0]*0.5)),
                 (int(img.shape[1]) - 300, int(img.shape[0]*0.5)), (255, 0, 0), 1)
        cv2.line(img, (200, int(img.shape[0]*0.8)),
                 (int(img.shape[1])-200, int(img.shape[0]*0.8)), (255, 0, 0), 1)
        
        # self.vid_add_lines.write(cv2.merge([img,img,img]))

        # 保留轮廓
        contours, hierarchy = cv2.findContours(
            img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        # contours[i]代表的是第i个轮廓，len(contours[i])代表的是第i个轮廓上所有的像素点数
        max_id = maxarea = tmp = 0
        for i in range(len(contours)):  # 找出面积最大的轮廓
            tmp = abs(cv2.contourArea(contours[i]))
            if tmp > maxarea:
                maxarea = tmp
                max_id = i

        img = np.zeros(img.shape, dtype=np.uint8)

        cv2.drawContours(img, contours, max_id, 255, 1, 8, hierarchy)  # 轮廓
        # cv2.drawContours(img, contours, max_id, 255, cv2.FILLED)  # 轮廓
        # self.vid_contour.write(cv2.merge([img,img,img]))

        img = cv2.bitwise_and(img_gray, img)  # 消除横线

        # self.vid_remove_lines.write(cv2.merge([img,img,img]))

        img = cv2.dilate(img, np.ones((3, 3), np.uint8))  # 膨胀

        # self.vid_dilate.write(cv2.merge([img,img,img]))
        return img

    def _find_line(self, binary_warped):
        # Take a histogram of the bottom half of the image
        histogram = np.sum(
            binary_warped[binary_warped.shape[0]//2:, :], axis=0)
        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int32(histogram.shape[0]/2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        # Choose the number of sliding windows
        nwindows = 9
        # Set height of windows
        window_height = np.int32(binary_warped.shape[0]/nwindows)
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Current positions to be updated for each window
        leftx_current = leftx_base
        rightx_current = rightx_base
        # Set the width of the windows +/- margin
        margin = 95
        # Set minimum number of pixels found to recenter window
        minpix = 50
        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []

        # Step through the windows one by one
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = binary_warped.shape[0] - (window+1)*window_height
            win_y_high = binary_warped.shape[0] - window*window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin
            # Identify the nonzero pixels in x and y within the window
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                              (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                               (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > minpix:
                leftx_current = np.int32(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:
                rightx_current = np.int32(np.mean(nonzerox[good_right_inds]))

        # Concatenate the arrays of indices
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        # Fit a second order polynomial to each
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)
        return left_fit, right_fit  # , left_lane_inds, right_lane_inds

    def _draw_area(self, undist, binary_warped, expend=False, colors=[(255, 0, 255), (255, 0, 0)]):
        ploty = np.linspace(
            0, binary_warped.shape[0]-1, binary_warped.shape[0])

        warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

        left_fitx = self.left_fit[0]*ploty**2 + \
            self.left_fit[1]*ploty + self.left_fit[2]
        right_fitx = self.right_fit[0]*ploty**2 + \
            self.right_fit[1]*ploty + self.right_fit[2]

        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array(
            [np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))
        cv2.fillPoly(color_warp, np.int_([pts]), colors[0])

        # self.vid_fill_warp.write(color_warp)

        newwarp = cv2.warpPerspective(
            color_warp, self.Minv, (undist.shape[1], undist.shape[0]))
        # self.vid_newwarp.write(newwarp)

        if expend:
            offset = np.abs(right_fitx - left_fitx)
            left_fitx += offset
            right_fitx += offset
            pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
            pts_right = np.array(
                [np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
            pts = np.hstack((pts_left, pts_right))
            cv2.fillPoly(color_warp, np.int_([pts]), colors[1])

            left_fitx -= offset*2
            right_fitx -= offset*2
            pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
            pts_right = np.array(
                [np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
            pts = np.hstack((pts_left, pts_right))
            cv2.fillPoly(color_warp, np.int_([pts]), colors[1])

        newwarp = cv2.warpPerspective(
            color_warp, self.Minv, (undist.shape[1], undist.shape[0]))
        # result = cv2.addWeighted(undist, 1, newwarp, 0.4, 0)

        # self.vid_newwarp_with_expend.write(newwarp)

        return newwarp  # result

    def get_edge(self, interval=1, expend=False, src='0'):
        if src != '0':
            self.source = src
        elif self.source == '0':
            print('请输入视频流啊喂!')
            return
        frame_num = 0
        src = cv2.VideoCapture(self.source)

        # self.vid_sobelx = cv2.VideoWriter(
        #     'sobelx.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (720, 480))
        # self.vid_threshold = cv2.VideoWriter(
        #     'threshold.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (720, 480))
        # self.vid_add_lines = cv2.VideoWriter(
        #     'add_lines.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (720, 480))
        # self.vid_contour = cv2.VideoWriter(
        #     'contour.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (720, 480))
        # self.vid_remove_lines = cv2.VideoWriter(
        #     'remove_lines.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (720, 480))
        # self.vid_dilate = cv2.VideoWriter(
        #     'dilate.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (720, 480))
        # self.vid_warped = cv2.VideoWriter(
        #     'warped.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (720, 480))
        # self.vid_fill_warp = cv2.VideoWriter(
        #     'fill_warp.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (720, 480))
        # self.vid_newwarp = cv2.VideoWriter(
        #     'newwarp.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (720, 480))
        # self.vid_newwarp_with_expend = cv2.VideoWriter(
        #     'newwarp_with_expend.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (720, 480))

        while True:
            flag, self.frame = src.read()
            frame_num += 1
            if frame_num % interval == 0 and self.frame is not None:
                self.frame = cv2.resize(self.frame, self.img_resolution)
                img_binary = self._get_edge_binary(self.frame)

                try:
                    # 透视变换
                    img_warped = cv2.warpPerspective(
                        img_binary, self.M, img_binary.shape[::-1], flags=cv2.INTER_LINEAR)
                    
                    # self.vid_warped.write(cv2.merge([img_warped,img_warped,img_warped]))

                    self.left_fit, self.right_fit = self._find_line(img_warped)

                    newwarp = self._draw_area(
                        self.frame, img_warped, expend=expend)

                    self.frame = cv2.addWeighted(self.frame, 1, newwarp, 0.4, 0)
                except:
                    pass
                cv2.imshow('output', self.frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        src.release()
        cv2.destroyAllWindows()
        # self.vid_sobelx.release()
        # self.vid_threshold.release()
        # self.vid_add_lines.release()
        # self.vid_contour.release()
        # self.vid_remove_lines.release()
        # self.vid_dilate.release()
        # self.vid_warped.release()
        # self.vid_fill_warp.release()
        # self.vid_newwarp.release()
        # self.vid_newwarp_with_expend.release()

    def detect(self, src='0', expend=False, interval=1):
        warning_colors = [(255, 0, 255), (255, 0, 0)]
        if src != '0':
            self.source = src
        elif self.source == '0':
            print('请输入视频流啊喂!')
            return
        self.webcam = self.source == '0' or self.source.startswith(
            'rtsp') or self.source.startswith('http') or self.source.endswith('.txt')

        # Set Dataloader
        vid_path, vid_writer = None, None
        # Check if environment supports image displays
        if self.show_vid:
            self.show_vid = check_imshow()

        # Dataloader
        if self.webcam:
            self.view_img = check_imshow()
            cudnn.benchmark = True  # set True to speed up constant image size inference
            dataset = LoadStreams(self.source, img_size=self.imgsz, stride=self.stride,
                                  auto=self.pt and not self.jit, img_resolution=self.img_resolution)
            bs = len(dataset)  # batch_size
        else:
            dataset = LoadImages(self.source, img_size=self.imgsz, stride=self.stride,
                                 auto=self.pt and not self.jit, img_resolution=self.img_resolution)
            bs = 1  # batch_size
        vid_path, vid_writer = [None] * bs, [None] * bs

        # Get names and colors
        names = self.model.module.names if hasattr(
            self.model, 'module') else self.model.names

        save_path = str(Path(self.out))

        if self.pt and self.device.type != 'cpu':
            self.model(torch.zeros(1, 3, *self.imgsz).to(self.device)
                       .type_as(next(self.model.model.parameters())))  # warmup

        for frame_idx, (path, img, im0s, vid_cap, s) in enumerate(dataset):

            if interval != 1 and frame_idx % interval == 0:
                continue

            img = torch.from_numpy(img).to(self.device)
            img = img.half() if self.half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # Inference
            visualize = increment_path(
                save_dir / Path(path).stem, mkdir=True) if self.opt.visualize else False
            pred = self.model(img, augment=self.opt.augment,
                              visualize=visualize)

            # Apply NMS
            pred = non_max_suppression(pred, self.opt.conf_thres, self.opt.iou_thres,
                                       self.opt.classes, self.opt.agnostic_nms, max_det=self.opt.max_det)

            # Process detections
            for i, det in enumerate(pred):  # detections per image
                if self.webcam:  # batch_size >= 1
                    p, im0, frame = path[i], im0s[i].copy(), dataset.count
                else:
                    p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

                self.frame = im0.copy()
                img_binary = self._get_edge_binary(self.frame)
                img_warped = cv2.warpPerspective(
                    img_binary, self.M, img_binary.shape[::-1], flags=cv2.INTER_LINEAR)

                try:
                    if interval != 1 and i % interval == 0 or i == 0:
                        self.left_fit, self.right_fit = self._find_line(
                            img_warped)
                    self.frame = self._draw_area(
                        self.frame, img_warped, expend=expend, colors=warning_colors)
                except:
                    print('请检查ROI区域 或 视频')

                save_path = str(Path(self.out) / Path(p).name)

                annotator = Annotator(im0, line_width=2, pil=not ascii)

                if det is not None and len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(
                        img.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        # add to string
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "

                    xywhs = xyxy2xywh(det[:, 0:4])
                    confs = det[:, 4]
                    clss = det[:, 5]

                    # pass detections to deepsort
                    outputs = self.deepsort.update(
                        xywhs.cpu(), confs.cpu(), clss.cpu(), im0)

                    # draw boxes for visualization
                    if len(outputs) > 0:
                        for j, (output, conf) in enumerate(zip(outputs, confs)):
                            bboxes = output[0:4]
                            id = output[4]
                            cls = output[5]

                            mid = np.array(
                                [(output[0] + output[2])/2, (output[1] + output[3])/2], dtype=np.int_)
                            print(
                                f'[xx-xx-xx] info: detected {names[int(cls)]}_{id}')

                            try:
                                b, g, r = self.frame[mid[0]][mid[1]]

                                if b != 0 or g != 0 or r != 0:
                                    warning_colors = [(0, 0, 255), (0, 0, 255)]
                                    im0 = cv2.putText(im0, f'{names[int(cls)]}_{id} invasion', (
                                        50, 50 + j * 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 200), 2)
                                    print(
                                        f'[xx-xx-xx] warning: {names[int(cls)]}_{id} invasion')
                                else:
                                    warning_colors = [
                                        (255, 0, 255), (255, 0, 0)]
                            except:
                                pass

                            c = int(cls)  # integer class
                            label = f'{id} {names[c]} {conf:.2f}'
                            annotator.box_label(
                                bboxes, label, color=colors(c, True))
                else:
                    self.deepsort.increment_ages()

                # Stream results
                im0 = annotator.result()
                im0 = cv2.addWeighted(im0, 1, self.frame, 0.4, 0)
                if self.show_vid:
                    cv2.imshow(p, im0)
                    if cv2.waitKey(1) == ord('q'):  # q to quit
                        raise StopIteration

                # Save results (image with detections)
                if self.save_vid:
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer
                        # if vid_cap:  # video
                        #     # fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        #     # w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        #     # h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        # else:  # stream
                        fps, w, h = 30, im0.shape[1], im0.shape[0]
                        if save_path.split('.')[-1] != 'mp4':
                            save_path += '.mp4'

                        vid_writer = cv2.VideoWriter(
                            save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer.write(im0)
