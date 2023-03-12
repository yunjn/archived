import os
import sys
import cv2
import json
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms

sys.path.append('..')
from resnet.model import resnet50

class Predict:
    def __init__(self, file_name: str, num=5):
        self.file_name = file_name
        self.num = num
        self.device = torch.device(
            'cuda:0' if torch.cuda.is_available() else 'cpu')
        # load fog_classes
        json_path = '../resources/out/classes.json'
        assert os.path.exists(
            json_path), "file: '{}' dose not exist.".format(json_path)
        json_file = open(json_path, "r")
        self.class_name = json.load(json_file)

        # create model
        self.model = resnet50(num_classes=3).to(self.device)

        # load model weights
        weights_path = '../resources/out/fog.pth'
        assert os.path.exists(
            weights_path), "file: '{}' dose not exist.".format(weights_path)
        self.model.load_state_dict(torch.load(
            weights_path, map_location=self.device))

    def video(self):
        frame_num = 1
        cap = cv2.VideoCapture(self.file_name)
        data_transform = transforms.Compose(
            [transforms.Resize(256),
             transforms.CenterCrop(224),
             transforms.ToTensor(),
             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        while(True):
            ret, frame = cap.read()
            frame_num += 1
            if frame_num % self.num == 0 and frame is not None:
                img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                img = data_transform(img)
                img = torch.unsqueeze(img, dim=0)
                self.model.eval()
                with torch.no_grad():
                    output = torch.squeeze(
                        self.model(img.to(self.device))).cpu()
                    result = torch.softmax(output, dim=0)
                    result_sort = np.argsort(result).numpy()
                min_rate = f'{self.class_name[str(result_sort[0])]:^9} {int(result[result_sort[0]]*1000)/1000}'
                mid_rate = f'{self.class_name[str(result_sort[1])]:^9} {int(result[result_sort[1]]*1000)/1000}'
                max_rate = f'{self.class_name[str(result_sort[2])]:^9} {int(result[result_sort[2]]*1000)/1000}'
                frame = cv2.resize(frame, (720, 405))
                cv2.putText(frame, max_rate, (40, 50),
                            cv2.FONT_HERSHEY_DUPLEX, 0.9, (0, 0, 255), 1)
                cv2.putText(frame, mid_rate, (40, 80),
                            cv2.FONT_HERSHEY_DUPLEX, 0.9, (0, 0, 255), 1)
                cv2.putText(frame, min_rate, (40, 110),
                            cv2.FONT_HERSHEY_DUPLEX, 0.9, (0, 0, 255), 1)
                cv2.imshow(self.file_name.split('/')[-1], frame)
            if not ret or cv2.waitKey(5) & 0xFF == ord('q'):
                break
        cv2.destroyAllWindows()

    def image(self):
        data_transform = transforms.Compose(
            [transforms.Resize(256),
             transforms.CenterCrop(224),
             transforms.ToTensor(),
             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        img = Image.open(self.file_name)
        plt.imshow(img)
        img = data_transform(img)
        img = torch.unsqueeze(img, dim=0)
        self.model.eval()
        with torch.no_grad():
            output = torch.squeeze(
                self.model(img.to(self.device))).cpu()
            result = torch.softmax(output, dim=0)
            result_max = torch.argmax(result).numpy()
        print_res = "class: {}   probability: {:.3}".format(self.class_name[str(result_max)],
                                                     result[result_max].numpy())
        plt.title(print_res)
        plt.show()
