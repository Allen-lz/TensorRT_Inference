# -*-coding: utf-8 -*-
"""
https://zhuanlan.zhihu.com/p/159379768
使用onnx runtime进行inference
"""
import os
import sys
import matplotlib.pylab as plt
import numpy as np
import time
sys.path.append(os.getcwd())
sys.path.append(".")
sys.path.append("..")
# 这个和训练的config是一样的
from configs_T import config
import torch
import onnxruntime as ort
import cv2
from PIL import Image
import base64


class ONNXModel():
    def __init__(self, onnx_path):
        """
        :param onnx_path:
        """
        # 建立inference的session(这点类似于tensorflow)
        # self.onnx_session = ort.InferenceSession(onnx_path)
        # self.onnx_session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
        self.onnx_session = ort.InferenceSession(onnx_path, providers=['CUDAExecutionProvider'])

        self.input_name = self.get_input_name(self.onnx_session)
        self.output_name = self.get_output_name(self.onnx_session)
        self.transforms = config['transforms']
        # print("input_name:{}".format(self.input_name))
        # print("output_name:{}".format(self.output_name))

    def get_output_name(self, onnx_session):
        """
        output_name = onnx_session.get_outputs()[0].name
        :param onnx_session:
        :return: 输出节点名称的list
        """
        output_name = []
        for node in onnx_session.get_outputs():
            output_name.append(node.name)
        return output_name

    def get_input_name(self, onnx_session):
        """
        input_name = onnx_session.get_inputs()[0].name
        :param onnx_session:
        :return: 输入节点名称的list
        """
        input_name = []
        for node in onnx_session.get_inputs():
            input_name.append(node.name)
        return input_name

    def get_input_feed(self, input_name, image_numpy):
        """
        将真实的输入numpy放入到input_feed中去, 构成一个{node_name: numpy}
        input_feed={self.input_name: image_numpy}
        :param input_name:
        :param image_numpy:
        :return:
        """
        input_feed = {}
        for name in input_name:
            input_feed[name] = image_numpy
        return input_feed

    def forward(self, image_numpy):
        '''
        # image_numpy = image.transpose(2, 0, 1)
        # image_numpy = image_numpy[np.newaxis, :]
        # onnx_session.run([output_name], {input_name: x})
        # :param image_numpy:
        # :return:
        '''
        # 输入数据的类型必须与模型一致,以下三种写法都是可以的
        # scores, boxes = self.onnx_session.run(None, {self.input_name: image_numpy})
        # scores, boxes = self.onnx_session.run(self.output_name, input_feed={self.input_name: iimage_numpy})
        input_feed = self.get_input_feed(self.input_name, image_numpy)
        fake_img = self.onnx_session.run(self.output_name, input_feed=input_feed)
        return fake_img

    def preprocess(self, img_path):
        img = Image.open(img_path).convert('RGB')
        img = self.transforms(img).unsqueeze(0)
        img = np.array(img.float())

        return img

    def base64_cv2(self, base64_str):
        """
        :param base64_str:
        :return: 被解码成的图片
        """
        imgString = base64.b64decode(base64_str)
        # 以数据流的形式读入,并转化为array格式的对象
        nparr = np.frombuffer(imgString, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return image

    def preprocess_base64toimg(self, base64):
        cv_img = cv2.cvtColor(self.base64_cv2(base64), cv2.COLOR_BGR2RGB)
        img = Image.fromarray(cv_img)
        img = self.transforms(img).unsqueeze(0)
        img = np.array(img.float())
        return img

    def postprocess(self, var):
        var = var[0]
        var = torch.Tensor(var).permute(1, 2, 0).numpy()
        var[var < 0] = 0
        var[var > 1] = 1
        var = var * 255
        # return Image.fromarray(var.astype('uint8'))
        return cv2.cvtColor(var.astype('uint8'), cv2.COLOR_BGR2RGB)

if __name__ == "__main__":
    # E:\PycharmProjects\Face_to_Parameter\web_deployment\weights
    # model_path = r"E:\PycharmProjects\Face_to_Parameter\web_deployment\weights\Face2params_Man_with_Points_best.onnx"
    model_path = r"weights/scrfd_500m_bnkps_shape640x640.onnx"
    # img_dir = r"E:\datasets\user_head"
    img_dir = "test_images"
    img_list = os.listdir(img_dir)
    net = ONNXModel(model_path)
    total_cost_time = 0
    for index, img_name in enumerate(img_list):
        img_path = os.path.join(img_dir, img_name)
        vis_img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)

        cost_time1 = time.time()
        # <<<<<<<<<<<<<<主要的推理代码<<<<<<<<<<<<<<<<
        input_img = net.preprocess(img_path)
        fake_img = net.forward(input_img)

        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        # params = fake_img[0][0] * 200 - 100
        # for p in params:
        #     print(str(round(p)) + ",", end="")
        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

        fake_img = net.postprocess(fake_img[0])
        fake_img = cv2.cvtColor(fake_img, cv2.COLOR_BGR2RGB)
        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        cost_time2 = time.time() - cost_time1
        total_cost_time += cost_time2

        plt.subplot(121), plt.imshow(vis_img)
        plt.subplot(122), plt.imshow(fake_img)
        plt.show()


