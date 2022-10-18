# coding = gbk
"""
parse image url and transform image to tensor
"""
import time
from urllib import request

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms import RandomResizedCrop
from torchvision.transforms import functional as F
from model import *
# import selectivesearch

DOWNLOAD_FAIL=401
PREDICT_FAIL=402
SUCCESS=200
filter_size_w_h_s = (640,480,0.01) # 0.1
import ssl
ssl._create_default_https_context = ssl._create_unverified_context



def transform_image_torch(img, nor_mean, nor_std):
    detect_transform = transforms.Compose([
        # (h, w)
        transforms.Resize((360, 640)),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(nor_mean, nor_std)
    ])

    return detect_transform(img).unsqueeze(0)


def get_input(img_url):
    """
    从图片url得到输入模型中的tensor类型
    :param img_url: url
    :param nor_mean: 均值
    :param nor_std: 标准差
    :param rgb_form: 彩色or灰度
    :return: tensor
    """
    # 根据url抓包获取数据,解码得到图片,Numpy格式
    down_image_start_time = time.time()
    try:
        # resp_img_src = request.urlopen(img_url, timeout=8)
        img = cv2.imread(img_url)
    except Exception as e:
        return None
        # return 'parse image error.'
    else:
        down_image_take_time = time.time() - down_image_start_time
        transform_image_start_time = time.time()
        # img_np = np.asarray(bytearray(resp_img_src.read()), dtype='uint8')
        # # numpy array to opencv image
        #
        # img_cv = cv2.imdecode(img_np, cv2.IMREAD_UNCHANGED)
        img_cv = img
        # img = cv2.resize(img, (640, 360))
        # Opencv image  to PIL image
        # if rgb_form:
        #     img_src = Image.fromarray(cv2.cvtColor(img_src, cv2.COLOR_BGR2RGB)).convert('RGB')
        #     img_dest = Image.fromarray(cv2.cvtColor(img_dest, cv2.COLOR_BGR2RGB)).convert('RGB')
        #     pil_img = Image.fromarray(np.uint8(np.concatenate((np.array(img_src), np.array(img_dest)), axis=0)))
        # else:
        #     img_src = Image.fromarray(cv2.cvtColor(img_src, cv2.COLOR_BGR2GRAY)).convert('L')
        #     img_dest = Image.fromarray(cv2.cvtColor(img_dest, cv2.COLOR_BGR2GRAY)).convert('L')
        #     pil_img = Image.fromarray(np.uint8(np.concatenate((np.array(img_src), np.array(img_dest)), axis=0)))

        # 调用transform_image函数,得到输入模型中的格式，tensor类型
        # img_tensor = transform_image_torch(img, nor_mean, nor_std)
        transform_image_take_time = time.time() - transform_image_start_time

        return img_cv


def get_prediction(img_url, model):
    """
    model inference
    :param img_url: 请求接受的图片url
    :param model:
    :param nor_mean:
    :param nor_std:
    :param rgb_form:
    :return:
    """
    # 调用get_input函数根据url得到tensor类型的数据
    img_cv = get_input(img_url=img_url)

    # model = request.environ['HTTP_FLASK_MODEL']

    if img_cv is None:
        return None

    xyxys,confs,name_ens = model.predict(img_cv)

    return xyxys,confs,name_ens

