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


#
# def crop(pil_img, zone,is_enhance=False):#(x1,y1,x2,y2)
#     img = pil_img.copy()
#     left, upper, right, lower = zone
#     """
#         所截区域图片保存
#     :param path: 图片路径
#     :param left: 区块左上角位置的像素点离图片左边界的距离
#     :param upper：区块左上角位置的像素点离图片上边界的距离
#     :param right：区块右下角位置的像素点离图片左边界的距离
#     :param lower：区块右下角位置的像素点离图片上边界的距离
#      故需满足：lower > upper、right > left
#     :param save_path: 所截图片保存位置
#     """
#     # img = Image.open(path)  # 打开图像
#     box = (left, upper, right, lower)
#     roi = img.crop(box)
#     if is_enhance:
#         roi = image_enhanced(roi)
#     return roi
#
# def get_ss_zone(img_path=None,scale=500):
#     # print(img_path)
#     # assert 1>2
#     # pilimg = Image.open(img_path).convert('RGB')  # (h,w)
#     img = np.asarray(img_path)
#     # img = img.astype('uint8')
#     img_lbl, regions = selectivesearch.selective_search(img, scale=scale, sigma=2.5, min_size=20) #sigma=2.5\0.9, min_size=20
#     # 创建一个集合 元素list(左上角x，左上角y,宽,高)
#     candidates = set()
#     for r in regions:
#         if r['rect'] in candidates:  # 排除重复的候选区
#             continue
#         if r['size'] < 500*2:  # 排除小于 2000 pixels的候选区域(并不是bounding box中的区域大小)
#             continue
#         x, y, w, h = r['rect']
#         # if w / h > 2 or h / w > 2:  # 排除扭曲的候选区域边框  即只保留近似正方形的
#         #     continue
#         candidates.add(r['rect'])
#
#     zones = [] ##(x1,y1,x2,y2)
#     pil_images = []
#     for x, y, w, h in candidates:
#         # print(x, y, w, h)
#         # cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 1)
#         zones.append( [x, y, x + w, y + h])
#         pil_images.append( crop(img_path,(x,y,x+w,y+h)) )
#     return pil_images,zones
#
# def gene_anchors2(image_path,scale=500):##(x1,y1,x2,y2)
#     return get_ss_zone(image_path,scale)
#
# class SelectsearchResizedCrop():
#     def __init__(self):
#         pass
#     def  forward(self, img):
#         pass
#
# class CustomRandomResizedCrop(RandomResizedCrop):
#     """Crop a random portion of image and resize it to a given size.
#
#     If the image is torch Tensor, it is expected
#     to have [..., H, W] shape, where ... means an arbitrary number of leading dimensions
#
#     A crop of the original image is made: the crop has a random area (H * W)
#     and a random aspect ratio. This crop is finally resized to the given
#     size. This is popularly used to train the Inception networks.
#
#     Args:
#         size (int or sequence): expected output size of the crop, for each edge. If size is an
#             int instead of sequence like (h, w), a square output size ``(size, size)`` is
#             made. If provided a sequence of length 1, it will be interpreted as (size[0], size[0]).
#
#             .. note::
#                 In torchscript mode size as single int is not supported, use a sequence of length 1: ``[size, ]``.
#         scale (tuple of float): Specifies the lower and upper bounds for the random area of the crop,
#             before resizing. The scale is defined with respect to the area of the original image.
#         ratio (tuple of float): lower and upper bounds for the random aspect ratio of the crop, before
#             resizing.
#         interpolation (InterpolationMode): Desired interpolation enum defined by
#             :class:`torchvision.transforms.InterpolationMode`. Default is ``InterpolationMode.BILINEAR``.
#             If input is Tensor, only ``InterpolationMode.NEAREST``, ``InterpolationMode.BILINEAR`` and
#             ``InterpolationMode.BICUBIC`` are supported.
#             For backward compatibility integer values (e.g. ``PIL.Image.NEAREST``) are still acceptable.
#
#     """
#     def forward(self, img):
#         """
#         Args:
#             img (PIL Image or Tensor): Image to be cropped and resized.
#
#         Returns:
#             PIL Image or Tensor: Randomly cropped and resized image.
#         """
#         i, j, h, w = self.get_params(img, self.scale, self.ratio)
#         # while not (j>filter_size_w_h_s[0]*filter_size_w_h_s[2] and i> filter_size_w_h_s[1]*filter_size_w_h_s[2] and \
#         #         filter_size_w_h_s[0] - (j+w)>filter_size_w_h_s[0]*filter_size_w_h_s[2] and \
#         #     filter_size_w_h_s[1] - (i+h)>filter_size_w_h_s[1]*filter_size_w_h_s[2]):
#
#         # x1,y1,x2,y2
#         while not (j > filter_size_w_h_s[0] * (filter_size_w_h_s[2] +0.06) and  \
#                    i > filter_size_w_h_s[1] * (filter_size_w_h_s[2]-0.02)   and  \
#                    filter_size_w_h_s[0] - (j + w) > filter_size_w_h_s[0] * (filter_size_w_h_s[2]+0.08) and \
#                    filter_size_w_h_s[1] - (i + h) > filter_size_w_h_s[1] * filter_size_w_h_s[2]):
#             i, j, h, w = self.get_params(img, self.scale, self.ratio)
#         return F.resized_crop(img, i, j, h, w, self.size, self.interpolation),torch.tensor([i, j, h, w]) ## y1,x1,h,w
#
# def numpy2cv(shape=(10,10)): ## shape = cvimg[0:2]
#     img = np.ones(shape)
#     img = np.float32(img)*0.5
#     img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
#     return img
#
# def blush(stack_img):##(480, 640, 3)
#     b= int(0.07*480)
#     midu_min = b*b*0.9*0.9
#     init_status = (np.random.randint(0,480-b-1) , np.random.randint(0,640-b-1))
#     nstep_x = ((480-init_status[0])//b)-1
#     init_midu = stack_img[init_status[0]:(init_status[0]+b),init_status[1]:(init_status[1]+b)].sum()
#     loop = 0
#     max_loop = 50
#     while init_midu<midu_min:
#         loop += 1
#         if loop > max_loop:
#             break
#         init_status = (np.random.randint(0, 480 - b - 1), np.random.randint(0, 640 - b - 1))
#         nstep_x = ((480-init_status[0])//b)-1
#         pos = stack_img[init_status[0]:(init_status[0] + b), init_status[1]:(init_status[1] + b)]
#         init_midu = pos.sum()
#     blus_pos = None
#     blus_x = None
#     blus_y = None
#     for step in range(0,nstep_x):
#         x = init_status[0] + (1+step)*b
#         y = init_status[1]
#         pos = stack_img[x:(x + b), y:(y + b)]
#         midu = pos.sum()
#         flag_x = False
#         flag_y_left = False
#         flag_y_right = False
#         if midu < midu_min:
#             blus_pos = pos
#             blus_x = x
#             blus_y = y
#             for step_x in range(step+1,nstep_x):
#                 x = init_status[0] + (1 + step_x) * b
#                 y = init_status[1]
#                 pos = stack_img[x:(x + b), y:(y + b)]
#                 midu = pos.sum()
#                 if midu>=midu_min:
#                     flag_x = True
#                     break
#     nstep_y_right = ((640 - init_status[1]) // b) - 1
#     nstep_y_left = (init_status[1] // b) - 1
#     if not flag_x:
#         return stack_img
#     for step_y_left in (0,nstep_y_left):
#         x = blus_x
#         y = blus_y -  (1+step_y_left)*b
#         pos = stack_img[x:(x + b), y:(y + b)]
#         midu = pos.sum()
#         if midu >= midu_min:
#             flag_y_left = True
#             break
#     if not flag_y_left:
#         return stack_img
#     for nstep_y_right in (0,nstep_y_right):
#         x = blus_x
#         y = blus_y +  (1+nstep_y_right)*b
#         pos = stack_img[x:(x + b), y:(y + b)]
#         midu = pos.sum()
#         if midu >= midu_min:
#             flag_y_right = True
#             break
#     if flag_x and flag_y_right and flag_y_left:
#         stack_img[blus_x:(blus_x+b),blus_y:(blus_y+b)] = 1.0
#     return stack_img
#
#
# def blush3(stack_img):##(480, 640, 3)
#
#     # stack_img = stack_img2.copy()
#     b= int(0.07*1000)#0.07*480
#     midu_min = b*b*3*0.99
#     init_status = (np.random.randint(0,480-2*b-1) , np.random.randint(0,640-2*b-1))
#     init_pos = stack_img[init_status[0]:(init_status[0]+b),init_status[1]:(init_status[1]+b)]
#     init_midu = init_pos.sum()
#     del init_pos
#     # print(init_midu)
#     # print(midu_min)
#
#     if init_midu>midu_min:
#         return stack_img
#     #up
#     # b = int(0.07 * 680)#0.07*480
#     # haf_b = int(b/2.0)
#     if init_status[0]- b-1<=0:
#         up_dis = 0
#     else:
#         up_dis = np.random.randint(0,init_status[0]- b-1)
#     x = up_dis
#     y = init_status[1]
#     up_pos = stack_img[x:(x + b), y:(y + b)]
#     up_midu = up_pos.sum()
#     del up_pos
#     # down
#     # b = int(0.07 * 480 * 5)
#     # haf_b = int(b / 2.0)
#     down_dis = np.random.randint(init_status[0]  +  b -1 , 480 - b-1)
#     x = down_dis
#     y = init_status[1]
#     down_pos = stack_img[x:(x + b), y:(y + b)]
#     down_midu = down_pos.sum()
#
#     # left
#     # b = int(0.07 * 480 * 5)
#     # haf_b = int(b / 2.0)
#     if init_status[1] - b - 1 <= 0:
#         left_dis = 0
#     else:
#         left_dis = np.random.randint(0, init_status[1] - b - 1)
#     y = left_dis
#     x = init_status[0]
#     left_pos = stack_img[x:(x+b), y:(y + b)]
#     left_midu = left_pos.sum()
#     del left_pos
#     # right
#     # b = int(0.07 * 480 * 5)
#     # haf_b = int(b / 2.0)
#
#     right_dis = np.random.randint(init_status[1] +b - 1, 640 - b - 1)
#     y = right_dis
#     x = init_status[0]
#     right_pos = stack_img[x:(x + b), y:(y + b)]
#     right_midu = right_pos.sum()
#     del right_pos
#
#     if up_midu>init_midu and down_midu>init_midu and left_midu>init_midu and right_midu>init_midu:
#         stack_img[init_status[0]:(init_status[0] + b), init_status[1]:(init_status[1] + b)] = True
#
#     return stack_img
#
# def blush2(stack_img):##(480, 640, 3)
#     b= int(0.07*480)
#     midu_min = b*b*0.9*0.9
#     init_status = (np.random.randint(0,480-b-1) , np.random.randint(0,640-b-1))
#     init_pos = stack_img[init_status[0]:(init_status[0]+b),init_status[1]:(init_status[1]+b)]
#     init_midu = init_pos.sum()
#
#     #up
#     b = int(0.07 * 480*5)
#     haf_b = int(b/2.0)
#     x = max(0,init_status[0]- b - 10)
#     y = max(0,init_status[1]-haf_b)
#     up_pos = stack_img[x:(x + b), y:(y + b)]
#     up_midu = up_pos.sum()
#
#     # down
#     # b = int(0.07 * 480 * 5)
#     # haf_b = int(b / 2.0)
#     x = init_status[0]  +  10
#     y = max(0,init_status[1] - haf_b)
#     down_pos = stack_img[x:(x + b), y:(y + b)]
#     down_midu = down_pos.sum()
#
#     # left
#     # b = int(0.07 * 480 * 5)
#     # haf_b = int(b / 2.0)
#     y = max(0,init_status[1] - b - 10)
#     x = max(0,init_status[0]-haf_b)
#     left_pos = stack_img[x:(x+b), y:(y + b)]
#     left_midu = left_pos.sum()
#
#     # right
#     # b = int(0.07 * 480 * 5)
#     # haf_b = int(b / 2.0)
#     y = init_status[1]  + 10
#     x = max(0,init_status[0] - haf_b)
#     right_pos = stack_img[x:(x + b), y:(y + b)]
#     right_midu = right_pos.sum()
#     if up_midu>init_midu and down_midu>init_midu and left_midu>init_midu and right_midu>init_midu:
#         stack_img[init_status[0]:(init_status[0] + b), init_status[1]:(init_status[1] + b)] = True
#     return stack_img
#
#
# def numpy2cv_stack(shape=(10,10)): ## shape = cvimg[0:2]
#     img = np.zeros(shape)
#     img = np.float32(img)
#     img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
#     return img
#
#
# def paste2blank(img,boxs=None,img_blank=None,outpath=None): #box (y1,x2,y2,x1)
#     img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)  ## pil2cv
#     # img = cv2.imread(img)
#     size = img.shape
#     img_blank = cv2.imread(img_blank)
#     img_blank = img_blank.copy()
#     img_blank = cv2.resize(img_blank,(size[1],size[0]))
#     for i in boxs:
#         y1=i[0]
#         x2=i[1]
#         y2=i[2]
#         x1=i[3]
#         img_blank[y1:y2,x1:x2]=img[y1:y2,x1:x2]
#     # cv2.imshow('blank',img_blank)
#     # cv2.waitKey(0)
#     # cv2.imwrite('blank_x.png',img_blank)
#     if outpath:
#         cv2.imwrite(outpath,img_blank)
#     return img_blank
#
# def hilight_img_box(img,boxs=[(100,300,200,0)],outpath=None): #box (y1,x2,y2,x1) ## y1,x1,h,w
#     img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR) # pil2cv
#     # img = cv2.imread(img)
#     size = img.shape[0:2]
#     ones_img = numpy2cv(size)
#     stack_img = numpy2cv_stack(size)
#     assert ones_img.shape==img.shape
#     # img_copy = img.copy()
#     # boxs = boxs[0:1]
#     for i in boxs:
#         y1=i[0]
#         x1=i[1]
#         y2=y1+i[2]
#         x2=x1+i[3]
#         ones_img[y1:y2,x1:x2]=1.0
#         stack_img[y1:y2,x1:x2] = stack_img[y1:y2,x1:x2] + 1.0
#     # img = img*ones_img
#     stack_img = stack_img>3
#     # stack_img = cv2.boxFilter(stack_img, -1, (7,7), normalize=1)
#     for _ in range(50):
#         stack_img = blush3(stack_img)
#     # img = img*(stack_img>3)
#     img = img * stack_img
#     if outpath:
#         cv2.imwrite(outpath,img)
#     del ones_img
#     return img

def transform_image_torch(img, nor_mean, nor_std):
    detect_transform = transforms.Compose([
        # (h, w)
        transforms.Resize((360, 640)),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(nor_mean, nor_std)
    ])

    return detect_transform(img).unsqueeze(0)


def get_input(img_url, rgb_form):
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
        img_url_src = img_url['img_src']
        img_url_dest = img_url['img_dest']
        resp_img_src = request.urlopen(img_url_src, timeout=8)
        resp_img_dest = request.urlopen(img_url_dest, timeout=8)
    except Exception as e:
        return None,None,0,0
        # return 'parse image error.'
    else:
        down_image_take_time = time.time() - down_image_start_time
        transform_image_start_time = time.time()
        img_src = np.asarray(bytearray(resp_img_src.read()), dtype='uint8')
        img_dest = np.asarray(bytearray(resp_img_dest.read()), dtype='uint8')
        # numpy array to opencv image
        img_src = cv2.imdecode(img_src, cv2.IMREAD_UNCHANGED)
        img_dest = cv2.imdecode(img_dest, cv2.IMREAD_UNCHANGED)
        # img = cv2.resize(img, (640, 360))
        # Opencv image  to PIL image
        if rgb_form:
            img_src = Image.fromarray(cv2.cvtColor(img_src, cv2.COLOR_BGR2RGB)).convert('RGB')
            img_dest = Image.fromarray(cv2.cvtColor(img_dest, cv2.COLOR_BGR2RGB)).convert('RGB')
            pil_img = Image.fromarray(np.uint8(np.concatenate((np.array(img_src), np.array(img_dest)), axis=0)))
        else:
            img_src = Image.fromarray(cv2.cvtColor(img_src, cv2.COLOR_BGR2GRAY)).convert('L')
            img_dest = Image.fromarray(cv2.cvtColor(img_dest, cv2.COLOR_BGR2GRAY)).convert('L')
            pil_img = Image.fromarray(np.uint8(np.concatenate((np.array(img_src), np.array(img_dest)), axis=0)))

        # 调用transform_image函数,得到输入模型中的格式，tensor类型
        # img_tensor = transform_image_torch(img, nor_mean, nor_std)
        transform_image_take_time = time.time() - transform_image_start_time

        return pil_img, down_image_take_time, transform_image_take_time


def get_prediction(img_url, model,  rgb_form=True):
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
    pil_img,down_image_take_time, transform_image_take_time = get_input(img_url=img_url,
                                                                            rgb_form=rgb_form)

    # model = request.environ['HTTP_FLASK_MODEL']
    if not pil_img:
        return None,None,0,DOWNLOAD_FAIL
    result = model(pil_img)
    if not result:
        return result, down_image_take_time, PREDICT_FAIL
    else:
        return result, down_image_take_time, SUCCESS

