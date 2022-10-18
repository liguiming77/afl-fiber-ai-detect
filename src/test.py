# # -*- coding: utf-8 -*-
# from torchvision import  transforms,models
# from torch.utils.data import Dataset, DataLoader
# import torch
# from torch import nn
# import warnings
# import os
# from PIL import Image
# from label2name import result
# import numpy as np
# import cv2
# from utils import CustomRandomResizedCrop,hilight_img_box
#
# input_size = 224#380
# newsize=(640,480) # w,h
# min_area = 400
# random_pice_num=10#500
#
#
# val_transforms = transforms.Compose([
#         transforms.Resize(input_size),
#         transforms.CenterCrop(input_size),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#     ])
# from torchvision.transforms import RandomResizedCrop
# crop_transform = CustomRandomResizedCrop(size=(newsize[1],newsize[0]),scale=(0.002,0.15),ratio=(0.1,10) )
# img = Image.open('1.png').convert('RGB')
# a = CustomRandomResizedCrop.forward(crop_transform,img=img)
# print(a)
#
# def default_loader(img_pil):
#     # img_pil =  Image.open(path).convert("RGB")##
#     # img_pil = img_pil.resize(newsize)
#     # img_tensor = preprocess(img_pil)
#     return img_pil.resize(newsize)
#
#
# def gene_anchor():
#     while True:
#         xslice = int(newsize[0]/20)
#         yslice = int(newsize[1]/20)
#
#         xsets =  random.randint(xslice,newsize[0]-xslice,size=2)
#         ysets = random.randint(yslice, newsize[1]-yslice, size=2)
#         x1 = min(xsets)
#         x2 = max(xsets)
#         y1 = min(ysets)
#         y2 = max(ysets)
#         if (x2-x1)*(y2-y1)>min_area:
#             return [x1,y1,x2,y2]
#
# class cropset(Dataset):
#     def __init__(self, img_src = None,img_dst = None,crop_transform = None,loader=default_loader,transform=None,times=500):
#         #定义好 image 的路径
#         self.img_src = img_src
#         self.img_dst = img_dst
#         self.crop_transform = crop_transform
#         self.loader=loader
#         self.transform = transform
#         self.times = times
#
#     def __getitem__(self, index):## same is 0 else 1
#         img_src = self.loader(self.img_src)
#         img_dst = self.loader(self.img_dst)
#         img_crop,zone =  self.crop_transform(img_dst) #zone [i, j, h, w] ## y1,x1,h,w
#         img = Image.fromarray(np.uint8(np.concatenate((np.array(img_src), np.array(img_crop)), axis=0)))
#         img = self.transform(img)
#         del img_src
#         del img_dst
#         del img_crop
#         return img,zone  #zone [i, j, h, w] ## y1,x1,h,w
#
#     def __len__(self):
#         return self.times
#
# def pre_process(img1,img2,zone_num=random_pice_num):
#     # assert img1 is not None
#     # assert img2 is not None
#     # assert crop_transform is not None
#     # assert val_transforms is not None
#     # assert zone_num is not  None
#     dataset=cropset(img_src = img1,img_dst = img2,crop_transform=crop_transform,transform=val_transforms,times=zone_num)
#     dataloader = torch.utils.data.DataLoader(dataset, batch_size=zone_num, shuffle=False, num_workers=12)
#     dataloader = iter(dataloader)
#     (imgs, zones) = next(dataloader)
#     print(zones[0])
#
#     return imgs, zones
#
# def predict(img1,img2,thresh=0.5,outpath=None): # pil outpath='6vs5.png'
#     img, zone = pre_process(img1,img2)
#     with torch.no_grad():
#         img = img.to(device)
#         zone = zone.to(device)
#         outputs = net(img)
#         outputs = outputs[:,0] ##取出每行第一列的值，不相似的概率
#         outputs = outputs.lt(thresh) # 小于阈值的认为是不相似的
#         idxs = torch.where(outputs) ## 不相似的索引
#         zone = zone[idxs].cpu()     ## 获取不相似索引对应的bonding box
#         from numpy import random                                             # new tag
#         outpath = outpath if outpath else str(random.randint(0,1000))+'.png' # new tag
#         hilight_img_box(img2,zone,outpath)
#     return outpath
#
# def sortby(elem):
#     return elem['prob']
# def predict_bac(img,thresh=0.25): # pil
#     with torch.no_grad():
#         img = val_transforms(img).unsqueeze(0)
#         img = img.to(device)
#         outputs = net(img)
#         # outputs = nn.Softmax(dim=1)(outputs)
#         idxs = [idx for idx, w in enumerate(outputs[0].gt(thresh).cpu()) if w]
#         probs = [str(round(v*100.0,2))+'%' for v in outputs[0][idxs].cpu().numpy()]
#         if idxs:
#             names = [tag2name.get(id, None) for id in idxs]
#             result = [{'id':name2id.get(n,''),'name':n,'prob':p} for n, p in zip(names, probs)]
#             result.sort(key=sortby,reverse=True)
#         else:
#             result = {}
#         return result
#
#
#
# import cv2
# #读取图片，该图在此代码的同级目录下
# bgr_img = cv2.imread("376.jpg")
# #二化值
# gray_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
# th, binary = cv2.threshold(gray_img, 0, 255, cv2.THRESH_OTSU)
# #获取轮廓的点集
# binary,contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#
# #取最大的边缘轮廓点集
# contours = max(contours, key=cv2.contourArea)
#
# #求取轮廓的矩
# M = cv2.moments(contours)
#
# #画出轮廓
# cv2.drawContours(bgr_img, contours, -1, (0, 0, 255), 3)
# bounding_boxes = [cv2.boundingRect(cnt) for cnt in contours]
#
# #在图片上画出矩形边框
# for bbox in bounding_boxes:
#     [x, y, w, h] = bbox
#     cv2.rectangle(bgr_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
# #通过矩来计算轮廓的中心坐标
# cx = int(M['m10']/M['m00'])
# cy = int(M['m01']/M['m00'])
# print(f"({cx},{cy})")
# cv2.imshow("name", bgr_img)
# cv2.waitKey(0)
# import cv2
# import numpy as np
#
# def empty(a):
#     pass
#
# def stackImages(scale,imgArray):
#     rows = len(imgArray)
#     cols = len(imgArray[0])
#     rowsAvailable = isinstance(imgArray[0], list) # 判断是否是列表形式，若是则回True
#     width = imgArray[0][0].shape[1]
#     height = imgArray[0][0].shape[0]
#     if rowsAvailable:
#         for x in range(0, rows):
#             for y in range(0, cols):
#                 if imgArray[x][y].shape[:2] == imgArray[0][0].shape[:2]:
#                     imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
#                 else:
#                     imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
#                 if len(imgArray[x][y].shape) == 2: imgArray[x][y] = cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
#         imageBlank = np.zeros((height, width, 3), np.uint8)
#         hor = [imageBlank]*rows
#         hor_con = [imageBlank]*rows
#         for x in range(0, rows):
#             hor[x] = np.hstack(imgArray[x])
#         ver = np.vstack(hor)
#     else:
#         for x in range(0, rows):
#             if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
#                 imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
#             else:
#                 imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None,scale, scale)
#             if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
#         hor= np.hstack(imgArray)
#         ver = hor	#返回的是图像，只需要调用imshow()函数正常显示即可。
#     return ver
#
# path = "376.jpg"
# cv2.namedWindow("TrackBars")
# cv2.resizeWindow("TrackBars",640,340)
# # 创建跟踪栏
# cv2.createTrackbar("Hue Min","TrackBars", 0, 179, empty)   #色调
# cv2.createTrackbar("Hue Max","TrackBars", 179, 179, empty)
# cv2.createTrackbar("Sat Min","TrackBars", 0, 255, empty)   #饱和度
# cv2.createTrackbar("Sat Max","TrackBars", 255, 255, empty)
# cv2.createTrackbar("Val Min","TrackBars", 0, 255, empty)  #亮度,用HSV空间更能体现人眼对不同颜色的差别
# cv2.createTrackbar("Val Max","TrackBars", 255, 255, empty)
#
# if True:
#     img = cv2.imread(path)
#     imgHSV = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)        # 转换为HSV空间
#     h_min = cv2.getTrackbarPos("Hue Min", "TrackBars")
#     h_max = cv2.getTrackbarPos("Hue Max", "TrackBars")
#     s_min = cv2.getTrackbarPos("Sat Min", "TrackBars")
#     s_max = cv2.getTrackbarPos("Sat Max", "TrackBars")
#     v_min = cv2.getTrackbarPos("Val Min", "TrackBars")
#     v_max = cv2.getTrackbarPos("Val Max", "TrackBars")
#     print(h_min,h_max,s_min,s_max,v_min,v_max)
#
#     lower = np.array([h_min,s_min,v_min])
#     upper = np.array([h_max,s_max,v_max])
#     mask = cv2.inRange(imgHSV,lower,upper)
#     imgResult = cv2.bitwise_and(img,img,mask=mask)
#     imgStacked = stackImages(0.6, ([img, imgHSV], [mask, imgResult]))
#     cv2.imwrite('111.jpg',imgStacked)
#     cv2.imshow("Stacked images", imgStacked)
#     cv2.waitKey(1)
#
# import cv2
# import numpy as np
#
#
# # 导入相应的模块
#
# class ContourFilter(object):
#     def __init__(self):
#         super(ContourFilter, self).__init__()
#         # 输入参数区
#         self.threshold = lambda image: cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#                                                              cv2.THRESH_BINARY_INV, 15, 21)
#         # 二值化算法，这里使用的是 自适应二值化，也可以使用Canny边缘检测算法 等。
#         self.areaRanges = []
#         # 轮廓面积范围
#         # 存储格式为 [(minArea1,maxArea1),(minArea2,maxArea2),...]
#         self.perimeterRanges = []
#         # 轮廓周长范围
#         # 存储格式同上
#         self.contourColor = (255, 127, 127)
#         # 轮廓的绘制颜色
#         self.contourThickness = 3
#         # 轮廓的绘制粗细
#         self.inPlace = False
#         # 是否处理后显示在原图上
#         self.paint = True
#
#     # 是否进行绘制
#
#     def __call__(self, src: np.ndarray) -> np.ndarray:
#         # 必要的函数注释
#         if not self.areaRanges:
#             self.areaRanges = [(0, float('inf'))]
#         if not self.perimeterRanges:
#             self.perimeterRanges = [(0, float('inf'))]
#         # 如果周长和面积范围未赋值(None)，那么默认为(0,+∞)
#
#         if self.inPlace:
#             dest = src.copy()
#         else:
#             dest = np.zeros_like(src)
#         # 处理显示在 原图拷贝上 或者 在空图像上
#
#         if len(src.shape) == 3 and src.shape[2] == 3:
#             gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
#         elif len(src.shape) == 2:
#             gray = src
#         # 转灰度图处理，如果本身就是单通道，那么不进行转换
#
#         binary = self.threshold(gray)
#         # 二值化处理
#
#         contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#         # 轮廓提取，这里提取的是外轮廓且忽略轮廓层次信息
#         resultContours = []
#         # 轮廓的筛选结果列表
#         for contour in contours:
#             # 对每一轮廓进行遍历
#             perimeter = cv2.arcLength(contour, True)
#             # 计算轮廓长度
#             area = cv2.contourArea(contour)
#             # 计算轮廓面积
#
#             for perimeterRange, areaRange in zip(self.perimeterRanges, self.areaRanges):
#                 if perimeterRange[0] < perimeter <= perimeterRange[1] and areaRange[0] < area <= areaRange[1]:
#                     resultContours.append(contour)
#                 # 记录符合筛选条件的轮廓
#         if self.paint:
#             cv2.drawContours(dest, resultContours, -1, self.contourColor, self.contourThickness)
#         # 绘制轮廓
#
#         return dest
#
#
# # 调用测试
# if __name__ == '__main__':
#     src = cv2.imread('376.jpg')  # 图片路径
#     contourFilter = ContourFilter()  # 初始化ContourFilter对象
#     contourFilter.areaRanges.append((minArea, maxArea))
#     contourFilter.perimeterRanges.append((minPerimeter, maxPerimeter))
#     # 对轮廓的面积和周长进行条件限制
#     dest = colorFilter(src)
#     cv2.imshow('dest', dest)
#     cv2.waitKey(0)

# import stack
#
# # -*- coding=utf-8 -*-
# import cv2 as cv
# import numpy as np
#
#
# def showImg(img_name, img):
#     cv.imshow(img_name, img)
#     cv.waitKey()
#     cv.destroyAllWindows()
#
# # 指定颜色替换
# def fill_image(image):
#     copyImage = image.copy()  # 复制原图像
#     h, w = image.shape[:2]  # 读取图像的宽和高
#     mask = np.zeros([h + 2, w + 2], np.uint8)  # 新建图像矩阵  +2是官方函数要求
#
#     cv.floodFill(copyImage, mask, (430, 240), (0, 100, 255), (100, 100, 50), (50, 50, 50), cv.FLOODFILL_FIXED_RANGE)
#
#     # showImg("copyImage", copyImage)
#     result_mask = np.zeros(shape=copyImage.shape)
#     ids = np.where((copyImage==(0, 100, 255)))
#     result_mask[ids[:2]] = (255, 255, 255)
#     mask = cv.inRange(result_mask, (200, 0, 0), (255, 255, 2555))
#     # result_mask = cv.cvtColor(result_mask, cv.COLOR_BGR2GRAY)
#     # cv.imshow("填充", copyImage)
#     # showImg("copyImage", copyImage)
#     # showImg("result_mask", result_mask)
#     # showImg("image", image)
#     res = cv.bitwise_and(image, image, mask=mask)
#     showImg("res", res)
#
#
#
# src = cv.imread("376.jpg")
# # cv.imshow("原来", src)
# # showImg("src", src)
# fill_image(src)

#
# import cv2
# import numpy as np
#
# img = cv2.imread('376.jpg')
# rows, cols, ch = img.shape
# # 边缘提取
# Ksize = 3
# L2g = True
# edge = cv2.Canny(img, 50, 100, apertureSize=Ksize, L2gradient=L2g)
#
# # 提取轮廓
# '''
# findcontour()函数中有三个参数，第一个img是源图像，第二个model是轮廓检索模式，第三个method是轮廓逼近方法。输出等高线contours和层次结构hierarchy。
# model:  cv2.RETR_EXTERNAL  仅检索极端的外部轮廓。 为所有轮廓设置了层次hierarchy[i][2] = hierarchy[i][3]=-1
#         cv2.RETR_LIST  在不建立任何层次关系的情况下检索所有轮廓。
#         cv2.RETR_CCOMP  检索所有轮廓并将其组织为两级层次结构。在顶层，组件具有外部边界；在第二层，有孔的边界。如果所连接零部件的孔内还有其他轮廓，则该轮廓仍将放置在顶层。
#         cv2.RETR_TREE  检索所有轮廓，并重建嵌套轮廓的完整层次。
#         cv2.RETR_FLOODFILL  输入图像也可以是32位的整型图像(CV_32SC1)
# method：cv2.CHAIN_APPROX_NONE  存储所有的轮廓点，任何一个包含一两个点的子序列（不改变顺序索引的连续的）相邻。
#         cv2.CHAIN_APPROX_SIMPLE  压缩水平，垂直和对角线段，仅保留其端点。 例如，一个直立的矩形轮廓编码有4个点。
#         cv2.CHAIN_APPROX_TC89_L1 和 cv2.CHAIN_APPROX_TC89_KCOS 近似算法
# '''
# bin,contours, hierarchy = cv2.findContours(edge, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# # 绘制轮廓 第三个参数是轮廓的索引（在绘制单个轮廓时有用。要绘制所有轮廓，请传递-1）
# dst = np.ones(img.shape, dtype=np.uint8)
# # cv2.drawContours(dst, contours, -1, (0, 255, 0), 1)
# # for i in range(12):
# #     print(len(contours[i]))
# '''
# cv2.ellipse(image, centerCoordinates, axesLength, angle, startAngle, endAngle, color [, thickness[, lineType[, shift]]])
# image:它是要在其上绘制椭圆的图像。
# centerCoordinates:它是椭圆的中心坐标。坐标表示为两个值的元组，即(X坐标值，Y坐标值)。
# axesLength:它包含两个变量的元组，分别包含椭圆的长轴和短轴(长轴长度，短轴长度)。
# angle:椭圆旋转角度，以度为单位。
# startAngle:椭圆弧的起始角度，以度为单位。
# endAngle:椭圆弧的终止角度，以度为单位。
# color:它是要绘制的形状边界线的颜色。对于BGR，我们通过一个元组。例如：(255，0，0)为蓝色。
# thickness:是形状边界线的粗细像素。厚度-1像素将用指定的颜色填充形状。
# lineType:这是一个可选参数，它给出了椭圆边界的类型。
# shift:这是一个可选参数。它表示中心坐标中的小数位数和轴的值。
# '''
# for cnt in contours:
#     if len(cnt)<5:
#         continue
#     # cnt = contours[i]
#
#     ellipse = cv2.fitEllipse(cnt)
#
#     cv2.ellipse(dst, ellipse, (0, 0, 255), 2)
# cv2.imshow("dst", dst)
# cv2.waitKey()
#
# '''
# #print(contours)
# # 绘制单个轮廓
# cnt = contours[3]
# cv2.drawContours(dst, [cnt], 0, (0, 0, 255), 1)
#
# # 特征矩
# cnt = contours[3]
# M = cv2.moments(cnt)
# print(M)
# cx = int(M['m10']/M['m00'])
# cy = int(M['m01']/M['m00'])
# cv2.circle(dst, (cx, cy), 2, (0, 0, 255), -1)   # 绘制圆点
#
# # 轮廓面积
# area = cv2.contourArea(cnt)
# print(area)
#
# # 轮廓周长：第二个参数指定形状是闭合轮廓(True)还是曲线
# perimeter = cv2.arcLength(cnt, True)
# print(perimeter)
#
# # 轮廓近似：epsilon是从轮廓到近似轮廓的最大距离--精度参数
# epsilon = 0.01 * cv2.arcLength(cnt, True)
# approx = cv2.approxPolyDP(cnt, epsilon, True)
# cv2.polylines(dst, [approx], True, (0, 255, 255))   # 绘制多边形
# print(approx)
#
# # 轮廓凸包：returnPoints：默认情况下为True。然后返回凸包的坐标。如果为False，则返回与凸包点相对应的轮廓点的索引。
# hull = cv2.convexHull(cnt, returnPoints=True)
# cv2.polylines(dst, [hull], True, (255, 255, 255), 2)   # 绘制多边形
# print(hull)
#
# # 检查凸度：检查曲线是否凸出的功能，返回True还是False。
# k = cv2.isContourConvex(cnt)
# print(k)
#
# # 边界矩形:最小外接矩形
# # 直角矩形
# x, y, w, h = cv2.boundingRect(cnt)
# cv2.rectangle(dst, (x, y), (x+w, y+h), (255, 255, 0), 2)
# # 旋转矩形
# rect = cv2.minAreaRect(cnt)
# box = cv2.boxPoints(rect)
# box = np.int0(box)
# cv2.drawContours(dst, [box], 0, (0, 0, 255), 2)
#
# # 最小外接圆
# (x, y), radius = cv2.minEnclosingCircle(cnt)
# center = (int(x), int(y))
# radius = int(radius)
# cv2.circle(dst, center, radius, (0, 255, 0), 2)
#
# # 拟合椭圆
# ellipse = cv2.fitEllipse(cnt)
# cv2.ellipse(dst, ellipse, (0, 0, 255), 2)
#
# # 拟合直线
# rows, cols = img.shape[:2]
# [vx, vy, x, y] = cv2.fitLine(cnt, cv2.DIST_L2, 0, 0.01, 0.01)
# lefty = int((-x*vy/vx) + y)
# righty = int(((cols-x)*vy/vx)+y)
# cv2.line(dst, (cols-1, righty), (0, lefty), (255, 255, 255), 2)
#
#
# cv2.imshow("dst", dst)
# cv2.waitKey()'''