"""
author: gm.li
"""
import json
# coding = gbk
import time
from datetime import datetime, date
from functools import wraps

import numpy as np
import torch
# from chinese_calendar import is_holiday
from flask import Flask, jsonify
from flask import request
# from joblib import load

from log import *
from utils import get_prediction  #,load_mode
from build_model import predictor
DOWNLOAD_FAIL=401
PREDICT_FAIL=402
SUCCESS=200
PARAM_WRONG=301
app = Flask(__name__)

# -------------------Load Model Into Global---------------------
# 订单异常图片识别模型
# abnormal_picture_classified_service_model = load_mode(service_name='abnormal_picture')
predict_model = predictor(weight="image_model/yolov6n_jit.pth")

def save_log(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        request_time = time.time()
        request_time_str = datetime.strftime(datetime.now(cst_tz), '%Y-%m-%d %H:%M:%S.%f')[:-3]
        request_data = request.json

        result = func(*args, **kwargs)

        response_time = time.time()
        response_time_str = datetime.strftime(datetime.now(cst_tz), '%Y-%m-%d %H:%M:%S.%f')[:-3]
        take_time = (response_time - request_time) * 1000

        logger.info('> request-time: %s\n' % request_time_str +
                    '> url: %s\n' % request.url +
                    '> http-method: %s\n' % request.method +
                    '> content-type: %s\n' % request.headers['Content-Type'] +
                    '> content-length: %s\n' % request.headers['Content-Length'] +
                    '> host: %s\n' % request.headers['Host'] +
                    '> user-agent: %s\n' % request.headers['User-Agent'] +
                    # '> x-forwarded-for: %s\n' % request.headers['X-Forwarded-For'] +
                    '> x-forwarded-for: %s\n' % '-' +
                    '> extra-param: {}\n' +
                    '> body-param: {}\n'.format(request_data) +
                    '\n< response-time: %s\n' % response_time_str +
                    '< http-code: %s\n' % '200' +
                    '< content-type: application/json\n' +
                    '< take-time: {:.2f}\n'.format(take_time) +
                    '< response-data: {}\n'.format(result) +
                    '< business-data: -\n< sw-tid: -')
        return result

    return wrapper



# 图片识别接口
@app.route('/afl-fiber-ai-detect', methods=['POST'])
@save_log
def fiber_predict():

    if request.method == 'POST':
        # file = request.files[file]
        # img_bytes = file.read()
        # 请求参数
        img_url = request.json.get('data',None)
        if '||' not in img_url:
            out = {'result': None, 'status': PARAM_WRONG, 'desc': 'params is wrong'}
            out_json = jsonify(out)
            return out_json
        if img_url:
            sps = img_url.split('||')
            assert len(sps)==2
            img_url = {'img_src':sps[0].strip(),'img_dest':sps[1].strip()}
            # 将bytes格式解码为string格式
            # img_url = str(img_bytes, encoding='utf-8')

            # 运行模型得到结果
            results, down_image_time, code = get_prediction(img_url=img_url,
                                                                        model=predict_model,
                                                                        rgb_form=True)
            if code == DOWNLOAD_FAIL:
                out = {'result': None,'status':DOWNLOAD_FAIL,'desc':'download pic timeout'}
                out_json = jsonify(out)
                return out_json
            elif code == PREDICT_FAIL:
                out = {'result': None, 'status': PREDICT_FAIL, 'desc': 'cannot predict picture'}
                out_json = jsonify(out)
                return out_json
            else:
                out={'result': results,'status':SUCCESS,'desc':'OK'}
                out_json = jsonify(out)
                return out_json
        else:
            out = {'result': None,'status':PARAM_WRONG,'desc':'params is wrong'}
            out_json = jsonify(out)
            return out_json

    else:
        # request method not POST
        return '< Error! request method not POST.'

# 健康检查
@app.route('/afl-fiber-ai-detect/health')
def health_check():
    return 'hello'


if __name__ != '__main__':
     logger = get_logger(app=app)

if __name__ == '__main__':
    logger = get_logger(app=app)
    app.run(host='0.0.0.0', port='8080', debug=False)
