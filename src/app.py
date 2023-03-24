"""
author: gm.li
"""
import json
# coding = gbk

from flask import Flask, jsonify
from flask import request
# from joblib import load

from log import *
from utils import get_prediction  #,load_mode
from build_model import predictor
from loguru import logger as logging
logging.add('fiber.log')
pic_save_dir = '/datasets'#'/pic_save_dir'
DOWNLOAD_FAIL=401
PREDICT_FAIL=402
SUCCESS=200
PARAM_WRONG=301
app = Flask(__name__)

# -------------------Load Model Into Global---------------------
# 图片识别模型
predict_model = predictor(weight="image_model/yolov6n_jit.pth")


# 图片识别接口
@app.route('/afl-fiber-ai-detect', methods=['POST'])
def fiber_predict():
    if request.method == 'POST':
        # file = request.files[file]
        # img_bytes = file.read()
        # 请求参数
        img_url = request.json.get('data',None)
        img_url = os.path.join(pic_save_dir,img_url)
        # logging.info(img_url)
        if img_url:
            # 运行模型得到结果
            xyxys,confs,name_ens = get_prediction(img_url=img_url,
                                                                        model=predict_model)


            if not xyxys:
                out = {'result': None,'status':"",'desc':''}
                out_json = jsonify(out)
                return out_json
            else:

                results = {'xyxys':xyxys,'confs':confs,'class_names':name_ens }

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
