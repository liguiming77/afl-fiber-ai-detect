# FROM jupyter/datascience-notebook:latest
# FROM registry-intl.cn-hangzhou.aliyuncs.com/markable-cn/markable-cv-service:rq_pool-5c04f84-cuda10.2

FROM nvidia/cuda:10.2-cudnn7-devel-ubuntu16.04 as compile

ENV LANG C.UTF-8
ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
        apt-get install -y software-properties-common && \
        add-apt-repository ppa:deadsnakes/ppa && \
        apt-get update -y  && \
        apt-get install -y build-essential python3.6 python3.6-dev python3-pip python3.6-venv && \
        apt-get install -y --no-install-recommends \
        git \
        curl \
        wget \
        libsm6 \
        libssl-dev \
        libxext6 \
        libxrender-dev \
        ca-certificates \
        libsnappy-dev \
        libjpeg-dev \
        libpng-dev &&\
     rm -rf /var/lib/apt/lists/*

RUN  python3.6 -m pip  install --upgrade pip -i https://pypi.douban.com/simple

RUN mkdir -p ~/.pip
RUN touch ~/.pip/pip.conf
RUN echo "[global]">>~/.pip/pip.conf
RUN echo "index-url = http://mirrors.aliyun.com/pypi/simple/">>~/.pip/pip.conf
RUN echo "[install]">>~/.pip/pip.conf
RUN echo "trusted-host = mirrors.aliyun.com">>~/.pip/pip.conf

RUN  python3.6 -m pip   install wheel 
RUN  python3.6 -m pip  install setuptools


COPY  requirements.txt /mdata/
WORKDIR /mdata/
#RUN python3.6 -m pip install pip install -r requirements.txt -i https://mirrors.cloud.tencent.com/pypi/simple
#RUN python3.6 -m pip install pip install -r requirements.txt -i https://pypi.douban.com/simple
RUN python3.6 -m pip install pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/
#RUN python3.6 -m pip install pip install -r requirements.txt -i https://pypi.python.org/simple

# WORKDIR /

CMD ["python3.6", "main/api/server.py"]
