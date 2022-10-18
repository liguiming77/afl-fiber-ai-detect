ARG PYTORCH="1.8.0"
ARG CUDA="11.1"
ARG CUDNN="8"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel
ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0+PTX"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"

RUN  sed -i s@/archive.ubuntu.com/@/mirrors.aliyun.com/@g /etc/apt/sources.list
RUN  apt-get clean
RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys  A4B469963BF863CC
RUN apt-get update && apt-get install -y ffmpeg libsm6 libxext6 git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
RUN mkdir -p ~/.pip
RUN touch ~/.pip/pip.conf
RUN echo "[global]">>~/.pip/pip.conf
RUN echo "index-url = http://mirrors.aliyun.com/pypi/simple/">>~/.pip/pip.conf
RUN echo "[install]">>~/.pip/pip.conf
RUN echo "trusted-host = mirrors.aliyun.com">>~/.pip/pip.conf

COPY  . /app/
WORKDIR /app/
RUN python3.6 -m pip install pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/
CMD ["python3.6", "app.py"]
