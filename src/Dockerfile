FROM ansible/centos7-ansible:latest as compile

ENV LANG en_US.UTF-8
#RUN yum install -y wget
#RUN wget -O /etc/yum.repos.d/CentOS-Base.repo http://mirrors.aliyun.com/repo/Centos-7.repo
#COPY apps/CentOS-Base.repo /etc/yum.repos.d/

RUN yum clean all
RUN yum makecache 
RUN yum upgrade -y && \
    yum install -y wget git vim zlib-devel bzip2-devel openssl-devel ncurses-devel sqlite-devel readline-devel tk-devel gcc make  libffi-devel xz-devel python-backports-lzma python-tools sudo lrzsz

RUN yum install -y  python3-devel
RUN pip3 install --upgrade pip -i https://pypi.douban.com/simple
RUN yum install -y libjpeg libjpeg-devel zlib zlib-devel freetype freetype-devel lcms lcms-devel
RUN yum install -y python-imaging    
RUN yum install initscripts -y
RUN yum install openssh-server -y
RUN yum install gcc-c++ -y


RUN yum install mesa-libGL.x86_64 -y
RUN yum install mesa-libGL -y


## config pip
RUN mkdir -p ~/.pip
RUN touch ~/.pip/pip.conf
RUN echo "[global]">>~/.pip/pip.conf 
RUN echo "index-url = http://mirrors.aliyun.com/pypi/simple/">>~/.pip/pip.conf 
RUN echo "[install]">>~/.pip/pip.conf 
RUN echo "trusted-host = mirrors.aliyun.com">>~/.pip/pip.conf 


COPY  . /app/
WORKDIR /app/
RUN python3.6 -m pip install pip install -r requirements.txt -i  https://pypi.douban.com/simple
CMD ["python3", "app.py"]


