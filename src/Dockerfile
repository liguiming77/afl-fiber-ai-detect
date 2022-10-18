FROM ansible/centos7-ansible:latest as compile

ENV LANG en_US.UTF-8
#RUN yum install -y wget
#RUN wget -O /etc/yum.repos.d/CentOS-Base.repo http://mirrors.aliyun.com/repo/Centos-7.repo
#COPY apps/CentOS-Base.repo /etc/yum.repos.d/
RUN echo 'nameserver 8.8.8.8'>>/etc/resolv.conf
RUN echo 'nameserver 114.114.114.114'>>/etc/resolv.conf
#RUN wget -O /etc/yum.repos.d/CentOS-Base.repo http://mirrors.cloud.tencent.com/repo/centos7_base.repo

RUN yum clean all
RUN yum makecache 
RUN yum upgrade -y && \
    yum install -y wget git vim zlib-devel bzip2-devel openssl-devel ncurses-devel sqlite-devel readline-devel tk-devel gcc make  libffi-devel xz-devel python-backports-lzma python-tools sudo lrzsz
#ADD https://npm.taobao.org/mirrors/python/3.8.5/Python-3.8.5.tar.xz  /tmp
#RUN wget -O - https://npm.taobao.org/mirrors/python/3.6.5/Python-3.6.5.tar.xz | tar -C /tmp/ -xvJf  -
#RUN cd /tmp/Python-3.6.5 && ./configure prefix=/usr/local/python3 && make && make install && ln -s /usr/local/python3/bin/python3 /usr/bin/python3 && ln -s /usr/local/python3/bin/pip3 /usr/bin/pip3 

RUN yum install -y  python3-devel
RUN pip3 install --upgrade pip -i https://pypi.douban.com/simple
RUN yum install -y libjpeg libjpeg-devel zlib zlib-devel freetype freetype-devel lcms lcms-devel
RUN yum install -y python-imaging    
RUN yum install initscripts -y
RUN yum install openssh-server -y
RUN yum install gcc-c++ -y
RUN echo 'PubkeyAuthentication yes'>>/etc/ssh/sshd_config    
RUN echo 'PermitRootLogin yes'>>/etc/ssh/sshd_config
RUN echo "root ALL=(ALL)   ALL">>/etc/sudoers

RUN yum install mesa-libGL.x86_64 -y
RUN yum install mesa-libGL -y

##  install gcc > 4.8
#RUN yum install centos-release-scl-rh -y
#RUN yum install devtoolset-7-gcc devtoolset-7-gcc-c++  -y
#RUN "source /opt/rh/devtoolset-7/enable">>/etc/profile 

#RUN rm /usr/bin/gcc
#RUN ln -s /opt/rh/devtoolset-7/root/usr/bin/gcc /usr/bin/gcc



## config pip
RUN mkdir -p ~/.pip
RUN touch ~/.pip/pip.conf
RUN echo "[global]">>~/.pip/pip.conf 
RUN echo "index-url = http://mirrors.aliyun.com/pypi/simple/">>~/.pip/pip.conf 
RUN echo "[install]">>~/.pip/pip.conf 
RUN echo "trusted-host = mirrors.aliyun.com">>~/.pip/pip.conf 



## install requirements.txt
#RUN pip3 install numpy
#COPY apps/requirements.txt  /usr/local/src/
#WORKDIR /usr/local/src/
#RUN pip3 install -r requirements.txt 


## install cocoapi
#RUN pip3 install numpy
#RUN pip3 install matplotlib==2.1.0
#ADD apps/cocoapi.tar.gz /usr/local/src/
#WORKDIR /usr/local/src/
#WORKDIR /usr/local/src/cocoapi/PythonAPI
#RUN python3 setup.py install


## install cmake
#ADD apps/cmake-3.8.0.tar.gz /usr/local/src/
#WORKDIR /usr/local/src/cmake-3.8.0/
#RUN ./bootstrap
#RUN gmake
#RUN make install

## copy lib
#COPY apps/libpython3.6m.so.1.0 /usr/lib64/

COPY  . /app/
WORKDIR /app/
RUN python3.6 -m pip install pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/
CMD ["python3", "app.py"]


