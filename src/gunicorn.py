# coding=gbk
import multiprocessing
import os

# �˿ں�
bind = '127.0.0.1:8000'

timeout = 30

worker_class = 'gevent'
# ���й���������
# workers = multiprocessing.cpu_count() * 2 +
workers = 9
# ָ��ÿ�������ߵ��߳���
threads = 2

reload = True
debug = False

max_requests = 2000
# ������󲢷���
worker_connections = 2000

loglevel = 'info'

# now_time = datetime.now()
# time_str = datetime.strftime(now_time, '%y-%m-%d')

# access_log_name = 'request.log'
# error_log_name = 'error.log'


#
# accesslog = access_log_dir + access_log_name
# errorlog = error_log_dir + error_log_name
# if not os.path.isfile(accesslog):
#     f = open(accesslog, mode='w', encoding='utf-8')
#     f.close()
# if not os.path.isfile(errorlog):
#     f = open(errorlog, mode='w', encoding='utf-8')
#     f.close()
