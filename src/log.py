import logging.handlers
import os
import sys

from logging import Logger
from logging import raiseExceptions


from pytz import timezone

cst_tz = timezone('Asia/Shanghai')

out_dir = './data/log/afl-abnormal-device-detect-service/'
access_log_dir = out_dir + 'request/'
error_log_dir = out_dir + 'error/'
info_log_dir = out_dir + 'info/'
if not os.path.exists(access_log_dir):
    os.makedirs(access_log_dir)
if not os.path.exists(error_log_dir):
    os.makedirs(error_log_dir)
if not os.path.exists(info_log_dir):
    os.makedirs(info_log_dir)


def get_logger(app):
    """
    save log to different file by different log level into the log path
    and print all log in console
    :return:
    """
    logging.setLoggerClass(AppLogger)
    formatter = logging.Formatter(fmt='%(asctime)s [%(threadName)s] %(levelname)s %(name)s -\n%(message)s\n')
    formatter.default_msec_format = '%s.%03d'

    log_files = {
        logging.INFO: os.path.join(access_log_dir, 'request.log'),
        # logging.WARNING: os.path.join(log_path, logfile_name + '-warning.log'),
        logging.ERROR: os.path.join(error_log_dir, 'error.log'),
    }
    # 和flask默认使用同一个logger
    logger = app.logger
    logger.setLevel(logging.INFO)
    for log_level, log_file in log_files.items():
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


class AppLogger(Logger):
    """
    自定义logger
    如果handler名称为console表示在终端打印所有大于等于设置级别的日志
    其他handler则只记录等于设置级别的日志
    """

    def __init__(self, name, level=logging.NOTSET):
        super(AppLogger, self).__init__(name, level)

    def callHandlers(self, record):
        """
        Pass a record to all relevant handlers.

        Loop through all handlers for this logger and its parents in the
        logger hierarchy. If no handler was found, output a one-off error
        message to sys.stderr. Stop searching up the hierarchy whenever a
        logger with the "propagate" attribute set to zero is found - that
        will be the last logger whose handlers are called.
        """
        c = self
        found = 0
        while c:
            for hdlr in c.handlers:
                found = found + 1
                if hdlr.name == 'console':
                    if record.levelno >= hdlr.level:
                        hdlr.handle(record)
                else:
                    if record.levelno == hdlr.level:
                        hdlr.handle(record)
            if not c.propagate:
                c = None  # break out
            else:
                c = c.parent
        if (found == 0) and raiseExceptions and not self.manager.emittedNoHandlerWarning:  # noqa
            sys.stderr.write("No handlers could be found for logger"
                             " \"%s\"\n" % self.name)
            self.manager.emittedNoHandlerWarning = 1
