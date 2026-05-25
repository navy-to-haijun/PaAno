import logging
import inspect

def init_logger(logfile='run.log', name=None):
    """
    初始化日志配置
    """
    if name is None:
        # 获取调用模块名称
        frame = inspect.stack()[1]
        module = inspect.getmodule(frame[0])
        name = module.__name__ if module else "root"

    # 避免重复初始化 root logger
    logger = logging.getLogger(name)
    if not logger.hasHandlers():
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s-[%(levelname)s] : %(message)s',
            handlers=[
                # logging.FileHandler(logfile, encoding='utf-8'),  # 文件输出
                logging.StreamHandler()  # 控制台输出
            ]
        )
    
    return logger