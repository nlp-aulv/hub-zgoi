import logging # 日志打印模块

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log"),  # 输出到文件
        logging.StreamHandler(),         # 同时输出到控制台
    ]
)

logger = logging.getLogger(__name__)