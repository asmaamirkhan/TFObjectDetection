import logging
import coloredlogs
import sys

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logger = logging.getLogger(__name__)
log_format = "%(asctime)s  %(levelname)s %(message)s"
coloredlogs.install(level='DEBUG', fmt=log_format)