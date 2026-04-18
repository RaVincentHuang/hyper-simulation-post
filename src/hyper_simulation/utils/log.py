import logging
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler
from tqdm import tqdm
from contextvars import ContextVar
current_task: ContextVar[str] = ContextVar("task", default="hotpotqa")
current_query_id: ContextVar[str] = ContextVar("query_id", default="")
class TqdmLoggingHandler(logging.StreamHandler):
    def __init__(self, level=logging.NOTSET):
        super().__init__(sys.stderr)
        self.setLevel(level)
    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg, file=sys.stderr)
            self.flush()
        except Exception:
            self.handleError(record)
def getLogger(name: str, level: str = "INFO", log_dir: str = "logs") -> logging.Logger:
    level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR
    }
    log_level = level_map.get(level.upper(), logging.INFO)
    qid = current_query_id.get()
    task = current_task.get()
    log_path = Path(log_dir) / task
    if qid:
        log_path = log_path / qid
    log_path.mkdir(exist_ok=True, parents=True)
    formatter = logging.Formatter(
        fmt='%(message)s',
    )
    console = TqdmLoggingHandler()
    console.setLevel(logging.ERROR)
    console.setFormatter(formatter)
    file_handler = RotatingFileHandler(
        filename=log_path / f"{name}.log",
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8"
    )
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    if logger.hasHandlers():
        logger.handlers.clear()
    logger.addHandler(console)
    logger.propagate = False
    return logger