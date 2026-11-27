"""进程间文件锁（fcntl.flock）。锁文件放在数据目录（workstation_dir）下的 .locks 里：
- 不放在包目录：装到 site-packages 后包目录通常不可写；
- 每次加锁前都确认目录存在：目录被删掉（例如清理数据目录）也不会让后台任务报错。"""
import fcntl
import os
from contextlib import contextmanager
from pathlib import Path

from .config import get_config


def lock_dir() -> Path:
    path = Path(get_config("workstation_dir")).expanduser() / ".locks"
    path.mkdir(parents=True, exist_ok=True)
    return path


@contextmanager
def lock_it(lock_name: str):
    lock_file = lock_dir() / f"{lock_name}.lock"
    fd = os.open(lock_file, os.O_CREAT | os.O_RDWR, 0o644)
    with os.fdopen(fd, "r+") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            f.seek(0)
            yield
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)
