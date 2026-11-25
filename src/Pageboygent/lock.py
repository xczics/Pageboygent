import os
import fcntl
from contextlib import contextmanager
from pathlib import Path

lock_dir = Path(__file__).parent / ".locks"
lock_dir.mkdir(exist_ok=True)

@contextmanager
def lock_it(lock_name: str):
    LOCK_FILE = lock_dir / f"{lock_name}.lock"

    fd = os.open(LOCK_FILE, os.O_CREAT | os.O_RDWR, 0o644)

    with os.fdopen(fd, "r+") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            f.seek(0)
            yield
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)