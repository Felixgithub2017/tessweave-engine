import json
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
import threading
import time


class Trace:
    """Bounded rotating JSONL audit; prompts, text and raw token IDs are omitted."""
    def __init__(self, path=None):
        self.lock = threading.Lock()
        self.handler = None
        if path:
            p = Path(path)
            p.parent.mkdir(parents=True, exist_ok=True)
            self.handler = RotatingFileHandler(p, maxBytes=20 * 1024 * 1024, backupCount=4, encoding="utf-8")
            self.handler.setFormatter(logging.Formatter("%(message)s"))

    def emit(self, kind, **data):
        if self.handler:
            row = json.dumps({"time": time.time(), "monotonic": time.monotonic(), "event": kind, **data}, allow_nan=False)
            with self.lock:
                self.handler.emit(logging.LogRecord("flow", logging.INFO, "", 0, row, (), None))

    def close(self):
        if self.handler:
            self.handler.close()
