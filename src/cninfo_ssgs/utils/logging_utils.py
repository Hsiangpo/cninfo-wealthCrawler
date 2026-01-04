from __future__ import annotations

import logging
import sys
from pathlib import Path


class _TqdmCompatibleHandler(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        try:
            from tqdm import tqdm  # type: ignore

            self._tqdm = tqdm
        except Exception:
            self._tqdm = None

    def emit(self, record: logging.LogRecord) -> None:
        msg = self.format(record)
        try:
            if self._tqdm is not None:
                self._tqdm.write(msg, file=sys.stderr)
            else:
                sys.stderr.write(msg + "\n")
        except Exception:
            # 避免日志系统自身异常影响主流程
            pass


def setup_logging(*, level: str = "INFO", log_file: Path | None = None) -> None:
    """
    统一日志输出：
    - 控制台：使用 tqdm.write 避免与进度条冲突
    - 文件（可选）：便于长任务留痕与排查
    """
    lvl = getattr(logging, (level or "INFO").upper(), logging.INFO)
    root = logging.getLogger()
    root.setLevel(lvl)

    # 清理旧 handler，避免重复打印
    for h in list(root.handlers):
        root.removeHandler(h)

    fmt = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] [%(threadName)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console = _TqdmCompatibleHandler()
    console.setLevel(lvl)
    console.setFormatter(fmt)
    root.addHandler(console)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        # Windows 下 PowerShell 的 Get-Content 对“无 BOM 的 UTF-8”常会按 ANSI 解码导致中文乱码；
        # 这里仅在新文件/空文件时写入 UTF-8 BOM，后续追加时保持纯 UTF-8，避免在文件中间插入 BOM。
        encoding = "utf-8"
        try:
            if not log_file.exists() or log_file.stat().st_size == 0:
                encoding = "utf-8-sig"
        except Exception:
            encoding = "utf-8"
        fh = logging.FileHandler(log_file, encoding=encoding)
        fh.setLevel(lvl)
        fh.setFormatter(fmt)
        root.addHandler(fh)

    # 降噪：部分第三方库在 DEBUG 下会输出大量内部细节（例如 pdfminer 的 token 级解析日志），影响可读性。
    # 这里强制把它们降到 WARNING，仅保留真正的异常/告警。
    for noisy in ("pdfminer", "pdfplumber", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
