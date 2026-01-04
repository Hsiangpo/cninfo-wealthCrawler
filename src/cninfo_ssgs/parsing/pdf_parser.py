from __future__ import annotations

import multiprocessing as mp
import os
import threading
from dataclasses import dataclass
from io import BytesIO

import pdfplumber
import logging


_PDF_LOCK = threading.Lock()
_PDFIUM_SAFE = (os.environ.get("CNINFO_PDFIUM_SAFE") or "1").strip().lower() not in ("0", "false", "off", "no")
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PdfContent:
    text: str
    tables: list[list[list[str]]]


def _extract_text_pdfium(pdf_bytes: bytes, *, max_pages: int | None) -> str:
    try:
        import pypdfium2 as pdfium
    except Exception:
        logger.debug("pypdfium2 不可用，跳过第一级解析")
        return ""

    try:
        doc = pdfium.PdfDocument(pdf_bytes)
    except Exception as exc:
        logger.debug("pypdfium2 打开失败：%s", exc)
        return ""

    texts: list[str] = []
    try:
        total = len(doc)
        limit = total if max_pages is None else max(0, min(total, int(max_pages)))
        for i in range(limit):
            try:
                page = doc[i]
                textpage = page.get_textpage()
                text = textpage.get_text_range() or ""
                if text:
                    texts.append(text)
            except Exception:
                continue
    finally:
        try:
            doc.close()
        except Exception:
            pass
    return "\n".join(t for t in texts if t).strip()


def _pdfium_worker(pdf_bytes: bytes, max_pages: int | None, q: mp.Queue) -> None:
    try:
        q.put(_extract_text_pdfium(pdf_bytes, max_pages=max_pages) or "")
    except Exception:
        q.put("")


def _extract_text_pdfium_safe(pdf_bytes: bytes, *, max_pages: int | None, timeout_s: float = 20.0) -> str:
    ctx = mp.get_context("spawn")
    q: mp.Queue = ctx.Queue()
    p = ctx.Process(target=_pdfium_worker, args=(pdf_bytes, max_pages, q), daemon=True)
    p.start()
    p.join(timeout_s)
    if p.is_alive():
        p.terminate()
        p.join()
        logger.warning("pypdfium2 解析超时，已终止子进程")
        return ""
    if p.exitcode not in (0, None):
        logger.warning("pypdfium2 子进程异常退出 exitcode=%s", p.exitcode)
    try:
        if not q.empty():
            return q.get_nowait() or ""
    except Exception:
        return ""
    return ""


def parse_pdf(pdf_bytes: bytes, *, max_pages: int | None = 10, min_text_chars: int = 200) -> PdfContent:
    """
    解析 PDF 文本与表格（尽量轻量，避免引入 OCR/外部服务）。

    max_pages:
      - None: 全部页面
      - int: 最多解析前 N 页（多数“理财公告”核心信息在前几页）
    """
    # 第一级：pypdfium2 极速解析
    if _PDFIUM_SAFE:
        text = _extract_text_pdfium_safe(pdf_bytes, max_pages=max_pages)
    else:
        text = _extract_text_pdfium(pdf_bytes, max_pages=max_pages)
    if len(text or "") >= max(0, int(min_text_chars)):
        return PdfContent(text=text, tables=[])

    # 第二级：pdfplumber 布局还原 + 表格
    with _PDF_LOCK:
        with pdfplumber.open(BytesIO(pdf_bytes)) as pdf:
            pages = pdf.pages if max_pages is None else pdf.pages[: max(0, max_pages)]
            texts: list[str] = []
            tables: list[list[list[str]]] = []
            for page in pages:
                try:
                    texts.append(page.extract_text() or "")
                except Exception:
                    texts.append("")
                try:
                    page_tables = page.extract_tables() or []
                    for t in page_tables:
                        if t:
                            tables.append(t)
                except Exception:
                    continue

    text = "\n".join(t for t in texts if t).strip()
    return PdfContent(text=text, tables=tables)
