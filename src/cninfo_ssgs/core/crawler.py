from __future__ import annotations

import csv
import dataclasses
import datetime as dt
import json
import logging
import random
import re
import threading
import time
from collections import deque
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

import requests
from tqdm import tqdm

from ..integrations.cninfo_api import (
    Announcement,
    StockInfo,
    bootstrap_session,
    create_cninfo_client,
    download_pdf_bytes,
    fetch_stock_list_auto,
    iter_his_announcements,
)
from .extract_wealth_mgmt import PurchaseRecord, extract_records, is_candidate_announcement
from ..integrations.http_client import HttpClient, RateLimiter
from ..llm.llm_wealth_mgmt import WealthMgmtLlmExtractor, load_llm_config_from_env
from ..parsing.pdf_parser import parse_pdf


_CODE_RE = re.compile(r"^\d{6}$")
_TAG_RE = re.compile(r"<[^>]+>")
logger = logging.getLogger(__name__)


def _clean_title(title: str) -> str:
    return _TAG_RE.sub("", title or "").strip()


def _load_processed_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def _append_processed_id(path: Path, ann_id: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="") as f:
        f.write(f"{ann_id}\n")


def _ensure_csv_header(path: Path, fieldnames: list[str], *, encoding: str = "utf-8") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 0:
        return
    with path.open("w", encoding=encoding, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()


def _append_csv_rows(path: Path, fieldnames: list[str], rows: Iterable[dict]) -> None:
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        for r in rows:
            writer.writerow(r)


def _iter_date_chunks(start_date: str, end_date: str, chunk: str) -> Iterable[tuple[str, str]]:
    """
    将较长日期区间切分为更小的块，降低 cninfo 搜索结果过大导致的潜在漏召回风险。

    chunk:
      - none：不切分
      - month：按自然月切分
      - auto：>45 天按月切分，否则不切分
    """
    mode = (chunk or "auto").strip().lower()
    if mode not in {"none", "month", "auto"}:
        raise ValueError(f"unsupported date_chunk: {mode}")

    try:
        start = dt.date.fromisoformat(start_date)
        end = dt.date.fromisoformat(end_date)
    except Exception:
        yield start_date, end_date
        return

    if mode == "auto":
        mode = "month" if (end - start).days > 45 else "none"

    if mode == "none" or start > end:
        yield start_date, end_date
        return

    cur = start.replace(day=1)
    while cur <= end:
        nxt = (cur.replace(day=28) + dt.timedelta(days=4)).replace(day=1)
        month_start = max(start, cur)
        month_end = min(end, nxt - dt.timedelta(days=1))
        yield month_start.isoformat(), month_end.isoformat()
        cur = nxt


def _load_stock_list_from_file(path: Path) -> list[StockInfo]:
    if not path.exists():
        raise FileNotFoundError(f"stock list file not found: {path}")
    suffix = path.suffix.lower()
    items: list[StockInfo] = []
    seen: set[str] = set()

    def _append(code: str, org_id: str, name: str, category: str = "", pinyin: str = "") -> None:
        code = (code or "").strip()
        org_id = (org_id or "").strip()
        if not code or not org_id:
            return
        key = f"{code}|{org_id}"
        if key in seen:
            return
        seen.add(key)
        items.append(StockInfo(code=code, org_id=org_id, name=(name or "").strip(), category=category, pinyin=pinyin))

    if suffix in {".json", ".jsonl"}:
        text = path.read_text(encoding="utf-8", errors="replace").strip()
        if not text:
            return items
        if suffix == ".jsonl":
            for line in text.splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                if isinstance(obj, dict):
                    _append(obj.get("code"), obj.get("orgId") or obj.get("org_id"), obj.get("zwjc") or obj.get("name") or "")
            return items
        try:
            obj = json.loads(text)
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"invalid stock list json: {exc}") from exc
        if isinstance(obj, dict) and isinstance(obj.get("stockList"), list):
            for item in obj["stockList"]:
                if isinstance(item, dict):
                    _append(item.get("code"), item.get("orgId") or item.get("org_id"), item.get("zwjc") or item.get("name") or "")
            return items
        if isinstance(obj, list):
            for item in obj:
                if isinstance(item, dict):
                    _append(item.get("code"), item.get("orgId") or item.get("org_id"), item.get("zwjc") or item.get("name") or "")
            return items
        return items

    with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not isinstance(row, dict):
                continue
            _append(
                row.get("code") or row.get("sec_code") or row.get("证券代码") or "",
                row.get("orgId") or row.get("org_id") or row.get("org") or row.get("组织机构代码") or "",
                row.get("zwjc") or row.get("name") or row.get("sec_name") or row.get("证券简称") or "",
            )
    return items


def crawl_wealth_management(
    *,
    start_date: str = "2025-01-01",
    end_date: str = "2025-12-31",
    keywords: list[str] | None = None,
    stock_source: str = "off",  # off/auto/file
    stock_list_path: Path | None = None,
    stock_limit: int | None = None,
    column: str = "szse",
    page_size: int = 30,
    workers: int = 4,
    min_interval_s: float = 0.4,
    date_chunk: str = "auto",
    max_pages: int | None = None,
    max_announcements: int | None = None,
    max_pdf_pages: int | None = 10,
    output_csv: Path = Path("output/results_2025.csv"),
    processed_ids_path: Path = Path("output/processed_ids.txt"),
    pdf_dir: Path = Path("cache/pdfs"),
    errors_path: Path = Path("output/errors.jsonl"),
    force: bool = False,
    schema: str = "prd",
    llm_mode: str = "off",  # off/fallback/always
    llm_base_url: str | None = None,
    llm_model: str | None = None,
    llm_effort: str | None = None,
    llm_instructions_file: str | None = None,
    llm_timeout_s: float = 120.0,
    llm_max_input_chars: int = 12000,
    llm_concurrency: int = 1,
    llm_min_interval_s: float = 0.0,
    llm_trace: str = "off",  # off/brief/dump
    llm_trace_dir: Path = Path("output/llm_traces"),
    llm_trace_max_chars: int = 1200,
    llm_prefilter: bool | None = None,
    log_announcements: bool = False,
    retry_max: int = 10,  # 0=无限重试
    retry_sleep_base_s: float = 2.0,
    retry_sleep_max_s: float = 60.0,
) -> None:
    """
    端到端任务：
      1) 按关键词分页查询公告
      2) 下载 PDF
      3) 提取表格/正文并抽取字段
      4) 输出 CSV，断点续跑
    """

    keywords = keywords or ["理财", "委托理财", "现金管理", "结构性存款", "收益凭证", "大额存单", "国债逆回购", "协定存款", "通知存款", "定期存款"]
    stock_source = (stock_source or "off").strip().lower()
    if stock_source not in {"off", "auto", "file"}:
        raise ValueError(f"unsupported stock_source: {stock_source}")
    use_title_filter = stock_source == "off"
    stock_list: list[StockInfo] = []
    if stock_source != "off":
        if stock_source == "auto":
            stock_list = fetch_stock_list_auto(create_cninfo_client(min_interval_s=min_interval_s))
        else:
            if stock_list_path is None:
                raise ValueError("stock_list_path is required when stock_source=file")
            stock_list = _load_stock_list_from_file(stock_list_path)
        if stock_limit is not None:
            stock_list = stock_list[: max(0, int(stock_limit))]
        if date_chunk == "auto":
            date_chunk = "none"
        logger.info("stock list loaded=%s source=%s", len(stock_list), stock_source)
        if not stock_list:
            raise RuntimeError("stock list is empty; check stock_source/stock_list_path")
    pdf_dir.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    errors_path.parent.mkdir(parents=True, exist_ok=True)

    processed_ids = _load_processed_ids(processed_ids_path) if not force else set()
    initial_processed_count = len(processed_ids)
    seen_ids: set[str] = set(processed_ids)

    fieldnames_full = [
        "announcement_url",
        "pdf_url",
        "company_name",
        "sec_code",
        "announcement_title",
        "announcement_date",
        "product_name",
        "product_type",
        "purchase_amount",
        "purchase_amount_yuan",
        "purchase_date",
        "extract_source",
        "snippet",
    ]
    fieldnames_prd = ["公告链接", "上市公司名字", "理财产品类型", "购买金额", "购买时间"]
    if schema not in {"full", "prd"}:
        raise ValueError(f"unsupported schema: {schema}")
    fieldnames = fieldnames_prd if schema == "prd" else fieldnames_full
    # PRD 交付一般需要用 Excel 打开，CSV 使用 UTF-8 BOM 更友好
    header_encoding = "utf-8-sig" if schema == "prd" else "utf-8"
    _ensure_csv_header(output_csv, fieldnames, encoding=header_encoding)

    def to_row(rec: PurchaseRecord) -> dict:
        if schema == "full":
            return asdict(rec)
        name = (rec.product_name or "").strip()
        typ = (rec.product_type or "").strip()
        if name and typ:
            # 避免重复：名称里已包含类型时不再追加
            compact_name = re.sub(r"\s+", "", name)
            compact_typ = re.sub(r"\s+", "", typ)
            product_display = name if compact_typ in compact_name else f"{name}（{typ}）"
        else:
            product_display = name or typ
        return {
            "公告链接": rec.announcement_url,
            "上市公司名字": rec.company_name,
            "理财产品类型": product_display,
            "购买金额": rec.purchase_amount,
            "购买时间": rec.purchase_date,
        }

    def _blank(v: object) -> bool:
        if v is None:
            return True
        if isinstance(v, str):
            return not v.strip()
        return False

    def _fill_prd_defaults(row: dict, *, ann: Announcement | None = None) -> dict:
        if _blank(row.get("公告链接")) and ann is not None:
            row["公告链接"] = ann.detail_url
        if _blank(row.get("上市公司名字")) and ann is not None:
            row["上市公司名字"] = ann.sec_name
        if _blank(row.get("购买时间")) and ann is not None:
            row["购买时间"] = ann.announcement_date
        if _blank(row.get("理财产品类型")):
            row["理财产品类型"] = "未披露"
        if _blank(row.get("购买金额")):
            row["购买金额"] = "未披露"
        return row

    def _parse_date(value: str | None) -> dt.date | None:
        if not value:
            return None
        text = str(value).strip()
        if not text:
            return None
        text = text.replace("/", "-")
        try:
            return dt.date.fromisoformat(text)
        except Exception:
            return None

    def _purchase_date_in_range(row: dict) -> bool:
        # 购买时间可解析时必须落在指定区间内，否则丢弃
        purchase = row.get("购买时间")
        pd = _parse_date(purchase)
        if pd is None:
            return True
        sd = _parse_date(start_date)
        ed = _parse_date(end_date)
        if sd is None or ed is None:
            return True
        return sd <= pd <= ed

    def _prd_missing_fields(row: dict) -> list[str]:
        required = ["公告链接", "上市公司名字", "理财产品类型", "购买金额", "购买时间"]
        missing: list[str] = []
        for k in required:
            if _blank(row.get(k)):
                missing.append(k)
        return missing

    # 共享限速器：多线程时仍控制整体请求速率
    shared_limiter = RateLimiter(min_interval_s=float(min_interval_s), jitter_s=0.2)
    thread_local = threading.local()

    llm_mode = (llm_mode or "off").strip().lower()
    if llm_mode not in {"off", "fallback", "always"}:
        raise ValueError(f"unsupported llm_mode: {llm_mode}")

    llm_trace = (llm_trace or "off").strip().lower()
    if llm_trace not in {"off", "brief", "dump"}:
        raise ValueError(f"unsupported llm_trace: {llm_trace}")

    llm_extractor: WealthMgmtLlmExtractor | None = None
    if llm_mode != "off":
        cfg = load_llm_config_from_env(
            base_url=llm_base_url,
            model=llm_model,
            reasoning_effort=llm_effort,
            instructions_file=llm_instructions_file,
            timeout_s=llm_timeout_s,
            max_input_chars=llm_max_input_chars,
            concurrency=llm_concurrency,
            min_interval_s=llm_min_interval_s,
        )
        if cfg is None:
            raise RuntimeError(
                "LLM 已启用，但未设置 CNINFO_LLM_API_KEY（建议用环境变量设置，避免泄露密钥）"
            )
        if not (cfg.instructions or "").strip():
            logger.warning(
                "LLM 已启用但未找到 instructions 模板：codex_api_proxy 可能会返回 400（建议设置 CNINFO_LLM_INSTRUCTIONS_FILE 或 --llm-instructions-file）"
            )
        else:
            logger.info("LLM instructions 已加载：len=%s", len(cfg.instructions or ""))
        llm_extractor = WealthMgmtLlmExtractor(
            cfg,
            trace_mode=llm_trace,
            trace_dir=llm_trace_dir,
            trace_max_chars=int(llm_trace_max_chars),
        )

    if llm_prefilter is None:
        prefilter_enabled = stock_source != "off"
    else:
        prefilter_enabled = bool(llm_prefilter)
    if prefilter_enabled and llm_mode == "off":
        logger.warning("已开启 LLM 预筛，但 llm=off，预筛将被跳过")

    def get_client() -> HttpClient:
        c = getattr(thread_local, "client", None)
        if c is not None:
            return c
        session = requests.Session()
        c = HttpClient(session=session, rate_limiter=shared_limiter)
        bootstrap_session(c)
        thread_local.client = c
        return c

    def _amount_key(r: PurchaseRecord) -> str:
        if r.purchase_amount_yuan is not None:
            return str(r.purchase_amount_yuan)
        s = r.purchase_amount or ""
        m = re.search(r"\d{1,3}(?:,\d{3})*(?:\.\d+)?|\d+(?:\.\d+)?", s)
        return (m.group(0).replace(",", "") if m else "").strip()

    def _dedupe_records(records: list[PurchaseRecord]) -> list[PurchaseRecord]:
        """
        同一份 PDF 里表格抽取时，可能会因为分页/重复表头导致“同一行被解析多次”。
        这里做“公告内去重”，避免 CSV 出现大量重复记录。
        """

        def compact(s: str) -> str:
            return re.sub(r"\s+", "", (s or "")).strip()

        def key_of(r: PurchaseRecord) -> tuple[str, str, str, str]:
            # 以“金额(归一)+名称+类型+日期”为主键；同一公告内足够区分大多数记录
            return (
                _amount_key(r) or compact(r.purchase_amount),
                compact(r.product_name),
                compact(r.product_type),
                (r.purchase_date or "").strip(),
            )

        def score(r: PurchaseRecord) -> int:
            sc = 0
            if (r.product_name or "").strip():
                sc += 4
            typ = (r.product_type or "").strip()
            if typ and typ not in ("理财产品",):
                sc += 3
            elif typ:
                sc += 1
            if r.purchase_amount_yuan is not None:
                sc += 1
            if "llm" in (r.extract_source or ""):
                sc += 1
            return sc

        def merge_best(a: PurchaseRecord, b: PurchaseRecord) -> PurchaseRecord:
            # 选更“信息量高”的为 base，并用另一条补齐空字段
            if score(b) > score(a):
                a, b = b, a

            name = (a.product_name or "").strip() or (b.product_name or "").strip()
            typ_a = (a.product_type or "").strip()
            typ_b = (b.product_type or "").strip()
            typ = typ_a
            if (not typ_a or typ_a in ("理财产品",)) and typ_b and typ_b not in ("理财产品",):
                typ = typ_b
            elif not typ_a and typ_b:
                typ = typ_b

            amount_yuan = a.purchase_amount_yuan if a.purchase_amount_yuan is not None else b.purchase_amount_yuan
            purchase_date = (a.purchase_date or "").strip() or (b.purchase_date or "").strip()
            extract_source = a.extract_source
            if b.extract_source and b.extract_source not in (extract_source or ""):
                extract_source = f"{extract_source}|{b.extract_source}"
            snippet = a.snippet if len((a.snippet or "")) >= len((b.snippet or "")) else b.snippet

            return PurchaseRecord(
                announcement_url=a.announcement_url,
                pdf_url=a.pdf_url,
                company_name=a.company_name,
                sec_code=a.sec_code,
                announcement_title=a.announcement_title,
                announcement_date=a.announcement_date,
                product_name=name,
                product_type=typ,
                purchase_amount=a.purchase_amount,
                purchase_amount_yuan=amount_yuan,
                purchase_date=purchase_date,
                extract_source=extract_source,
                snippet=snippet,
            )

        best: dict[tuple[str, str, str, str], PurchaseRecord] = {}
        order: list[tuple[str, str, str, str]] = []
        for r in records:
            k = key_of(r)
            if k not in best:
                best[k] = r
                order.append(k)
                continue
            best[k] = merge_best(best[k], r)
        return [best[k] for k in order]

    def _merge_rule_and_llm(rule_recs: list[PurchaseRecord], llm_recs: list[PurchaseRecord]) -> list[PurchaseRecord]:
        if not llm_recs:
            return rule_recs
        if not rule_recs:
            return llm_recs

        merged = list(rule_recs)
        seen: set[tuple[str, str, str]] = set()
        idx_by_amount: dict[str, list[int]] = {}

        for i, r in enumerate(merged):
            k = _amount_key(r)
            seen.add((k, (r.product_name or "").strip(), (r.product_type or "").strip()))
            if k:
                idx_by_amount.setdefault(k, []).append(i)

        for lr in llm_recs:
            k = _amount_key(lr)
            name = (lr.product_name or "").strip()
            typ = (lr.product_type or "").strip()
            if (k, name, typ) in seen:
                continue

            idxs = idx_by_amount.get(k or "")
            if k and idxs and len(idxs) == 1:
                i = idxs[0]
                r = merged[i]
                new_name = r.product_name
                new_typ = r.product_type

                if not (new_name or "").strip() and name:
                    new_name = name
                if (not (new_typ or "").strip() or (new_typ or "").strip() in ("理财产品",)) and typ and typ not in ("理财产品",):
                    new_typ = typ

                if new_name != r.product_name or new_typ != r.product_type:
                    merged[i] = PurchaseRecord(
                        announcement_url=r.announcement_url,
                        pdf_url=r.pdf_url,
                        company_name=r.company_name,
                        sec_code=r.sec_code,
                        announcement_title=r.announcement_title,
                        announcement_date=r.announcement_date,
                        product_name=new_name,
                        product_type=new_typ,
                        purchase_amount=r.purchase_amount,
                        purchase_amount_yuan=r.purchase_amount_yuan,
                        purchase_date=r.purchase_date,
                        extract_source=f"{r.extract_source}+llm",
                        snippet=r.snippet,
                    )
                    seen.add((k, (new_name or "").strip(), (new_typ or "").strip()))
                continue

            merged.append(lr)
            seen.add((k, name, typ))
            if k:
                idx_by_amount.setdefault(k, []).append(len(merged) - 1)

        return merged

    def process_one(ann: Announcement, *, keyword: str, attempt: int = 0) -> tuple[str, list[PurchaseRecord], bool]:
        clean_title = _clean_title(ann.title)
        if log_announcements:
            logger.info(
                "开始处理公告 attempt=%s keyword=%s id=%s 公司=%s(%s) 日期=%s 标题=%s",
                attempt,
                keyword,
                ann.announcement_id,
                ann.sec_name,
                ann.sec_code,
                ann.announcement_date,
                clean_title,
            )
            logger.info("公告链接 id=%s detail=%s pdf=%s", ann.announcement_id, ann.detail_url, ann.pdf_url)

        client = get_client()
        pdf_path = pdf_dir / f"{ann.announcement_id}.pdf"

        if pdf_path.exists() and not force:
            pdf_bytes = pdf_path.read_bytes()
        else:
            pdf_bytes = download_pdf_bytes(client, ann.pdf_url)
            pdf_path.write_bytes(pdf_bytes)

        should_call_llm = llm_extractor is not None and llm_mode in {"always", "fallback"}
        if prefilter_enabled and should_call_llm and llm_extractor is not None:
            pre_pdf = parse_pdf(pdf_bytes, max_pages=1)
            try:
                logger.info(
                    "LLM预筛开始 attempt=%s id=%s 公司=%s 日期=%s 标题=%s detail=%s",
                    attempt,
                    ann.announcement_id,
                    ann.sec_name,
                    ann.announcement_date,
                    clean_title,
                    ann.detail_url,
                )
                pre_relevant = llm_extractor.prefilter(
                    ann,
                    pre_pdf,
                    pdf_bytes=pdf_bytes,
                    attempt=attempt,
                )
                logger.info("LLM预筛完成 attempt=%s id=%s relevant=%s", attempt, ann.announcement_id, pre_relevant)
            except Exception as exc:
                with errors_path.open("a", encoding="utf-8") as f:
                    f.write(
                        json.dumps(
                            {
                                "error": f"llm_prefilter_error: {str(exc)}",
                                "attempt": attempt,
                                "keyword": keyword,
                                "announcement_id": ann.announcement_id,
                                "sec_code": ann.sec_code,
                                "company_name": ann.sec_name,
                                "announcement_title": _clean_title(ann.title),
                                "pdf_url": ann.pdf_url,
                                "detail_url": ann.detail_url,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                stats["errors_written"] += 1
                pre_relevant = True
            if not pre_relevant:
                logger.info("LLM预筛为非理财公告 id=%s", ann.announcement_id)
                return ann.announcement_id, [], False

        pdf_content = parse_pdf(pdf_bytes, max_pages=max_pdf_pages)
        rule_recs: list[PurchaseRecord] = []
        llm_relevant = True
        if llm_mode == "off":
            rule_recs = extract_records(ann, pdf_content)
            if log_announcements:
                logger.info("规则抽取完成 id=%s 记录数=%s", ann.announcement_id, len(rule_recs))

        llm_recs: list[PurchaseRecord] = []
        if should_call_llm and llm_extractor is not None:
            try:
                logger.info(
                    "LLM分析开始 attempt=%s id=%s 公司=%s 日期=%s 标题=%s detail=%s",
                    attempt,
                    ann.announcement_id,
                    ann.sec_name,
                    ann.announcement_date,
                    clean_title,
                    ann.detail_url,
                )
                llm_result = llm_extractor.extract(
                    ann,
                    pdf_content,
                    pdf_bytes=pdf_bytes,
                    attempt=attempt,
                )
                llm_relevant = bool(getattr(llm_result, "relevant", True))
                llm_recs = list(getattr(llm_result, "records", []))
                logger.info("LLM分析完成 attempt=%s id=%s 记录数=%s", attempt, ann.announcement_id, len(llm_recs))
                if not llm_relevant:
                    logger.info("LLM筛选为非理财公告 id=%s", ann.announcement_id)
                    return ann.announcement_id, [], False
            except Exception as exc:
                # LLM 失败不阻断主流程，但记录到 errors，便于排查
                with errors_path.open("a", encoding="utf-8") as f:
                    f.write(
                        json.dumps(
                            {
                                "error": f"llm_error: {str(exc)}",
                                "attempt": attempt,
                                "keyword": keyword,
                                "announcement_id": ann.announcement_id,
                                "sec_code": ann.sec_code,
                                "company_name": ann.sec_name,
                                "announcement_title": _clean_title(ann.title),
                                "pdf_url": ann.pdf_url,
                                "detail_url": ann.detail_url,
                                "llm_base_url": getattr(getattr(llm_extractor, "_client", None), "base_url", ""),
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                stats["errors_written"] += 1
                # 若规则抽取也没有结果，则视为“需要 LLM 才能补齐”，抛出异常走重试队列
                if not rule_recs:
                    raise
                llm_recs = []

        recs = llm_recs if llm_mode != "off" else rule_recs
        before = len(recs)
        recs = _dedupe_records(recs)
        if log_announcements:
            if len(recs) != before:
                logger.info("公告内去重 id=%s %s->%s", ann.announcement_id, before, len(recs))
            logger.info("合并完成 id=%s 输出记录数=%s", ann.announcement_id, len(recs))
        return ann.announcement_id, recs, llm_relevant

    total_submitted = 0
    futures: set[Future] = set()
    retry_counts: dict[str, int] = {}

    @dataclasses.dataclass(frozen=True)
    class RetryItem:
        not_before_ts: float
        ann: Announcement
        keyword: str
        attempt: int

    retry_queue: deque[RetryItem] = deque()
    in_retry_queue: set[str] = set()

    def _schedule_retry(ann: Announcement, *, keyword: str, reason: str) -> None:
        ann_id = ann.announcement_id
        attempt = retry_counts.get(ann_id, 0) + 1
        retry_counts[ann_id] = attempt

        if retry_max != 0 and attempt > retry_max:
            logger.warning("放弃重试 id=%s 已超过 retry_max=%s reason=%s", ann_id, retry_max, reason)
            return
        if ann_id in in_retry_queue:
            return

        base = max(0.0, float(retry_sleep_base_s))
        cap = max(base, float(retry_sleep_max_s))
        # 指数退避 + 少量随机抖动，避免“同一批失败同时重试”
        delay = min(cap, base * (2 ** max(0, attempt - 1)))
        delay = delay + random.random() * 0.5
        retry_queue.append(RetryItem(not_before_ts=time.time() + delay, ann=ann, keyword=keyword, attempt=attempt))
        in_retry_queue.add(ann_id)
        stats["retries_scheduled"] += 1
        logger.warning("计划重试 id=%s attempt=%s delay=%.1fs reason=%s", ann_id, attempt, delay, reason)

    def flush_done(done: set[Future]) -> None:
        nonlocal total_submitted
        rows_to_write: list[dict] = []
        for fut in done:
            try:
                ann_id, recs, relevant = fut.result()
                if not relevant:
                    stats["filtered_by_llm"] += 1
                    ann: Announcement | None = getattr(fut, "_ann", None)  # type: ignore[attr-defined]
                    if ann is not None:
                        with errors_path.open("a", encoding="utf-8") as f:
                            f.write(
                                json.dumps(
                                    {
                                        "error": "filtered_by_llm",
                                        "announcement_id": ann.announcement_id,
                                        "sec_code": ann.sec_code,
                                        "company_name": ann.sec_name,
                                        "announcement_title": _clean_title(ann.title),
                                        "pdf_url": ann.pdf_url,
                                        "detail_url": ann.detail_url,
                                    },
                                    ensure_ascii=False,
                                )
                                + "\n"
                            )
                    processed_ids.add(ann_id)
                    _append_processed_id(processed_ids_path, ann_id)
                    continue
                if not recs:
                    ann: Announcement | None = getattr(fut, "_ann", None)  # type: ignore[attr-defined]
                    if schema == "full":
                        if ann is not None:
                            recs = [
                                PurchaseRecord(
                                    announcement_url=ann.detail_url,
                                    pdf_url=ann.pdf_url,
                                    company_name=ann.sec_name,
                                    sec_code=ann.sec_code,
                                    announcement_title=_clean_title(ann.title),
                                    announcement_date=ann.announcement_date,
                                    product_name="",
                                    product_type="",
                                    purchase_amount="",
                                    purchase_amount_yuan=None,
                                    purchase_date=ann.announcement_date,
                                    extract_source="none",
                                    snippet="",
                                )
                            ]
                    else:
                        # PRD 输出：无记录不落行，仅记录 errors 便于复核
                        if ann is not None:
                            with errors_path.open("a", encoding="utf-8") as f:
                                f.write(
                                    json.dumps(
                                        {
                                            "error": "no_records_extracted",
                                            "announcement_id": ann.announcement_id,
                                            "sec_code": ann.sec_code,
                                            "company_name": ann.sec_name,
                                            "announcement_title": _clean_title(ann.title),
                                            "pdf_url": ann.pdf_url,
                                            "detail_url": ann.detail_url,
                                        },
                                        ensure_ascii=False,
                                    )
                                    + "\n"
                                )
                            stats["errors_written"] += 1
                        recs = []
                for r in recs:
                    row = to_row(r)
                    if schema == "prd":
                        row = _fill_prd_defaults(row, ann=getattr(fut, "_ann", None))  # type: ignore[arg-type]
                        if not _purchase_date_in_range(row):
                            with errors_path.open("a", encoding="utf-8") as f:
                                f.write(
                                    json.dumps(
                                        {
                                            "error": "purchase_date_out_of_range",
                                            "announcement_id": ann_id,
                                            "sec_code": r.sec_code,
                                            "company_name": r.company_name,
                                            "announcement_title": _clean_title(r.announcement_title),
                                            "pdf_url": r.pdf_url,
                                            "detail_url": r.announcement_url,
                                            "purchase_date": row.get("购买时间"),
                                        },
                                        ensure_ascii=False,
                                    )
                                    + "\n"
                                )
                            stats["errors_written"] += 1
                            continue
                        product_type = (row.get("理财产品类型") or "").strip()
                        purchase_amount = (row.get("购买金额") or "").strip()
                        if product_type in ("", "未披露") and purchase_amount in ("", "未披露"):
                            with errors_path.open("a", encoding="utf-8") as f:
                                f.write(
                                    json.dumps(
                                        {
                                            "error": "both_not_disclosed",
                                            "announcement_id": ann_id,
                                            "sec_code": r.sec_code,
                                            "company_name": r.company_name,
                                            "announcement_title": _clean_title(r.announcement_title),
                                            "pdf_url": r.pdf_url,
                                            "detail_url": r.announcement_url,
                                        },
                                        ensure_ascii=False,
                                    )
                                    + "\n"
                                )
                            stats["errors_written"] += 1
                            continue
                        missing = _prd_missing_fields(row)
                        if missing:
                            with errors_path.open("a", encoding="utf-8") as f:
                                f.write(
                                    json.dumps(
                                        {
                                            "error": "missing_required_fields",
                                            "missing_fields": missing,
                                            "announcement_id": ann_id,
                                            "sec_code": r.sec_code,
                                            "company_name": r.company_name,
                                            "announcement_title": _clean_title(r.announcement_title),
                                            "pdf_url": r.pdf_url,
                                            "detail_url": r.announcement_url,
                                            "product_name": r.product_name,
                                            "product_type": r.product_type,
                                            "purchase_amount": r.purchase_amount,
                                            "purchase_date": r.purchase_date,
                                            "extract_source": r.extract_source,
                                        },
                                        ensure_ascii=False,
                                    )
                                    + "\n"
                                )
                            stats["errors_written"] += 1
                    rows_to_write.append(row)
                    if _CODE_RE.match(r.sec_code or ""):
                        stats_unique_companies_output.add(r.sec_code)
                processed_ids.add(ann_id)
                _append_processed_id(processed_ids_path, ann_id)
            except Exception as exc:
                ann: Announcement | None = None
                kw: str = ""
                at: int = 0
                err = {"error": str(exc)}
                try:
                    # 尽量把上下文带出来（如果是我们提交的任务）
                    ann = getattr(fut, "_ann", None)  # type: ignore[attr-defined]
                    kw = getattr(fut, "_keyword", "")  # type: ignore[attr-defined]
                    at = int(getattr(fut, "_attempt", 0))  # type: ignore[attr-defined]
                    if ann:
                        err.update(
                            {
                                "attempt": at,
                                "keyword": kw,
                                "announcement_id": ann.announcement_id,
                                "sec_code": ann.sec_code,
                                "company_name": ann.sec_name,
                                "announcement_title": _clean_title(ann.title),
                                "pdf_url": ann.pdf_url,
                                "detail_url": ann.detail_url,
                            }
                        )
                except Exception:
                    pass
                with errors_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(err, ensure_ascii=False) + "\n")
                stats["errors_written"] += 1
                # 失败的公告不标记 processed，并放到队列末尾等待重试
                if ann is not None:
                    _schedule_retry(ann, keyword=kw or "(unknown)", reason=str(exc)[:200])

        if rows_to_write:
            _append_csv_rows(output_csv, fieldnames, rows_to_write)
            stats["rows_written"] += len(rows_to_write)

    # 预热一次（可选）：确保访问成功后再开始长任务
    bootstrap_session(create_cninfo_client(min_interval_s=min_interval_s))
    logger.info(
        "任务开始：%s~%s date_chunk=%s keywords=%s workers=%s llm=%s schema=%s output=%s processed_ids=%s",
        start_date,
        end_date,
        date_chunk,
        ",".join(keywords),
        workers,
        llm_mode,
        schema,
        str(output_csv),
        str(processed_ids_path),
    )

    stats = {"rows_written": 0, "errors_written": 0, "retries_scheduled": 0, "filtered_by_llm": 0}
    stats_unique_companies_seen: set[str] = set()
    stats_unique_companies_output: set[str] = set()

    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as ex:
        pbar = tqdm(desc="processing", unit="ann")

        if stock_source != "off":
            for stock in stock_list:
                keyword_label = f"stock:{stock.code}"
                for chunk_start, chunk_end in _iter_date_chunks(start_date, end_date, date_chunk):
                    logger.info("开始搜索公司：%s range=%s~%s", keyword_label, chunk_start, chunk_end)
                    for ann in iter_his_announcements(
                        get_client(),
                        start_date=chunk_start,
                        end_date=chunk_end,
                        searchkey="",
                        stock=stock.stock_param,
                        column=column,
                        page_size=page_size,
                        max_pages=max_pages,
                        retry_max=retry_max,
                        retry_sleep_base_s=retry_sleep_base_s,
                        retry_sleep_max_s=retry_sleep_max_s,
                    ):
                        if max_announcements is not None and total_submitted >= max_announcements:
                            break

                        if not _CODE_RE.match(ann.sec_code):
                            continue
                        stats_unique_companies_seen.add(ann.sec_code)

                        clean_title = _clean_title(ann.title)
                        if use_title_filter and not is_candidate_announcement(clean_title):
                            continue

                        if ann.announcement_id in seen_ids and not force:
                            continue
                        seen_ids.add(ann.announcement_id)

                        fut = ex.submit(process_one, ann, keyword=keyword_label, attempt=0)
                        setattr(fut, "_ann", ann)  # 记录上下文
                        setattr(fut, "_keyword", keyword_label)
                        setattr(fut, "_attempt", 0)
                        futures.add(fut)
                        total_submitted += 1
                        pbar.update(1)

                        if len(futures) >= max(8, int(workers) * 4):
                            done, pending = wait(futures, return_when=FIRST_COMPLETED)
                            futures = set(pending)
                            flush_done(set(done))

                    if max_announcements is not None and total_submitted >= max_announcements:
                        break

                if max_announcements is not None and total_submitted >= max_announcements:
                    break
        else:
            for keyword in keywords:
                for chunk_start, chunk_end in _iter_date_chunks(start_date, end_date, date_chunk):
                    logger.info("开始搜索关键词：%s range=%s~%s", keyword, chunk_start, chunk_end)
                    for ann in iter_his_announcements(
                        get_client(),
                        start_date=chunk_start,
                        end_date=chunk_end,
                        searchkey=keyword,
                        column=column,
                        page_size=page_size,
                        max_pages=max_pages,
                        retry_max=retry_max,
                        retry_sleep_base_s=retry_sleep_base_s,
                        retry_sleep_max_s=retry_sleep_max_s,
                    ):
                        if max_announcements is not None and total_submitted >= max_announcements:
                            break

                        if not _CODE_RE.match(ann.sec_code):
                            continue
                        stats_unique_companies_seen.add(ann.sec_code)

                        clean_title = _clean_title(ann.title)
                        if not is_candidate_announcement(clean_title):
                            continue

                        if ann.announcement_id in seen_ids and not force:
                            continue
                        seen_ids.add(ann.announcement_id)

                        fut = ex.submit(process_one, ann, keyword=keyword, attempt=0)
                        setattr(fut, "_ann", ann)  # 记录上下文
                        setattr(fut, "_keyword", keyword)
                        setattr(fut, "_attempt", 0)
                        futures.add(fut)
                        total_submitted += 1
                        pbar.update(1)

                        if len(futures) >= max(8, int(workers) * 4):
                            done, pending = wait(futures, return_when=FIRST_COMPLETED)
                            futures = set(pending)
                            flush_done(set(done))

                    if max_announcements is not None and total_submitted >= max_announcements:
                        break

                if max_announcements is not None and total_submitted >= max_announcements:
                    break
        # 先等完当前 in-flight
        if futures:
            done, _ = wait(futures)
            flush_done(set(done))
            futures.clear()

        # 再处理重试队列（失败的内容已放入队列末尾）
        while retry_queue:
            # 填充到一定并发
            threshold = max(8, int(workers) * 4)
            while retry_queue and len(futures) < threshold:
                item = retry_queue[0]
                now = time.time()
                if item.not_before_ts > now:
                    # 队首还没到重试时间，轮转到队尾，避免阻塞其它待重试项
                    retry_queue.rotate(-1)
                    # 若当前没有任务在跑，稍微 sleep 一下避免空转
                    if not futures:
                        time.sleep(min(1.0, max(0.0, item.not_before_ts - now)))
                    continue
                item = retry_queue.popleft()
                in_retry_queue.discard(item.ann.announcement_id)

                fut = ex.submit(process_one, item.ann, keyword=item.keyword, attempt=item.attempt)
                setattr(fut, "_ann", item.ann)
                setattr(fut, "_keyword", item.keyword)
                setattr(fut, "_attempt", item.attempt)
                futures.add(fut)
                pbar.update(1)

            if not futures:
                continue

            done, pending = wait(futures, return_when=FIRST_COMPLETED)
            futures = set(pending)
            flush_done(set(done))

        if futures:
            done, _ = wait(futures)
            flush_done(set(done))

        pbar.close()

    logger.info(
        "任务结束：新增processed=%s 总processed=%s rows=%s companies_seen=%s companies_output=%s errors=%s filtered_by_llm=%s retries_scheduled=%s",
        max(0, len(processed_ids) - int(initial_processed_count)),
        len(processed_ids),
        stats.get("rows_written", 0),
        len(stats_unique_companies_seen),
        len(stats_unique_companies_output),
        stats.get("errors_written", 0),
        stats.get("filtered_by_llm", 0),
        stats.get("retries_scheduled", 0),
    )
