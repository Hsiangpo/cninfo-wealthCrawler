from __future__ import annotations

import csv
import json
import logging
import random
import re
import time
from datetime import datetime
from html import unescape
from pathlib import Path
from urllib.parse import urlencode

from ..integrations.cninfo_api import StockInfo, create_cninfo_client, fetch_stock_list_auto
from ..integrations.http_client import HttpClient, RateLimiter


logger = logging.getLogger(__name__)

_CODE_RE = re.compile(r"^\d{6}$")
_TAG_RE = re.compile(r"<[^>]+>")

SZSE_LIST_URL = "https://www.szse.cn/api/report/ShowReport/data"
SSE_LIST_URL = "https://query.sse.com.cn/security/stock/getStockListData2.do"


def _clean_html(text: str) -> str:
    return unescape(_TAG_RE.sub("", text or "")).strip()


def _build_url(base: str, params: dict[str, str]) -> str:
    return f"{base}?{urlencode(params)}"


def _get_json_with_retry(
    client: HttpClient,
    *,
    url: str,
    headers: dict[str, str],
    timeout_s: float = 30.0,
    max_attempts: int = 6,
    base_sleep_s: float = 1.0,
    max_sleep_s: float = 10.0,
) -> object:
    last_err: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            client.rate_limiter.wait()
            resp = client.session.get(url, headers=headers, timeout=timeout_s)
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            if attempt >= max_attempts:
                break
            sleep_s = min(max_sleep_s, base_sleep_s * (2 ** (attempt - 1)))
            sleep_s += random.random() * 0.5
            logger.warning("stock list request retry=%s url=%s err=%s", attempt, url, str(exc)[:200])
            time.sleep(sleep_s)
    assert last_err is not None
    raise last_err


def _fetch_szse_codes(
    client: HttpClient,
    *,
    tab: str,
    max_pages: int | None = None,
) -> set[str]:
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Referer": "https://www.szse.cn/market/product/stock/list/index.html",
        "Connection": "close",
    }
    base_params = {"SHOWTYPE": "JSON", "CATALOGID": "1110", tab: tab}
    first_url = _build_url(SZSE_LIST_URL, base_params)
    first = _get_json_with_retry(client, url=first_url, headers=headers, timeout_s=30)
    if not isinstance(first, list) or not first:
        raise ValueError("unexpected SZSE response")
    meta = first[0].get("metadata") or {}
    pagecount = int(meta.get("pagecount") or 0)
    if pagecount <= 0:
        return set()
    if max_pages is not None:
        pagecount = min(pagecount, max_pages)

    codes: set[str] = set()
    for page in range(1, pagecount + 1):
        params = dict(base_params)
        params["PAGENO"] = str(page)
        url = _build_url(SZSE_LIST_URL, params)
        page_obj = _get_json_with_retry(client, url=url, headers=headers, timeout_s=30)
        rows = page_obj[0].get("data") if isinstance(page_obj, list) and page_obj else []
        if not isinstance(rows, list):
            continue
        for row in rows:
            if not isinstance(row, dict):
                continue
            code = (
                str(row.get("agdm") or "").strip()
                or str(row.get("bgdm") or "").strip()
                or str(row.get("cdrdm") or "").strip()
            )
            if code and _CODE_RE.match(code):
                codes.add(code)
    return codes


def _fetch_sse_codes(client: HttpClient, *, stock_type: str) -> set[str]:
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Referer": "https://www.sse.com.cn/assortment/stock/list/share/",
        "Accept": "application/json, text/plain, */*",
        "Connection": "close",
    }
    page_size = 2000
    page_no = 1
    codes: set[str] = set()
    while True:
        params = {
            "isPagination": "true",
            "stockCode": "",
            "csrcCode": "",
            "areaName": "",
            "stockType": stock_type,
            "pageHelp.cacheSize": "1",
            "pageHelp.pageSize": str(page_size),
            "pageHelp.pageNo": str(page_no),
        }
        url = _build_url(SSE_LIST_URL, params)
        obj = _get_json_with_retry(client, url=url, headers=headers, timeout_s=30)
        page_help = obj.get("pageHelp") if isinstance(obj, dict) else None
        rows = page_help.get("data") if isinstance(page_help, dict) else []
        if not isinstance(rows, list):
            break
        for row in rows:
            if not isinstance(row, dict):
                continue
            for key in ("SECURITY_CODE_A", "SECURITY_CODE_B", "COMPANY_CODE"):
                code = str(row.get(key) or "").strip()
                if code and code != "-" and _CODE_RE.match(code):
                    codes.add(code)
        total_pages = int(page_help.get("pageCount") or 1)
        if page_no >= total_pages:
            break
        page_no += 1
    return codes


def fetch_active_codes_from_exchanges(
    *,
    include_b_share: bool = True,
    include_cdr: bool = False,
    max_pages_szse: int | None = None,
    min_interval_s: float = 0.1,
) -> set[str]:
    client = HttpClient(rate_limiter=RateLimiter(min_interval_s=min_interval_s, jitter_s=0.1))
    codes: set[str] = set()

    codes.update(_fetch_szse_codes(client, tab="tab1", max_pages=max_pages_szse))
    if include_b_share:
        codes.update(_fetch_szse_codes(client, tab="tab2", max_pages=max_pages_szse))
    if include_cdr:
        codes.update(_fetch_szse_codes(client, tab="tab3", max_pages=max_pages_szse))

    codes.update(_fetch_sse_codes(client, stock_type="1"))
    codes.update(_fetch_sse_codes(client, stock_type="8"))
    if include_b_share:
        codes.update(_fetch_sse_codes(client, stock_type="2"))
    return codes


def build_stock_list(
    *,
    filter_active: bool = True,
    include_b_share: bool = True,
    include_cdr: bool = False,
    min_interval_s: float = 0.1,
) -> tuple[list[StockInfo], dict[str, int]]:
    cninfo_client = create_cninfo_client(min_interval_s=min_interval_s)
    all_stocks = fetch_stock_list_auto(cninfo_client)
    logger.info("cninfo stock list loaded=%s", len(all_stocks))
    stats = {"total": len(all_stocks), "filtered": 0, "fallback": 0}
    if not filter_active:
        return all_stocks, stats

    try:
        active_codes = fetch_active_codes_from_exchanges(
            include_b_share=include_b_share,
            include_cdr=include_cdr,
            min_interval_s=min_interval_s,
        )
        logger.info(
            "exchange active codes=%s include_b_share=%s include_cdr=%s",
            len(active_codes),
            include_b_share,
            include_cdr,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("exchange list fetch failed, fallback to A-share prefixes: %s", str(exc)[:200])
        active_codes = {s.code for s in all_stocks if s.code and s.code[0] in {"0", "3", "6"}}
        stats["fallback"] = len(active_codes)
    filtered = [s for s in all_stocks if s.code in active_codes]
    stats["filtered"] = len(filtered)
    logger.info("filtered stock list=%s", len(filtered))
    return filtered, stats


def save_stock_list(path: Path, items: list[StockInfo]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["code", "orgId", "zwjc", "category", "pinyin"])
            writer.writeheader()
            for item in items:
                writer.writerow(
                    {
                        "code": item.code,
                        "orgId": item.org_id,
                        "zwjc": item.name,
                        "category": item.category,
                        "pinyin": item.pinyin,
                    }
                )
        return

    payload = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "count": len(items),
        "stockList": [
            {
                "code": s.code,
                "orgId": s.org_id,
                "zwjc": s.name,
                "category": s.category,
                "pinyin": s.pinyin,
            }
            for s in items
        ],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
