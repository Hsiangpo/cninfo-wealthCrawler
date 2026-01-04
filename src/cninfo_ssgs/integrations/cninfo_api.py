from __future__ import annotations

import datetime as dt
import logging
import random
import time
from dataclasses import dataclass
from typing import Any, Iterable

import requests
from zoneinfo import ZoneInfo

from .http_client import HttpClient, RateLimiter

SEARCH_PAGE_URL = "https://www.cninfo.com.cn/new/commonUrl/pageOfSearch?url=disclosure/list/search"
HIS_ANNOUNCEMENT_QUERY_URL = "https://www.cninfo.com.cn/new/hisAnnouncement/query"
STATIC_BASE_URL = "https://static.cninfo.com.cn/"
STOCK_LIST_URL = "https://www.cninfo.com.cn/new/data/szse_stock.json"


_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)
logger = logging.getLogger(__name__)


def _default_headers(referer: str) -> dict[str, str]:
    return {
        "User-Agent": _UA,
        "Referer": referer,
        "Origin": "https://www.cninfo.com.cn",
        "X-Requested-With": "XMLHttpRequest",
        # 关键：带 charset=UTF-8，避免中文关键字在服务端被错误解析
        "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
        "Accept": "*/*",
        "Accept-Language": "zh-CN,zh;q=0.9",
    }


def _ms_to_date_str(ms: int | None) -> str:
    if not ms:
        return ""
    # cninfo 的 announcementTime 常见为“北京时间 00:00:00”的时间戳，
    # 若直接按 UTC 取 date 可能偏差 1 天，因此按 Asia/Shanghai 取公告日期。
    d = dt.datetime.fromtimestamp(ms / 1000, tz=ZoneInfo("Asia/Shanghai")).date()
    return d.strftime("%Y-%m-%d")


@dataclass(frozen=True)
class Announcement:
    sec_code: str
    sec_name: str
    org_id: str
    announcement_id: str
    title: str
    announcement_time_ms: int | None
    adjunct_url: str
    adjunct_type: str

    @property
    def announcement_date(self) -> str:
        return _ms_to_date_str(self.announcement_time_ms)

    @property
    def pdf_url(self) -> str:
        if self.adjunct_url.startswith("http"):
            return self.adjunct_url
        return f"{STATIC_BASE_URL}{self.adjunct_url.lstrip('/')}"

    @property
    def detail_url(self) -> str:
        # 与网页列表一致：detail 链接携带 announcementTime=YYYY-MM-DD
        return (
            "https://www.cninfo.com.cn/new/disclosure/detail"
            f"?stockCode={self.sec_code}"
            f"&announcementId={self.announcement_id}"
            f"&orgId={self.org_id}"
            f"&announcementTime={self.announcement_date}"
        )


@dataclass(frozen=True)
class StockInfo:
    code: str
    org_id: str
    name: str
    category: str = ""
    pinyin: str = ""

    @property
    def stock_param(self) -> str:
        return f"{self.code},{self.org_id}"


def fetch_stock_list_auto(client: HttpClient | None = None, *, url: str = STOCK_LIST_URL) -> list[StockInfo]:
    """
    从 cninfo 公开数据获取上市公司列表。
    源数据：https://www.cninfo.com.cn/new/data/szse_stock.json
    """
    owned_client = client or create_cninfo_client(min_interval_s=0.0)
    headers = {"User-Agent": _UA, "Referer": SEARCH_PAGE_URL}
    resp = owned_client.get(url, headers=headers, timeout_s=30)
    data = resp.json()
    if not isinstance(data, dict):
        raise ValueError(f"unexpected stock list response type: {type(data).__name__}")
    raw_list = data.get("stockList") or []
    if not isinstance(raw_list, list):
        raise ValueError("unexpected stockList format")
    out: list[StockInfo] = []
    seen: set[str] = set()
    for item in raw_list:
        if not isinstance(item, dict):
            continue
        code = str(item.get("code") or "").strip()
        org_id = str(item.get("orgId") or "").strip()
        if not code or not org_id:
            continue
        key = f"{code}|{org_id}"
        if key in seen:
            continue
        seen.add(key)
        out.append(
            StockInfo(
                code=code,
                org_id=org_id,
                name=str(item.get("zwjc") or "").strip(),
                category=str(item.get("category") or "").strip(),
                pinyin=str(item.get("pinyin") or "").strip(),
            )
        )
    return out


def create_cninfo_client(*, min_interval_s: float = 0.4) -> HttpClient:
    session = requests.Session()
    return HttpClient(session=session, rate_limiter=RateLimiter(min_interval_s=float(min_interval_s)))


def bootstrap_session(client: HttpClient) -> None:
    # 访问一次搜索页以获取必要 cookie（JSESSIONID 等）
    client.get(SEARCH_PAGE_URL, headers={"User-Agent": _UA}, timeout_s=30)


def query_his_announcements(
    client: HttpClient,
    *,
    page_num: int,
    page_size: int,
    start_date: str,
    end_date: str,
    searchkey: str = "",
    column: str = "szse",
    tab_name: str = "fulltext",
    plate: str = "",
    stock: str = "",
    secid: str = "",
    category: str = "",
    trade: str = "",
    sort_name: str = "",
    sort_type: str = "",
) -> dict[str, Any]:
    payload = {
        "pageNum": page_num,
        "pageSize": page_size,
        "column": column,
        "tabName": tab_name,
        "plate": plate,
        "stock": stock,
        "searchkey": searchkey,
        "secid": secid,
        "category": category,
        "trade": trade,
        "seDate": f"{start_date}~{end_date}",
        "sortName": sort_name,
        "sortType": sort_type,
        "isHLtitle": "true",
    }
    headers = _default_headers(SEARCH_PAGE_URL)
    resp = client.post_form(HIS_ANNOUNCEMENT_QUERY_URL, data=payload, headers=headers, timeout_s=30)
    data = resp.json()
    if not isinstance(data, dict):
        raise ValueError(f"unexpected cninfo response type: {type(data).__name__}")
    return data


def iter_his_announcements(
    client: HttpClient,
    *,
    start_date: str,
    end_date: str,
    searchkey: str = "",
    stock: str = "",
    column: str = "szse",
    page_size: int = 30,
    max_pages: int | None = None,
    retry_max: int = 0,
    retry_sleep_base_s: float = 2.0,
    retry_sleep_max_s: float = 60.0,
) -> Iterable[Announcement]:
    page_num = 1
    total_pages: int | None = None

    while True:
        if max_pages is not None and page_num > max_pages:
            return

        attempt = 0
        while True:
            try:
                data = query_his_announcements(
                    client,
                    page_num=page_num,
                    page_size=page_size,
                    start_date=start_date,
                    end_date=end_date,
                    searchkey=searchkey,
                    stock=stock,
                    column=column,
                )
                break
            except Exception as exc:  # noqa: BLE001
                if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                    raise
                attempt += 1
                if retry_max != 0 and attempt > retry_max:
                    raise
                base = max(0.0, float(retry_sleep_base_s))
                cap = max(base, float(retry_sleep_max_s))
                delay = min(cap, base * (2 ** max(0, attempt - 1))) + random.random() * 0.5
                logger.warning(
                    "检索请求失败 keyword=%s page=%s attempt=%s delay=%.1fs err=%s",
                    searchkey,
                    page_num,
                    attempt,
                    delay,
                    str(exc)[:200],
                )
                time.sleep(delay)

        if total_pages is None:
            tp = data.get("totalpages")
            if isinstance(tp, int) and tp > 0:
                total_pages = tp
            else:
                total = int(data.get("totalRecordNum") or 0)
                total_pages = (total + page_size - 1) // page_size if total else 1

        anns = data.get("announcements") or []
        try:
            logger.debug(
                "检索 keyword=%s page=%s/%s range=%s~%s 返回=%s",
                searchkey,
                page_num,
                total_pages,
                start_date,
                end_date,
                len(anns) if isinstance(anns, list) else 0,
            )
        except Exception:
            pass
        for ann in anns:
            yield Announcement(
                sec_code=str(ann.get("secCode") or ""),
                sec_name=str(ann.get("secName") or ""),
                org_id=str(ann.get("orgId") or ""),
                announcement_id=str(ann.get("announcementId") or ""),
                title=str(ann.get("announcementTitle") or ""),
                announcement_time_ms=int(ann.get("announcementTime") or 0) or None,
                adjunct_url=str(ann.get("adjunctUrl") or ""),
                adjunct_type=str(ann.get("adjunctType") or ""),
            )

        if page_num >= (total_pages or 1):
            return

        has_more = data.get("hasMore")
        if has_more is False:
            return

        page_num += 1


def download_pdf_bytes(client: HttpClient, pdf_url: str) -> bytes:
    resp = client.get(pdf_url, headers={"User-Agent": _UA, "Referer": SEARCH_PAGE_URL}, timeout_s=60, stream=True)
    return resp.content
