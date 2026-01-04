from __future__ import annotations

import re
from dataclasses import dataclass

from ..integrations.cninfo_api import Announcement
from ..parsing.pdf_parser import PdfContent
from ..parsing.table_utils import clean_cell, normalize_table, table_to_dicts


WEALTH_TERMS = (
    "理财",
    "委托理财",
    "现金管理",
    "结构性存款",
    "收益凭证",
    "国债逆回购",
    "大额存单",
    "协定存款",
    "通知存款",
    "定期存款",
)

# 过滤：一些公告虽然包含“理财”字样，但金额并非“购买/认购/申购/赎回”等交易金额，而是注册资本、
# 授权额度、财务指标等。这里做轻量规则，避免把这些金额写进“购买金额”字段里污染结果。
_SKIP_TEXT_CONTEXT_TERMS = (
    # 授权/额度类
    "不超过",
    "不超",
    "最高",
    "上限",
    "额度",
    "总额度",
    "任一时点",
    "滚动使用",
    "可滚动",
    "拟使用",
    "计划使用",
    "授权",
    "审议通过",
    "决议",
    "有效期",
    # 非交易/公司事项
    "注册资本",
    "资本金",
    "获准开业",
    "开业",
    "批复",
    "成立",
    "设立",
    "增资",
    "出资",
    # 财务指标
    "资产总额",
    "负债总额",
    "净资产",
    "净利润",
    "现金流量净额",
)

_SKIP_TABLE_NAME_TERMS = (
    "最近12个月",
    "单日最高",
    "累计收益",
    "理财额度",
    "总理财额度",
    "尚未使用",
    "目前已使用",
    "净资产",
    "净利润",
    "（%）",
    "%",
)

NEGATIVE_TITLE_PATTERNS = (
    r"额度预计",
    r"授权额度",
    r"额度的公告",
    r"额度调整",
    r"审议通过",
    r"股东大会",
    r"董事会决议",
)

HARD_NEGATIVE_TITLE_PATTERNS = (
    r"核查意见",
    r"法律意见书",
    r"获准开业",
    r"开业",
    r"成立",
    r"设立",
)

POSITIVE_TITLE_PATTERNS = (
    r"购买",
    r"认购",
    r"申购",
    r"续购",
    r"到期赎回",
    r"赎回",
    r"委托理财",
    r"现金管理",
    r"结构性存款",
)


def is_candidate_announcement(title: str) -> bool:
    t = title or ""
    if any(re.search(p, t) for p in HARD_NEGATIVE_TITLE_PATTERNS):
        return False
    if not any(w in t for w in WEALTH_TERMS):
        return False
    if any(re.search(p, t) for p in NEGATIVE_TITLE_PATTERNS) and not any(re.search(p, t) for p in ("购买", "认购", "申购", "赎回")):
        return False
    # 这里偏“召回优先”：只要标题出现理财/现金管理/结构性存款等核心词，且不是明显的授权/审议类公告，就纳入处理。
    return True


# 兼容 PDF 抽取后常见的“2025 年 12 月 30 日”这类带空格形式
_DATE_RE = re.compile(r"(?P<y>20\d{2})\s*[年./-]\s*(?P<m>\d{1,2})\s*[月./-]\s*(?P<d>\d{1,2})\s*日?")
_AMOUNT_RE = re.compile(
    r"(?:人民币|RMB|￥)?\s*(?P<num>\d{1,3}(?:,\d{3})*(?:\.\d+)?|\d+(?:\.\d+)?)\s*(?P<unit>亿元|亿|万元|万|元)",
    re.IGNORECASE,
)
_NUM_ONLY_RE = re.compile(r"\d{1,3}(?:,\d{3})*(?:\.\d+)?|\d+(?:\.\d+)?")


def normalize_date(raw: str) -> str:
    if not raw:
        return ""
    raw = re.sub(r"\s+", "", raw)
    m = _DATE_RE.search(raw)
    if not m:
        return raw.strip()
    y, mo, d = int(m.group("y")), int(m.group("m")), int(m.group("d"))
    return f"{y:04d}-{mo:02d}-{d:02d}"


def amount_to_yuan(raw: str) -> int | None:
    """
    仅处理单一数值 + 单位的金额表达；无法解析则返回 None。
    """
    if not raw:
        return None
    m = _AMOUNT_RE.search(raw)
    if not m:
        return None
    num = float(m.group("num").replace(",", ""))
    unit = m.group("unit")
    if unit in ("亿元", "亿"):
        return int(num * 100_000_000)
    if unit in ("万元", "万"):
        return int(num * 10_000)
    if unit == "元":
        return int(num)
    return None


def amount_to_yuan_with_unit_hint(raw: str, unit_hint: str | None) -> int | None:
    """
    允许金额缺少单位（例如表头已写明“(万元)”）。
    """
    v = amount_to_yuan(raw)
    if v is not None:
        return v

    if not raw:
        return None
    m = _NUM_ONLY_RE.search(raw)
    if not m:
        return None
    num = float(m.group(0).replace(",", ""))
    hint = (unit_hint or "").replace(" ", "")
    if "亿元" in hint or "（亿" in hint or "(亿" in hint:
        return int(num * 100_000_000)
    if "万元" in hint or "（万" in hint or "(万" in hint:
        return int(num * 10_000)
    if "元" in hint:
        return int(num)
    return None


def detect_product_type(text: str) -> str:
    s = text or ""
    s = re.sub(r"\s+", "", s)
    for kw in ("结构性存款", "大额存单", "协定存款", "通知存款", "定期存款", "收益凭证", "国债逆回购"):
        if kw in s:
            return kw
    if "现金管理" in s:
        return "现金管理"
    if "理财" in s:
        return "理财产品"
    return ""


def _compact(s: str) -> str:
    return re.sub(r"\s+", "", (s or "")).strip()


def _skip_text_candidate(snippet: str) -> bool:
    s = _compact(snippet)
    return any(t in s for t in _SKIP_TEXT_CONTEXT_TERMS)


def _skip_table_name(name: str) -> bool:
    s = _compact(name)
    return any(t in s for t in _SKIP_TABLE_NAME_TERMS)


def _choose_key(keys: list[str], contains_any: tuple[str, ...], excludes_any: tuple[str, ...] = ()) -> str | None:
    for k in keys:
        kk = re.sub(r"\s+", "", k or "")
        if any(x in kk for x in contains_any) and not any(x in kk for x in excludes_any):
            return k
    return None


@dataclass(frozen=True)
class PurchaseRecord:
    announcement_url: str
    pdf_url: str
    company_name: str
    sec_code: str
    announcement_title: str
    announcement_date: str
    product_name: str
    product_type: str
    purchase_amount: str
    purchase_amount_yuan: int | None
    purchase_date: str
    extract_source: str  # table/text/fallback
    snippet: str


def extract_records(ann: Announcement, pdf: PdfContent) -> list[PurchaseRecord]:
    records: list[PurchaseRecord] = []
    records.extend(_extract_from_tables(ann, pdf))
    if records:
        return _force_purchase_date_to_announcement_date(records, ann.announcement_date)
    records.extend(_extract_from_text(ann, pdf))
    return _force_purchase_date_to_announcement_date(records, ann.announcement_date)


def _force_purchase_date_to_announcement_date(records: list[PurchaseRecord], announcement_date: str) -> list[PurchaseRecord]:
    """
    业务口径（老板 PRD）：购买日期按“公告日期”计。

    为避免不同公告格式导致日期口径不一致，这里统一将 purchase_date 设置为公告日期，
    即使表格/正文中存在更细的购买/起息日期，也不使用。
    """
    if not records:
        return records
    out: list[PurchaseRecord] = []
    for r in records:
        out.append(
            PurchaseRecord(
                announcement_url=r.announcement_url,
                pdf_url=r.pdf_url,
                company_name=r.company_name,
                sec_code=r.sec_code,
                announcement_title=r.announcement_title,
                announcement_date=r.announcement_date,
                product_name=r.product_name,
                product_type=r.product_type,
                purchase_amount=r.purchase_amount,
                purchase_amount_yuan=r.purchase_amount_yuan,
                purchase_date=announcement_date,
                extract_source=r.extract_source,
                snippet=r.snippet,
            )
        )
    return out


def _extract_from_tables(ann: Announcement, pdf: PdfContent) -> list[PurchaseRecord]:
    out: list[PurchaseRecord] = []
    last_headers: list[str] | None = None

    def looks_like_header_row(first_row_cells: list[str]) -> bool:
        joined = "".join(first_row_cells)
        header_hints = (
            "序号",
            "产品名称",
            "产品类型",
            "认购金额",
            "购买金额",
            "起息日",
            "购买日",
            "购买日期",
            "认购日",
            "受托方",
            "赎回日期",
            "赎回本金",
        )
        return any(h in joined for h in header_hints)

    for table in pdf.tables:
        raw_rows = [[clean_cell(c) for c in r] for r in (table or []) if r]
        if not raw_rows:
            continue

        # 表格可能跨页，第二页起可能不再重复表头；若检测不到表头，则尝试沿用上一张表头
        if looks_like_header_row(raw_rows[0]):
            nt = normalize_table(table)
            if nt:
                last_headers = nt.headers
            rows = table_to_dicts(table)
        elif last_headers and len(raw_rows[0]) == len(last_headers):
            rows = [{last_headers[i]: (r[i] if i < len(r) else "") for i in range(len(last_headers))} for r in raw_rows]
        else:
            rows = table_to_dicts(table)

        if not rows:
            continue
        keys = list(rows[0].keys())

        amount_key = _choose_key(keys, ("购买金额", "认购金额", "投入金额", "投资金额", "金额", "本金"))
        # 先找“购买/认购/起息”等日期；找不到再用“赎回/到期”等作为兜底
        date_key = _choose_key(keys, ("购买日", "购买日期", "认购日", "起息日", "投资日期", "购买时间", "投资时间"))
        if not date_key:
            date_key = _choose_key(keys, ("赎回日期", "到期日", "到期日期", "终止日"))
        type_key = _choose_key(keys, ("产品类型", "类型", "品种"))
        # 产品名称列经常和“受托方/银行名称”混淆，这里做更保守的匹配，宁可缺失也不要错列。
        name_key = _choose_key(keys, ("理财产品名称", "产品名称"), excludes_any=("受托", "银行", "发行", "机构"))
        if not name_key:
            name_key = _choose_key(keys, ("产品",), excludes_any=("产品类型", "类型", "受托", "银行", "发行", "机构", "品种"))
        if not name_key:
            name_key = _choose_key(keys, ("名称",), excludes_any=("受托", "银行", "发行", "机构", "公司", "对方"))

        # 没有金额/日期的表格通常不是购买明细
        if not amount_key and not date_key:
            continue

        # 过滤“键值对”样式的小表（如表头为“赎回金额/22700万元”，内容为“投资种类/资金来源”等）
        def row_has_amount_or_date(r: dict[str, str]) -> bool:
            if amount_key:
                v = (r.get(amount_key) or "").strip()
                if _NUM_ONLY_RE.search(v):
                    return True
            if date_key:
                v = (r.get(date_key) or "").strip()
                if _DATE_RE.search(v):
                    return True
            return False

        if not any(row_has_amount_or_date(r) for r in rows[:5]):
            continue

        for row in rows:
            amount = (row.get(amount_key, "") if amount_key else "").strip()
            raw_date = (row.get(date_key, "") if date_key else "").strip()
            typ = (row.get(type_key, "") if type_key else "").strip()
            name = (row.get(name_key, "") if name_key else "").strip()

            # 合计/总计行通常不是“单笔购买记录”
            if any(re.sub(r"\s+", "", v) in ("合计", "总计") for v in row.values() if isinstance(v, str)):
                continue
            # “最近12个月/理财额度/净资产”等统计行不是购买明细
            if name and _skip_table_name(name):
                continue
            if not amount or not _NUM_ONLY_RE.search(amount):
                # PRD 核心字段需要金额；同时避免把“空行/续行/备注行”误当成记录
                continue

            product_type = detect_product_type(f"{typ} {name} {re.sub(r'<.*?>', '', ann.title)}")
            purchase_date = normalize_date(raw_date)
            amount_yuan = amount_to_yuan_with_unit_hint(amount, amount_key) if amount_key else amount_to_yuan(amount)

            if not any([amount, raw_date, product_type]):
                continue

            out.append(
                PurchaseRecord(
                    announcement_url=ann.detail_url,
                    pdf_url=ann.pdf_url,
                    company_name=ann.sec_name,
                    sec_code=ann.sec_code,
                    announcement_title=re.sub(r"<.*?>", "", ann.title),
                    announcement_date=ann.announcement_date,
                    product_name=name,
                    product_type=product_type,
                    purchase_amount=amount,
                    purchase_amount_yuan=amount_yuan,
                    purchase_date=purchase_date,
                    extract_source="table",
                    snippet=" | ".join([x for x in (name, typ, amount, raw_date) if x])[:400],
                )
            )

    return out


def _extract_from_text(ann: Announcement, pdf: PdfContent) -> list[PurchaseRecord]:
    text = (pdf.text or "").replace("\u3000", " ")
    text = re.sub(r"\s+", " ", text)
    if not text:
        return []

    # PRD 口径：购买时间按公告日期计，因此正文抽取只要能定位“金额”即可（无需强绑定日期）。
    sentences = re.split(r"[。；;\n\r]+", text)
    candidates: list[tuple[str, str]] = []

    for s in sentences:
        s = s.strip()
        if not s:
            continue
        if not any(w in s for w in WEALTH_TERMS):
            continue
        # 跳过“额度/注册资本/财务指标”等非交易金额上下文
        if _skip_text_candidate(s):
            continue
        for am in _AMOUNT_RE.finditer(s):
            candidates.append((s, am.group(0)))
            if len(candidates) >= 20:
                break
        if len(candidates) >= 20:
            break

    if not candidates:
        # 降级：全文前几个金额（不再依赖日期）
        amounts = [m.group(0) for m in _AMOUNT_RE.finditer(text)]
        if not amounts:
            return []
        snippet = text[:300]
        if _skip_text_candidate(snippet):
            return []
        candidates = [(snippet, a) for a in amounts[:3]]
        source = "fallback"
    else:
        source = "text"

    out: list[PurchaseRecord] = []
    seen: set[tuple[str, str]] = set()
    for snippet, amount_raw in candidates:
        key = (amount_raw, snippet)
        if key in seen:
            continue
        seen.add(key)
        product_type = detect_product_type(snippet or ann.title)
        amount_yuan = amount_to_yuan(amount_raw)
        out.append(
            PurchaseRecord(
                announcement_url=ann.detail_url,
                pdf_url=ann.pdf_url,
                company_name=ann.sec_name,
                sec_code=ann.sec_code,
                announcement_title=re.sub(r"<.*?>", "", ann.title),
                announcement_date=ann.announcement_date,
                product_name="",
                product_type=product_type,
                purchase_amount=amount_raw,
                purchase_amount_yuan=amount_yuan,
                purchase_date="",
                extract_source=source,
                snippet=snippet[:400],
            )
        )
    return out
