from __future__ import annotations

import base64
import json
import logging
import multiprocessing as mp
import os
import re
import threading
from hashlib import sha256
from dataclasses import dataclass
from pathlib import Path

import requests

from ..integrations.cninfo_api import Announcement
from ..core.extract_wealth_mgmt import PurchaseRecord, amount_to_yuan, detect_product_type, normalize_date
from ..integrations.http_client import RateLimiter
from ..parsing.pdf_parser import PdfContent
from ..parsing.table_utils import clean_cell


_CODE_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)
_PREFILTER_MAX_CHARS = 8000
_ALLOWED_PRODUCT_TYPES = (
    "结构性存款",
    "理财产品",
    "现金管理",
    "收益凭证",
    "国债逆回购",
    "大额存单",
    "通知存款",
    "定期存款",
    "协定存款",
)
_SKIP_AMOUNT_TERMS = (
    # 授权/额度/计划类（非单笔交易金额）
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
    # 非交易/公司事项
    "注册资本",
    "资本金",
    "获准开业",
    "开业",
    "批复",
    "成立",
    "设立",
)
_SKIP_NAME_TERMS = (
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
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LlmConfig:
    base_url: str
    api_key: str
    model: str = "gpt-5.1-codex-max"
    reasoning_effort: str = "medium"
    # codex_api_proxy 的上游（ChatGPT Codex）要求提供一份较长的 instructions 模板；
    # 可从 codex_api_proxy/backend/_debug/last_openai_instructions.txt 获取，或自行指定文件路径。
    instructions: str | None = None
    timeout_s: float = 120.0
    max_input_chars: int = 50_000
    concurrency: int = 1
    min_interval_s: float = 0.0
    ocr_enabled: bool = True
    ocr_min_text_chars: int = 200
    ocr_max_pages: int = 2
    ocr_safe: bool = True


@dataclass(frozen=True)
class LlmExtractResult:
    relevant: bool
    records: list[PurchaseRecord]


def _load_instructions_from_file(path: Path) -> str | None:
    try:
        if not path.exists():
            return None
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return None
    text = (text or "").strip()
    return text or None


def _auto_detect_instructions() -> str | None:
    """
    尝试在常见的 codex_api_proxy 项目结构下，自动找到上游需要的 instructions 模板。
    失败时返回 None；调用方可通过 CNINFO_LLM_INSTRUCTIONS_FILE 显式指定。
    """
    try:
        here = Path(__file__).resolve()
        repo_root = None
        for parent in here.parents:
            if parent.name == "src":
                repo_root = parent.parent
                break
        if repo_root is None:
            repo_root = here.parents[3]
        masterpiece_root = repo_root.parents[2]
        candidates = [
            masterpiece_root / "System" / "codex_api_proxy" / "backend" / "_debug" / "last_openai_instructions.txt",
            masterpiece_root / "System" / "codex_api_proxy" / "_debug" / "last_openai_instructions.txt",
        ]
        for candidate in candidates:
            val = _load_instructions_from_file(candidate)
            if val:
                return val
    except Exception:
        pass
    # 最后兜底：使用本项目内置的默认 instructions（避免部分代理要求 instructions 时直接报错）。
    try:
        val = _load_instructions_from_file(Path(__file__).with_name("llm_instructions_default.txt"))
        if val:
            return val
    except Exception:
        pass
    return None


def _needs_ocr(pdf: PdfContent, *, min_text_chars: int) -> bool:
    text = (pdf.text or "").strip()
    return len(text) < max(0, int(min_text_chars))


def _render_pdf_images(pdf_bytes: bytes, *, max_pages: int) -> list[str]:
    try:
        import fitz  # PyMuPDF
    except Exception:
        logger.warning("OCR 回退未启用：缺少 PyMuPDF（pip install PyMuPDF）")
        return []
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception as exc:
        logger.warning("OCR 回退未启用：PDF 打开失败 %s", exc)
        return []

    images: list[str] = []
    try:
        limit = max(1, int(max_pages))
        for page in doc:
            if len(images) >= limit:
                break
            pix = page.get_pixmap(dpi=150, alpha=False)
            img_bytes = pix.tobytes("png")
            b64 = base64.b64encode(img_bytes).decode("ascii")
            images.append(f"data:image/png;base64,{b64}")
    finally:
        try:
            doc.close()
        except Exception:
            pass
    return images


def _render_pdf_images_worker(pdf_bytes: bytes, max_pages: int, q: mp.Queue) -> None:
    try:
        q.put(_render_pdf_images(pdf_bytes, max_pages=max_pages))
    except Exception:
        q.put([])


def _render_pdf_images_safe(pdf_bytes: bytes, *, max_pages: int, timeout_s: float = 30.0) -> list[str]:
    ctx = mp.get_context("spawn")
    q: mp.Queue = ctx.Queue()
    p = ctx.Process(target=_render_pdf_images_worker, args=(pdf_bytes, max_pages, q), daemon=True)
    p.start()
    p.join(timeout_s)
    if p.is_alive():
        p.terminate()
        p.join()
        logger.warning("OCR 渲染超时，已终止子进程")
        return []
    if p.exitcode not in (0, None):
        logger.warning("OCR 子进程异常退出 exitcode=%s", p.exitcode)
    try:
        if not q.empty():
            return q.get_nowait() or []
    except Exception:
        return []
    return []


def _build_input_payload(prompt: str, image_urls: list[str]) -> object:
    if not image_urls:
        return prompt
    content: list[dict] = [{"type": "input_text", "text": prompt}]
    for url in image_urls:
        content.append({"type": "input_image", "image_url": url})
    return [{"type": "message", "role": "user", "content": content}]


def load_llm_config_from_env(
    *,
    base_url: str | None = None,
    model: str | None = None,
    reasoning_effort: str | None = None,
    instructions_file: str | None = None,
    timeout_s: float | None = None,
    max_input_chars: int | None = None,
    concurrency: int | None = None,
    min_interval_s: float | None = None,
    ocr_enabled: bool | None = None,
    ocr_min_text_chars: int | None = None,
    ocr_max_pages: int | None = None,
) -> LlmConfig | None:
    """
    从环境变量读取 LLM 配置，避免把 key 写入代码/仓库。

    必需：
      - CNINFO_LLM_API_KEY

    可选：
      - CNINFO_LLM_BASE_URL（默认 https://api.gpteam.abrdns.com/v1）
      - CNINFO_LLM_MODEL（默认 gpt-5.1-codex-max）
      - CNINFO_LLM_EFFORT（默认 medium）
      - CNINFO_LLM_MAX_CHARS（默认 50000）
      - CNINFO_LLM_OCR（默认 1，图片型 PDF 走 OCR）
      - CNINFO_LLM_OCR_MIN_TEXT_CHARS（默认 200）
      - CNINFO_LLM_OCR_MAX_PAGES（默认 2）
    """
    api_key = (os.environ.get("CNINFO_LLM_API_KEY") or "").strip()
    if not api_key:
        return None

    instructions: str | None = None
    if (instructions_file or "").strip():
        instructions = _load_instructions_from_file(Path(instructions_file))
    else:
        env_file = (os.environ.get("CNINFO_LLM_INSTRUCTIONS_FILE") or "").strip()
        if env_file:
            instructions = _load_instructions_from_file(Path(env_file).expanduser())
        if not instructions:
            instructions = _auto_detect_instructions()

    return LlmConfig(
        base_url=(base_url or os.environ.get("CNINFO_LLM_BASE_URL") or "https://api.gpteam.abrdns.com/v1").strip(),
        api_key=api_key,
        model=(model or os.environ.get("CNINFO_LLM_MODEL") or "gpt-5.1-codex-max").strip(),
        reasoning_effort=(reasoning_effort or os.environ.get("CNINFO_LLM_EFFORT") or "medium").strip(),
        instructions=instructions,
        timeout_s=float(timeout_s if timeout_s is not None else float(os.environ.get("CNINFO_LLM_TIMEOUT_S") or 120.0)),
        max_input_chars=int(max_input_chars if max_input_chars is not None else int(os.environ.get("CNINFO_LLM_MAX_CHARS") or 50000)),
        concurrency=int(concurrency if concurrency is not None else int(os.environ.get("CNINFO_LLM_CONCURRENCY") or 1)),
        min_interval_s=float(min_interval_s if min_interval_s is not None else float(os.environ.get("CNINFO_LLM_MIN_INTERVAL_S") or 0.0)),
        ocr_enabled=bool(
            (ocr_enabled if ocr_enabled is not None else (os.environ.get("CNINFO_LLM_OCR") or "1"))
            .strip()
            .lower()
            not in ("0", "false", "off", "no")
        ),
        ocr_min_text_chars=int(
            ocr_min_text_chars
            if ocr_min_text_chars is not None
            else int(os.environ.get("CNINFO_LLM_OCR_MIN_TEXT_CHARS") or 200)
        ),
        ocr_max_pages=int(
            ocr_max_pages if ocr_max_pages is not None else int(os.environ.get("CNINFO_LLM_OCR_MAX_PAGES") or 2)
        ),
        ocr_safe=bool((os.environ.get("CNINFO_LLM_OCR_SAFE") or "1").strip().lower() not in ("0", "false", "off", "no")),
    )


def _truncate(s: str, max_chars: int) -> str:
    if max_chars <= 0:
        return ""
    s = s or ""
    return s if len(s) <= max_chars else s[: max_chars - 20] + "\n...[truncated]..."


def _tables_preview(tables: list[list[list[str]]], *, max_tables: int = 3, max_rows: int = 18, max_cols: int = 8) -> str:
    parts: list[str] = []
    for ti, table in enumerate(tables[: max(0, max_tables)]):
        if not table:
            continue
        rows = [[clean_cell(c) for c in r[: max(1, max_cols)]] for r in table[: max(1, max_rows)] if r]
        if not rows:
            continue
        parts.append(f"[表{ti+1}]")
        for r in rows:
            parts.append(" | ".join([c for c in r if c]))
    return "\n".join(parts).strip()


def _get_output_text(resp_json: dict) -> str:
    """
    兼容 Responses API 常见返回格式，尽量提取出最终文本。
    """
    if isinstance(resp_json.get("output_text"), str):
        return resp_json["output_text"].strip()

    out = resp_json.get("output")
    if isinstance(out, list):
        texts: list[str] = []
        for item in out:
            if not isinstance(item, dict):
                continue
            content = item.get("content")
            if not isinstance(content, list):
                continue
            for c in content:
                if not isinstance(c, dict):
                    continue
                if c.get("type") in ("output_text", "text"):
                    t = c.get("text")
                    if isinstance(t, str) and t.strip():
                        texts.append(t.strip())
        if texts:
            return "\n".join(texts).strip()

    # 兜底：把整个 JSON 序列化返回，便于定位
    return json.dumps(resp_json, ensure_ascii=False)


def _get_reasoning_summary(resp_json: dict) -> str:
    """
    尝试从 Responses API 返回中提取 reasoning summary（注意：不是 chain-of-thought）。
    若无，则返回空字符串。
    """
    out = resp_json.get("output")
    if not isinstance(out, list):
        return ""
    for item in out:
        if not isinstance(item, dict):
            continue
        if item.get("type") != "reasoning":
            continue
        summary = item.get("summary")
        if isinstance(summary, str):
            return summary.strip()
        if isinstance(summary, list):
            parts: list[str] = []
            for x in summary:
                if isinstance(x, str) and x.strip():
                    parts.append(x.strip())
            return "\n".join(parts).strip()
    return ""


def _redact_response_json(resp_json: dict, *, instructions: str | None) -> dict:
    """
    dump 模式下落盘用：避免把超长 instructions 重复写入文件/日志。
    """
    out = dict(resp_json)
    if "instructions" in out:
        out["instructions"] = "<redacted>"
    if isinstance(instructions, str) and instructions:
        out["_client_instructions_len"] = len(instructions)
        out["_client_instructions_sha256"] = sha256(instructions.encode("utf-8", errors="replace")).hexdigest()
    return out


def _parse_json_object(text: str) -> dict:
    s = (text or "").strip()
    s = _CODE_FENCE_RE.sub("", s).strip()
    # 只取第一个 JSON object
    start = s.find("{")
    end = s.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("LLM 输出不是 JSON object")
    blob = s[start : end + 1]
    return json.loads(blob)


def _compact_text(s: str) -> str:
    return re.sub(r"\s+", "", (s or "")).strip()


def _looks_like_non_trade_record(*, product_name: str, purchase_amount: str) -> bool:
    """
    过滤 LLM 误把“额度/注册资本/统计行”等当作购买记录的情况。
    """
    s = _compact_text(f"{product_name} {purchase_amount}")
    return any(t in s for t in _SKIP_AMOUNT_TERMS) or any(t in s for t in _SKIP_NAME_TERMS)


def _normalize_product_type(raw_type: str, *, context_text: str) -> str:
    raw = (raw_type or "").strip()
    compact = re.sub(r"\s+", "", raw)
    for t in _ALLOWED_PRODUCT_TYPES:
        t_compact = t.replace(" ", "")
        if t_compact and t_compact in compact:
            if t != "理财产品":
                return t
            break
    detected = detect_product_type(context_text)
    if detected and detected != "理财产品":
        return detected
    if raw:
        if raw in _ALLOWED_PRODUCT_TYPES:
            return raw
        return "理财产品"
    return detected or ""


class ResponsesApiClient:
    def __init__(self, cfg: LlmConfig) -> None:
        self._cfg = cfg
        self._thread_local = threading.local()
        self._limiter = RateLimiter(min_interval_s=float(cfg.min_interval_s), jitter_s=0.05)
        self._sem = threading.Semaphore(max(1, int(cfg.concurrency)))

    @property
    def base_url(self) -> str:
        return self._cfg.base_url

    def _get_session(self) -> requests.Session:
        s = getattr(self._thread_local, "session", None)
        if s is not None:
            return s
        s = requests.Session()
        self._thread_local.session = s
        return s

    def create(self, *, input_text: str | None = None, input_payload: object | None = None) -> dict:
        url = self._cfg.base_url.rstrip("/") + "/responses"
        headers = {
            "Authorization": f"Bearer {self._cfg.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        payload_input = input_payload if input_payload is not None else (input_text or "")
        payload = {
            "model": self._cfg.model,
            "input": payload_input,
            "reasoning": {"effort": self._cfg.reasoning_effort},
        }
        if isinstance(self._cfg.instructions, str) and self._cfg.instructions.strip():
            payload["instructions"] = self._cfg.instructions
        with self._sem:
            self._limiter.wait()
            resp = self._get_session().post(url, headers=headers, json=payload, timeout=float(self._cfg.timeout_s))
            resp.raise_for_status()
            return resp.json()

    def list_models(self) -> dict:
        url = self._cfg.base_url.rstrip("/") + "/models"
        headers = {
            "Authorization": f"Bearer {self._cfg.api_key}",
            "Accept": "application/json",
        }
        with self._sem:
            self._limiter.wait()
            resp = self._get_session().get(url, headers=headers, timeout=float(self._cfg.timeout_s))
            resp.raise_for_status()
            return resp.json()


class WealthMgmtLlmExtractor:
    def __init__(
        self,
        cfg: LlmConfig,
        *,
        trace_mode: str = "off",  # off/brief/dump
        trace_dir: Path = Path("output/llm_traces"),
        trace_max_chars: int = 1200,
    ) -> None:
        self._cfg = cfg
        self._client = ResponsesApiClient(cfg)
        self._trace_mode = (trace_mode or "off").strip().lower()
        if self._trace_mode not in {"off", "brief", "dump"}:
            raise ValueError(f"unsupported trace_mode: {self._trace_mode}")
        self._trace_dir = trace_dir
        self._trace_max_chars = max(200, int(trace_max_chars))

    def prefilter(
        self,
        ann: Announcement,
        pdf: PdfContent,
        *,
        pdf_bytes: bytes | None = None,
        attempt: int = 0,
    ) -> bool:
        title = re.sub(r"<.*?>", "", ann.title or "").strip()
        text = (pdf.text or "").replace("\u3000", " ")
        tables = _tables_preview(pdf.tables)
        image_urls: list[str] = []
        if self._cfg.ocr_enabled and pdf_bytes and _needs_ocr(pdf, min_text_chars=self._cfg.ocr_min_text_chars):
            if self._cfg.ocr_safe:
                image_urls = _render_pdf_images_safe(pdf_bytes, max_pages=1)
            else:
                image_urls = _render_pdf_images(pdf_bytes, max_pages=1)
            if image_urls:
                logger.info("LLM 预筛 OCR 回退启用 id=%s pages=%s", ann.announcement_id, len(image_urls))

        ocr_tip = "注意：PDF 可能为图片，请优先从图片内容判断是否为理财交易公告。\n" if image_urls else ""
        prompt = f"""你是公告预筛助手。请根据公司/标题/公告日期及 PDF 第一页内容，判断该公告是否为“理财交易公告”（包含购买/认购/申购/续购/赎回/到期赎回等具体交易）。

输出要求（必须严格遵守）：
- 只输出 JSON，不要输出任何解释/注释/Markdown
- JSON 格式：{{\"relevant\":true}} 或 {{\"relevant\":false}}
- 若仅为授权额度/审议/制度/计划/账户管理/声明等非交易公告，请输出 relevant=false
{ocr_tip}
公告信息：
- 公司：{ann.sec_name}（{ann.sec_code}）
- 标题：{title}
- 公告日期：{ann.announcement_date}

PDF 第一页表格预览：{tables or "(none)"}

PDF 第一页正文（已截断）：
{text or "(none)"}
"""
        before_len = len(prompt)
        prompt = _truncate(prompt, min(int(self._cfg.max_input_chars), _PREFILTER_MAX_CHARS))
        after_len = len(prompt)

        prefix = f"{ann.announcement_id}_prefilter_attempt{int(attempt)}"
        prompt_path = self._trace_dir / f"{prefix}_prompt.txt"
        resp_path = self._trace_dir / f"{prefix}_response.json"
        out_text_path = self._trace_dir / f"{prefix}_output_text.txt"
        reasoning_path = self._trace_dir / f"{prefix}_reasoning_summary.txt"

        if self._trace_mode in {"brief", "dump"}:
            logger.info("[LLM] 预筛请求 id=%s attempt=%s prompt_len=%s send_len=%s", ann.announcement_id, attempt, before_len, after_len)
            preview = prompt[: self._trace_max_chars]
            if preview:
                logger.info("[LLM] 预筛prompt预览(前%s字)\n%s", self._trace_max_chars, preview)
        if self._trace_mode == "dump":
            try:
                self._trace_dir.mkdir(parents=True, exist_ok=True)
                prompt_path.write_text(prompt, encoding="utf-8")
                logger.info("[LLM] 预筛prompt已落盘：%s", str(prompt_path))
            except Exception as exc:
                logger.warning("[LLM] 预筛prompt落盘失败：%s", exc)

        resp_json = self._client.create(input_payload=_build_input_payload(prompt, image_urls))
        if self._trace_mode == "dump":
            try:
                self._trace_dir.mkdir(parents=True, exist_ok=True)
                redacted = _redact_response_json(resp_json, instructions=self._cfg.instructions)
                resp_path.write_text(json.dumps(redacted, ensure_ascii=False, indent=2), encoding="utf-8")
                logger.info("[LLM] 预筛response已落盘：%s", str(resp_path))
            except Exception as exc:
                logger.warning("[LLM] 预筛response落盘失败：%s", exc)

        out_text = _get_output_text(resp_json)
        reasoning_summary = _get_reasoning_summary(resp_json)
        if self._trace_mode in {"brief", "dump"}:
            if reasoning_summary:
                preview = reasoning_summary[: self._trace_max_chars]
                logger.info("[LLM] 预筛reasoning_summary(前%s字)\n%s", self._trace_max_chars, preview)
            preview = out_text[: self._trace_max_chars]
            if preview:
                logger.info("[LLM] 预筛输出预览(前%s字)\n%s", self._trace_max_chars, preview)
        if self._trace_mode == "dump":
            try:
                self._trace_dir.mkdir(parents=True, exist_ok=True)
                out_text_path.write_text(out_text, encoding="utf-8")
                logger.info("[LLM] 预筛output_text已落盘：%s", str(out_text_path))
            except Exception as exc:
                logger.warning("[LLM] 预筛output_text落盘失败：%s", exc)
            try:
                if reasoning_summary:
                    self._trace_dir.mkdir(parents=True, exist_ok=True)
                    reasoning_path.write_text(reasoning_summary, encoding="utf-8")
                    logger.info("[LLM] 预筛reasoning_summary已落盘：%s", str(reasoning_path))
            except Exception as exc:
                logger.warning("[LLM] 预筛reasoning_summary落盘失败：%s", exc)

        data = _parse_json_object(out_text)
        relevant = data.get("relevant")
        if not isinstance(relevant, bool):
            raise ValueError("LLM prefilter output missing 'relevant' boolean")
        return bool(relevant)

    def extract(
        self,
        ann: Announcement,
        pdf: PdfContent,
        *,
        pdf_bytes: bytes | None = None,
        rule_snippet: str = "",
        attempt: int = 0,
    ) -> LlmExtractResult:
        title = re.sub(r"<.*?>", "", ann.title or "").strip()
        text = (pdf.text or "").replace("\u3000", " ")
        tables = _tables_preview(pdf.tables)
        image_urls: list[str] = []
        if self._cfg.ocr_enabled and pdf_bytes and _needs_ocr(pdf, min_text_chars=self._cfg.ocr_min_text_chars):
            if self._cfg.ocr_safe:
                image_urls = _render_pdf_images_safe(pdf_bytes, max_pages=self._cfg.ocr_max_pages)
            else:
                image_urls = _render_pdf_images(pdf_bytes, max_pages=self._cfg.ocr_max_pages)
            if image_urls:
                logger.info("LLM OCR 回退启用 id=%s pages=%s", ann.announcement_id, len(image_urls))

        ocr_tip = "注意：PDF 可能为图片，请优先从图片内容识别表格/金额。\n" if image_urls else ""
        prompt = f"""你是信息抽取助手。请先判断公告是否为“理财交易公告”（包含购买/认购/申购/续购/赎回/到期赎回等具体交易），再抽取交易记录。

输出要求（必须严格遵守）：
- 只输出 JSON，不要输出任何解释、注释或 Markdown。
- JSON 格式固定为：{{\"relevant\":true,\"records\":[{{\"product_name\":\"\",\"product_type\":\"\",\"purchase_amount\":\"\",\"purchase_date\":\"\"}}]}}
- relevant: 仅当公告包含理财产品的具体交易记录时为 true；若仅为授权额度/审议公告/制度说明/账户注销等非交易类公告则为 false。
- 若 relevant=false，则 records 必须为空数组。
- 每条 record 的 purchase_amount 必须包含数字；金额不明确就不要输出该条。
- 不要输出授权额度/上限/统计指标/注册资本等“非交易记录”（例如：最高额度不超过55亿元、总理财额度100000万元、注册资本20亿元）。
- product_type 尽量归一到以下之一：结构性存款/理财产品/现金管理/收益凭证/国债逆回购/大额存单/通知存款/定期存款/协定存款；无法判断可留空。
- purchase_date 只在公告中明确出现购买/认购/起息等日期时填写；否则留空（系统会用公告日期兜底）。
- 不要臆造；若确定为理财交易公告但未找到明确购买记录，则返回 {{\"relevant\":true,\"records\":[]}}。
{ocr_tip}

提取提示：
- 优先从表格/正文中的“产品名称/产品类型/购买金额/起息日/购买日”等字段抽取。
- 如果提供了图片（OCR 回退），请以图片中内容为准。
- 金额保留原文格式（如“1,000.00万元”“6000万元”）。

公告信息：
- 公司：{ann.sec_name}（{ann.sec_code}）
- 标题：{title}
- 公告日期：{ann.announcement_date}

规则抽取参考（可能不完整）：
{rule_snippet or "(none)"}

PDF 表格预览：
{tables or "(none)"}

PDF 正文（已截断）：
{text or "(none)"}
"""
        before_len = len(prompt)
        prompt = _truncate(prompt, int(self._cfg.max_input_chars))
        after_len = len(prompt)

        prefix = f"{ann.announcement_id}_attempt{int(attempt)}"
        prompt_path = self._trace_dir / f"{prefix}_prompt.txt"
        resp_path = self._trace_dir / f"{prefix}_response.json"
        out_text_path = self._trace_dir / f"{prefix}_output_text.txt"
        reasoning_path = self._trace_dir / f"{prefix}_reasoning_summary.txt"

        if self._trace_mode in {"brief", "dump"}:
            logger.info("[LLM] 准备请求 id=%s attempt=%s prompt_len=%s send_len=%s", ann.announcement_id, attempt, before_len, after_len)
            preview = prompt[: self._trace_max_chars]
            if preview:
                logger.info("[LLM] prompt预览(前%s字)\n%s", self._trace_max_chars, preview)
        if self._trace_mode == "dump":
            try:
                self._trace_dir.mkdir(parents=True, exist_ok=True)
                prompt_path.write_text(prompt, encoding="utf-8")
                logger.info("[LLM] prompt已落盘：%s", str(prompt_path))
            except Exception as exc:
                logger.warning("[LLM] prompt落盘失败：%s", exc)

        try:
            resp_json = self._client.create(input_payload=_build_input_payload(prompt, image_urls))
        except Exception as exc:  # noqa: BLE001
            status = None
            body = ""
            if isinstance(exc, requests.HTTPError) and getattr(exc, "response", None) is not None:
                try:
                    status = exc.response.status_code  # type: ignore[union-attr]
                    body = str(getattr(exc.response, "text", "") or "")  # type: ignore[union-attr]
                except Exception:
                    pass
            if self._trace_mode in {"brief", "dump"}:
                snippet = (body or "")[: self._trace_max_chars]
                logger.error(
                    "[LLM] 请求失败 id=%s attempt=%s status=%s err=%s%s",
                    ann.announcement_id,
                    attempt,
                    status,
                    str(exc),
                    f"\n[LLM] error_body(前{self._trace_max_chars}字)：\n{snippet}" if snippet else "",
                )
            if self._trace_mode == "dump":
                try:
                    self._trace_dir.mkdir(parents=True, exist_ok=True)
                    (self._trace_dir / f"{prefix}_error.txt").write_text(
                        (f"status={status}\n\n" if status is not None else "") + (body or str(exc)),
                        encoding="utf-8",
                    )
                    logger.info("[LLM] error已落盘：%s", str(self._trace_dir / f"{prefix}_error.txt"))
                except Exception as dump_exc:
                    logger.warning("[LLM] error落盘失败：%s", dump_exc)
            raise
        if self._trace_mode == "dump":
            try:
                self._trace_dir.mkdir(parents=True, exist_ok=True)
                redacted = _redact_response_json(resp_json, instructions=self._cfg.instructions)
                resp_path.write_text(json.dumps(redacted, ensure_ascii=False, indent=2), encoding="utf-8")
                logger.info("[LLM] response已落盘：%s", str(resp_path))
            except Exception as exc:
                logger.warning("[LLM] response落盘失败：%s", exc)

        out_text = _get_output_text(resp_json)
        reasoning_summary = _get_reasoning_summary(resp_json)
        if self._trace_mode in {"brief", "dump"}:
            if reasoning_summary:
                preview = reasoning_summary[: self._trace_max_chars]
                logger.info("[LLM] reasoning_summary(前%s字)\n%s", self._trace_max_chars, preview)
            preview = out_text[: self._trace_max_chars]
            if preview:
                logger.info("[LLM] 输出预览(前%s字)\n%s", self._trace_max_chars, preview)
        if self._trace_mode == "dump":
            try:
                self._trace_dir.mkdir(parents=True, exist_ok=True)
                out_text_path.write_text(out_text, encoding="utf-8")
                logger.info("[LLM] output_text已落盘：%s", str(out_text_path))
            except Exception as exc:
                logger.warning("[LLM] output_text落盘失败：%s", exc)
            try:
                if reasoning_summary:
                    self._trace_dir.mkdir(parents=True, exist_ok=True)
                    reasoning_path.write_text(reasoning_summary, encoding="utf-8")
                    logger.info("[LLM] reasoning_summary已落盘：%s", str(reasoning_path))
            except Exception as exc:
                logger.warning("[LLM] reasoning_summary落盘失败：%s", exc)
        data = _parse_json_object(out_text)
        relevant = data.get("relevant")
        if not isinstance(relevant, bool):
            raise ValueError("LLM output missing 'relevant' boolean")
        if not relevant:
            return LlmExtractResult(relevant=False, records=[])
        records = data.get("records")
        if not isinstance(records, list):
            return LlmExtractResult(relevant=True, records=[])

        out: list[PurchaseRecord] = []
        for r in records[:200]:
            if not isinstance(r, dict):
                continue
            product_name = str(r.get("product_name") or "").strip()
            product_type = str(r.get("product_type") or "").strip()
            purchase_amount = str(r.get("purchase_amount") or "").strip()
            purchase_date_raw = str(r.get("purchase_date") or "").strip()

            if not purchase_amount or not re.search(r"\d", purchase_amount):
                continue
            if _looks_like_non_trade_record(product_name=product_name, purchase_amount=purchase_amount):
                if self._trace_mode in {"brief", "dump"}:
                    logger.info(
                        "[LLM] 跳过疑似非交易记录 id=%s 金额=%s 产品=%s",
                        ann.announcement_id,
                        purchase_amount,
                        product_name,
                    )
                continue

            if not product_type:
                product_type = detect_product_type(f"{product_name} {title}")
            product_type = _normalize_product_type(
                product_type,
                context_text=f"{product_name} {title}",
            )

            purchase_date = normalize_date(purchase_date_raw)
            if not purchase_date:
                purchase_date = ann.announcement_date

            out.append(
                PurchaseRecord(
                    announcement_url=ann.detail_url,
                    pdf_url=ann.pdf_url,
                    company_name=ann.sec_name,
                    sec_code=ann.sec_code,
                    announcement_title=title,
                    announcement_date=ann.announcement_date,
                    product_name=product_name,
                    product_type=product_type,
                    purchase_amount=purchase_amount,
                    purchase_amount_yuan=amount_to_yuan(purchase_amount),
                    purchase_date=purchase_date,
                    extract_source="llm",
                    snippet=f"llm | {product_name} | {product_type} | {purchase_amount}"[:400],
                )
            )
        if self._trace_mode in {"brief", "dump"}:
            for r in out[:10]:
                logger.info(
                    "[LLM] 解析记录 id=%s 金额=%s 产品=%s 类型=%s",
                    ann.announcement_id,
                    r.purchase_amount,
                    (r.product_name or "").strip(),
                    (r.product_type or "").strip(),
                )
        return LlmExtractResult(relevant=True, records=out)
