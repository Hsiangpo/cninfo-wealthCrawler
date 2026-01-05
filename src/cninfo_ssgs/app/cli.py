from __future__ import annotations

import argparse
import json
import logging
import os
from datetime import datetime
from pathlib import Path

from ..core.crawler import crawl_wealth_management
from ..data.stock_list import build_stock_list, save_stock_list
from ..llm.llm_wealth_mgmt import ResponsesApiClient, load_llm_config_from_env
from ..utils.logging_utils import setup_logging


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cninfo-ssgs", description="CNINFO 2025 理财公告采集与字段抽取")
    sub = parser.add_subparsers(dest="command", required=True)

    crawl = sub.add_parser("crawl", help="采集并抽取理财购买公告")
    crawl.add_argument("--start", default="2025-01-01", help="开始日期，格式 YYYY-MM-DD")
    crawl.add_argument("--end", default="2025-12-31", help="结束日期，格式 YYYY-MM-DD")
    crawl.add_argument(
        "--keywords",
        nargs="*",
        default=["理财", "委托理财", "现金管理", "结构性存款", "收益凭证", "大额存单", "国债逆回购", "协定存款", "通知存款", "定期存款"],
        help="标题关键字（多个用空格分隔）",
    )
    crawl.add_argument(
        "--stock-source",
        choices=["off", "auto", "file"],
        default="off",
        help="按公司列表拉取公告：auto=cninfo 股票列表；file=自定义列表；off=关键词模式",
    )
    crawl.add_argument("--stock-list", type=Path, default=None, help="公司列表文件路径（stock-source=file 时必填）")
    crawl.add_argument("--stock-limit", type=int, default=None, help="公司列表模式下，最多处理前 N 家公司")
    crawl.add_argument("--column", default="szse", help="cninfo column 参数，深沪京一般使用 szse")
    crawl.add_argument("--page-size", type=int, default=30, help="每页数量（cninfo 默认 30）")
    crawl.add_argument("--workers", type=int, default=4, help="下载/解析并发数（建议 2-6）")
    crawl.add_argument("--min-interval", type=float, default=0.4, help="全局最小请求间隔秒数（限速）")
    crawl.add_argument(
        "--date-chunk",
        choices=["auto", "none", "month"],
        default="auto",
        help="日期切分：auto=区间>45天按月切分；month=按月切分；none=不切分",
    )
    crawl.add_argument("--max-pages", type=int, default=None, help="最多抓取每个关键词的页数（调试用）")
    crawl.add_argument("--max-announcements", type=int, default=None, help="最多处理公告数量（调试用）")
    crawl.add_argument("--max-pdf-pages", type=int, default=10, help="最多解析 PDF 前 N 页")
    crawl.add_argument("--schema", choices=["prd", "full"], default="prd", help="输出字段：prd=老板PRD字段；full=包含调试字段")
    crawl.add_argument("--output", type=Path, default=None, help="输出 CSV 路径（默认随 schema 自动选择）")
    crawl.add_argument("--processed-ids", type=Path, default=None, help="断点续跑：已处理公告ID列表（默认随 schema 自动选择）")
    crawl.add_argument("--pdf-dir", type=Path, default=Path("cache/pdfs"), help="PDF 缓存目录")
    crawl.add_argument("--errors", type=Path, default=None, help="错误日志输出（JSONL，默认随 schema 自动选择）")
    crawl.add_argument("--force", action="store_true", help="强制重新处理（忽略断点与缓存）")
    crawl.add_argument("--log-level", default="INFO", help="日志级别：DEBUG/INFO/WARNING/ERROR（默认 INFO）")
    crawl.add_argument("--log-file", type=Path, default=None, help="日志文件路径（可选，例如 output/20250101_120000/crawl.log）")
    crawl.add_argument("--log-announcements", action="store_true", help="打印每条公告的处理信息（会很啰嗦）")
    crawl.add_argument(
        "--llm",
        choices=["off", "fallback", "always"],
        default="off",
        help="LLM 辅助抽取（OpenAI Responses API）：off=关闭；fallback=规则抽取失败才调用；always=每个公告都调用",
    )
    crawl.add_argument("--llm-base-url", default=None, help="LLM Base URL（如 http://127.0.0.1:8888/v1；默认读 CNINFO_LLM_BASE_URL）")
    crawl.add_argument("--llm-model", default=None, help="LLM model（默认读 CNINFO_LLM_MODEL）")
    crawl.add_argument("--llm-effort", choices=["low", "medium", "high"], default=None, help="LLM reasoning.effort（默认读 CNINFO_LLM_EFFORT）")
    crawl.add_argument(
        "--llm-instructions-file",
        default=None,
        help="LLM instructions 模板文件路径（默认读 CNINFO_LLM_INSTRUCTIONS_FILE；使用 codex_api_proxy 时可指向 backend/_debug/last_openai_instructions.txt）",
    )
    crawl.add_argument("--llm-timeout", type=float, default=120.0, help="LLM 请求超时秒数（默认 120）")
    crawl.add_argument("--llm-max-chars", type=int, default=12000, help="发送给 LLM 的最大文本字符数（截断）")
    crawl.add_argument("--llm-concurrency", type=int, default=1, help="LLM 并发（建议 1-2）")
    crawl.add_argument("--llm-min-interval", type=float, default=0.0, help="LLM 最小请求间隔秒数（限速）")
    crawl.add_argument("--llm-trace", choices=["off", "brief", "dump"], default="off", help="LLM 过程日志：off/brief/dump（dump 会落盘 prompt/response）")
    crawl.add_argument("--llm-trace-dir", type=Path, default=None, help="LLM trace 落盘目录（默认 output/{timestamp}/llm_traces）")
    crawl.add_argument("--run-dir", type=Path, default=None, help="本次运行输出根目录（默认 output/{timestamp}）")
    crawl.add_argument("--llm-trace-max-chars", type=int, default=1200, help="控制台打印的 LLM prompt/输出最大字符数（默认 1200）")
    crawl.add_argument(
        "--llm-prefilter",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="LLM 预筛：只看公司+标题+PDF第一页决定是否理财（stock-source!=off 时默认启用）",
    )
    crawl.add_argument("--retry-max", type=int, default=10, help="失败重试次数（每条公告）。0=无限重试（可能跑很久）")
    crawl.add_argument("--retry-sleep-base", type=float, default=2.0, help="失败重试基础等待秒数（指数退避 base）")
    crawl.add_argument("--retry-sleep-max", type=float, default=60.0, help="失败重试最大等待秒数（指数退避上限）")

    stock = sub.add_parser("stock-list", help="生成本地上市公司列表（可选过滤退市）")
    stock.add_argument("--output", type=Path, default=None, help="输出文件路径（.json/.csv）")
    stock.add_argument(
        "--filter-active",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="是否按交易所在线列表过滤（建议保留）",
    )
    stock.add_argument(
        "--include-b-share",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="是否包含 B 股",
    )
    stock.add_argument(
        "--include-cdr",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="是否包含 CDR",
    )
    stock.add_argument("--min-interval", type=float, default=0.2, help="最小请求间隔（秒）")
    stock.add_argument("--log-level", default="INFO", help="日志级别：DEBUG/INFO/WARNING/ERROR（默认 INFO）")
    stock.add_argument("--log-file", type=Path, default=None, help="日志文件路径（可选）")

    check = sub.add_parser("llm-check", help="探活本地/远程 LLM（OpenAI Responses API）")
    check.add_argument("--base-url", default=None, help="LLM Base URL（默认读 CNINFO_LLM_BASE_URL）")
    check.add_argument("--model", default=None, help="LLM model（默认读 CNINFO_LLM_MODEL）")
    check.add_argument("--effort", choices=["low", "medium", "high"], default=None, help="reasoning.effort（默认读 CNINFO_LLM_EFFORT）")
    check.add_argument("--instructions-file", default=None, help="instructions 模板文件路径（默认读 CNINFO_LLM_INSTRUCTIONS_FILE）")
    check.add_argument("--timeout", type=float, default=30.0, help="请求超时秒数（默认 30）")
    check.add_argument("--log-level", default="INFO", help="日志级别：DEBUG/INFO/WARNING/ERROR（默认 INFO）")
    check.add_argument("--log-file", type=Path, default=None, help="日志文件路径（可选）")

    return parser


def main(argv: list[str] | None = None) -> int:
    logger = logging.getLogger(__name__)
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "crawl":
        setup_logging(level=str(args.log_level), log_file=args.log_file)
        os.environ["CNINFO_LLM_BASE_URL"] = "https://api.gpteam.abrdns.com/v1"
        os.environ["CNINFO_LLM_MODEL"] = "gpt-5.1-codex-mini"
        os.environ["CNINFO_LLM_EFFORT"] = "medium"
        os.environ["CNINFO_LLM_MAX_CHARS"] = "50000"
        os.environ["CNINFO_LLM_OCR"] = "1"
        os.environ["CNINFO_LLM_OCR_MAX_PAGES"] = "5"
        if not os.environ.get("CNINFO_LLM_API_KEY"):
            logger.warning("CNINFO_LLM_API_KEY 未设置，请通过环境变量提供（不要写入仓库）。")
        fixed_run_dir = Path(r"D:\Develop\Masterpiece\Spider\Website\cninfo_SSGS\output\20260103_full_8w")
        run_dir = fixed_run_dir
        args.workers = 2
        args.llm_concurrency = 2
        args.llm = "always"
        args.llm_prefilter = True
        args.retry_max = 0
        args.stock_source = "file"
        args.stock_list = Path(r"D:\Develop\Masterpiece\Spider\Website\cninfo_SSGS\output\stock_lists\stock_list_active.json")
        data_dir = run_dir / "data"
        if args.schema == "prd":
            default_output = data_dir / "results_2025_prd.csv"
            default_processed = data_dir / "processed_ids_prd.txt"
            default_errors = data_dir / "errors_prd.jsonl"
        else:
            default_output = data_dir / "results_2025_full.csv"
            default_processed = data_dir / "processed_ids_full.txt"
            default_errors = data_dir / "errors_full.jsonl"
        default_llm_trace_dir = run_dir / "llm_traces"

        try:
            crawl_wealth_management(
                start_date=args.start,
                end_date=args.end,
                keywords=list(args.keywords),
                stock_source=str(args.stock_source),
                stock_list_path=args.stock_list,
                stock_limit=args.stock_limit,
                column=args.column,
                page_size=args.page_size,
                workers=args.workers,
                min_interval_s=args.min_interval,
                date_chunk=str(args.date_chunk),
                max_pages=args.max_pages,
                max_announcements=args.max_announcements,
                max_pdf_pages=args.max_pdf_pages,
                output_csv=args.output or default_output,
                processed_ids_path=args.processed_ids or default_processed,
                pdf_dir=args.pdf_dir,
                errors_path=args.errors or default_errors,
                force=bool(args.force),
                schema=str(args.schema),
                llm_mode=str(args.llm),
                llm_base_url=args.llm_base_url,
                llm_model=args.llm_model,
                llm_effort=args.llm_effort,
                llm_instructions_file=args.llm_instructions_file,
                llm_timeout_s=float(args.llm_timeout),
                llm_max_input_chars=int(args.llm_max_chars),
                llm_concurrency=int(args.llm_concurrency),
                llm_min_interval_s=float(args.llm_min_interval),
                llm_trace=str(args.llm_trace),
                llm_trace_dir=args.llm_trace_dir or default_llm_trace_dir,
                llm_trace_max_chars=int(args.llm_trace_max_chars),
                llm_prefilter=args.llm_prefilter,
                log_announcements=bool(args.log_announcements),
                retry_max=int(args.retry_max),
                retry_sleep_base_s=float(args.retry_sleep_base),
                retry_sleep_max_s=float(args.retry_sleep_max),
            )
            return 0
        except KeyboardInterrupt:
            logger.warning("任务中断：收到 KeyboardInterrupt")
            return 130
        except Exception:
            logger.exception("任务异常退出")
            return 1

    if args.command == "stock-list":
        setup_logging(level=str(args.log_level), log_file=args.log_file)
        out_path = args.output or Path("output") / "stock_lists" / "stock_list_active.json"
        try:
            items, stats = build_stock_list(
                filter_active=bool(args.filter_active),
                include_b_share=bool(args.include_b_share),
                include_cdr=bool(args.include_cdr),
                min_interval_s=float(args.min_interval),
            )
            save_stock_list(out_path, items)
            logger.info(
                "stock list saved: %s total=%s filtered=%s",
                str(out_path),
                stats.get("total", 0),
                stats.get("filtered", len(items)),
            )
            return 0
        except Exception:
            logger.exception("生成公司列表失败")
            return 1

    if args.command == "llm-check":
        setup_logging(level=str(args.log_level), log_file=args.log_file)
        cfg = load_llm_config_from_env(
            base_url=args.base_url,
            model=args.model,
            reasoning_effort=args.effort,
            instructions_file=args.instructions_file,
            timeout_s=float(args.timeout),
        )
        if cfg is None:
            print("缺少 CNINFO_LLM_API_KEY：请用环境变量设置（避免泄露密钥）")
            return 2

        client = ResponsesApiClient(cfg)
        print(f"base_url: {cfg.base_url}")
        print(f"model: {cfg.model}")
        print(f"effort: {cfg.reasoning_effort}")
        print(f"instructions: {'loaded' if cfg.instructions else 'missing'}")
        if cfg.instructions:
            print(f"instructions_len: {len(cfg.instructions)}")

        models_ok = False
        try:
            models = client.list_models()
            ids: list[str] = []
            data = models.get("data") if isinstance(models, dict) else None
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict) and isinstance(item.get("id"), str):
                        ids.append(item["id"])
            print(f"/models ok, sample: {ids[:8]}")
            models_ok = True
        except Exception as exc:  # noqa: BLE001
            # 一些代理/网关会禁用 /models（403），但 /responses 仍然可用；因此这里只提示，不提前退出。
            print(f"/models failed (ignored): {exc}")

        try:
            resp = client.create(input_text="hello")
            out_text = ""
            if isinstance(resp.get("output_text"), str):
                out_text = resp["output_text"].strip()
            print("/responses ok")
            if out_text:
                print("output_text(sample):", out_text[:120])
            else:
                print("response_keys:", list(resp.keys())[:12] if isinstance(resp, dict) else type(resp).__name__)
        except Exception as exc:  # noqa: BLE001
            print(f"/responses failed: {exc}")
            resp_obj = getattr(exc, "response", None)
            try:
                if resp_obj is not None:
                    status = getattr(resp_obj, "status_code", None)
                    ct = ""
                    try:
                        ct = str(getattr(resp_obj, "headers", {}).get("content-type", ""))
                    except Exception:
                        ct = ""
                    body = ""
                    try:
                        body = str(getattr(resp_obj, "text", "") or "")
                    except Exception:
                        body = ""
                    if status is not None:
                        print("status:", status)
                    if ct:
                        print("content-type:", ct)
                    if body:
                        print("body_preview:", body[:800])
                        if "Instructions are required" in body or "Instructions are not valid" in body:
                            print(
                                "提示：你连接的是 codex_api_proxy，上游会校验 instructions。"
                                "需要使用由 Codex CLI capture 生成的 last_openai_instructions.txt，"
                                "并设置 CNINFO_LLM_INSTRUCTIONS_FILE 指向该文件。"
                            )
            except Exception:
                pass
            return 1

        if not models_ok:
            print("提示：/models 被禁用并不影响抓取；只要 /responses 正常即可。")
        return 0

    parser.error("未知命令")
    return 2
