# cninfo-wealthCrawler

目标：从巨潮资讯（cninfo）批量采集 **2025-01-01 ~ 2025-12-31** 期间“上市公司购买理财/现金管理”等相关公告，并抽取：

- 公告链接（详情页）
- 上市公司名称
- 理财产品名称/类型
- 购买金额
- 购买时间

## 运行环境

- Windows / macOS / Linux
- Python 3.11+

## 安装

建议使用虚拟环境：

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## 快速开始

默认采集 2025 全年，使用关键词集合（可通过参数覆盖），输出 **PRD 5 列** 到 `output/{timestamp}/data/results_2025_prd.csv`，支持断点续跑：

```bash
python run.py crawl
```

更细粒度控制：

```bash
python run.py crawl --start 2025-01-01 --end 2025-12-31 --keywords 理财 委托理财 现金管理 结构性存款 收益凭证 大额存单 --workers 4 --date-chunk auto
```

## 公司列表模式（覆盖更全，成本更高）

如果你希望**按上市公司逐个拉公告**（避免标题无关键词导致漏抓），可启用公司列表模式：

```bash
python run.py crawl --stock-source auto --llm always --llm-prefilter
```

说明：
- `--stock-source auto` 会从 cninfo 股票列表（`szse_stock.json`）加载公司清单
- 系统会先用 LLM 读取**公司名 + 标题 + PDF 第一页**进行预筛，非理财公告直接丢弃
- 只有判定为理财交易公告才会进入完整抽取（OCR + 全文）
- 成本与耗时显著增加，建议配合 `--workers`/`--llm-concurrency` 做限流

### 生成“过滤退市”的公司列表（推荐）

cninfo 的 `szse_stock.json` 不含上市状态字段。若你希望**过滤退市/停牌等非现有上市公司**，可以先生成本地公司列表，再用文件模式跑全量：

```bash
python run.py stock-list --output output/stock_lists/stock_list_active.json
python run.py crawl --stock-source file --stock-list output/stock_lists/stock_list_active.json --llm always --llm-prefilter
```

说明：
- `stock-list` 会拉取交易所在线列表（深交所 A/B 股 + 上交所主板/科创板/B 股）并与 cninfo 股票列表做交集
- 如需包含 CDR：`--include-cdr`（默认不含）
- 输出支持 `.json` 或 `.csv`（扩展名决定格式）

建议首次先做小规模抽样验证：

```bash
python run.py crawl --start 2025-12-30 --end 2025-12-30 --keywords 理财 --max-announcements 20 --workers 2
```

## 输出说明

`--schema prd`（默认）：输出 5 列（字段名为中文，Excel 友好）：

- 公告链接（详情页 URL）
- 上市公司名字
- 理财产品类型（优先填“产品名称”，必要时附带类型）
- 购买金额
- 购买时间

`--schema full`：输出更完整的调试字段（包含 `pdf_url`、`product_name`、抽取来源、片段等）。

说明：默认每次运行会创建一个新的时间戳目录（`output/{timestamp}/`），数据文件在 `output/{timestamp}/data/`。
如需固定输出目录（便于断点续跑），可使用 `--run-dir output/20250101_120000` 指向同一目录。

如遇到 PDF 表格识别失败或公告格式特殊，工具会自动降级为基于正文文本的正则抽取，并在 `extract_source` 字段标记来源（`table`/`text`/`fallback`）。

说明：按老板 PRD 口径，“购买时间”统一按 **公告日期** 计（即使表格/正文里出现更细的购买/起息日期，也不使用）。

说明：`--schema prd` 即使抽取失败也会写入一行，占位值为“未披露”，并将公告上下文写入 `output/{timestamp}/data/errors_prd.jsonl` 便于后续人工或二次处理。

## 全量 AI 抽取（Responses API）

本项目默认**全程由 LLM 输出结果**（不再与规则抽取合并）。只要你的模型服务兼容 OpenAI **Responses API**（支持 `POST /v1/responses`），即可直接使用。

1) 先设置环境变量（不要把 key 写进代码/仓库）：

```bash
set CNINFO_LLM_BASE_URL=https://api.gpteam.abrdns.com/v1
set CNINFO_LLM_API_KEY=sk-...
set CNINFO_LLM_MODEL=gpt-5.1-codex-max
set CNINFO_LLM_EFFORT=medium
set CNINFO_LLM_MAX_CHARS=50000
# 说明：如果你的代理会校验 instructions（例如 codex_api_proxy），需要指向它生成的 last_openai_instructions.txt；
# 否则 /v1/responses 可能返回 400（Instructions are required / Instructions are not valid）。
# 推荐显式设置：
# set CNINFO_LLM_INSTRUCTIONS_FILE=...\codex_api_proxy\backend\_debug\last_openai_instructions.txt
```

2) 运行时开启 LLM：

```bash
python run.py crawl --llm always
```

3) 探活（可选）：

```bash
python run.py llm-check
```

提示：部分代理会禁用 `GET /v1/models`（返回 403），但 `POST /v1/responses` 仍可用；以 llm-check 的 `/responses ok` 为准。

### PDF 三层提取架构（全文 + OCR）

遵循 `cninfo` 项目的“三层解析”策略：  
1) **pypdfium2** 极速文本提取  
2) **pdfplumber** 布局还原 + 表格  
3) **LLM Vision OCR**：当前两层文本过短时，渲染页面图片发给 LLM 识别  

依赖：`PyMuPDF`（已加入 `requirements.txt`，用于渲染 PDF 页面为图片）。

可选环境变量：

- `CNINFO_LLM_OCR=0`：关闭 OCR 回退
- `CNINFO_LLM_OCR_MIN_TEXT_CHARS`：判定“文本过短”的阈值（默认 200）
- `CNINFO_LLM_OCR_MAX_PAGES`：最多渲染页数（默认 2）

参数说明：

- `--llm off|fallback|always`：always 会对每个公告调用一次 LLM（最全但最慢），fallback 只在规则抽取无结果时调用。
- 可调：`--llm-concurrency`（建议 1-2）、`--llm-max-chars`（输入截断）、`--llm-min-interval`（限速）。
- 失败重试：`--retry-max`（每条公告重试次数，`0=无限重试`）、`--retry-sleep-base/--retry-sleep-max`（指数退避）。

### 日志（建议长任务开启）

- 控制台+文件：`--log-file output/20250101_120000/crawl.log`
- 每条公告打印：`--log-announcements`
- 分页/检索细节：`--log-level DEBUG`
- 观察 LLM 在“分析什么”：`--llm-trace brief`（控制台预览）或 `--llm-trace dump`（把每条公告的 prompt/response 落盘到 `output/{timestamp}/llm_traces/`）
- 大区间尽量全量：`--date-chunk auto`（默认，区间>45天按月切分）或显式 `--date-chunk month`

## 注意

- 本工具对 cninfo 请求做了限速与重试，但仍建议合理控制 `--workers`，避免对目标站点造成压力。

## 可选：安装为命令

如果你希望直接用命令行运行（如 `cninfo-ssgs crawl`）：

```bash
pip install -e .
cninfo-ssgs crawl
```
