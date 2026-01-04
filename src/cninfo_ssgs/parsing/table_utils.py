from __future__ import annotations

import re
from dataclasses import dataclass


_EMPTY_VALUES = {"", "无", "-", "—", "N/A", "n/a", "/", "－", "None", "null"}


def clean_cell(cell: object) -> str:
    if cell is None:
        return ""
    s = str(cell).strip()
    s = re.sub(r"\s+", " ", s)
    return s


def _clean_header(cell: object) -> str:
    s = clean_cell(cell)
    return re.sub(r"\s+", "", s)


def _is_subheader_row(row: list[object]) -> bool:
    cleaned = [_clean_header(c) for c in row]
    non_empty = [c for c in cleaned if c and c not in _EMPTY_VALUES]
    if len(non_empty) < max(2, len(cleaned) // 2):
        return False
    no_digit = sum(1 for c in non_empty if not re.search(r"\d", c))
    return (no_digit / max(1, len(non_empty))) >= 0.8


@dataclass(frozen=True)
class NormalizedTable:
    headers: list[str]
    rows: list[list[str]]


def normalize_table(table: list[list[object]]) -> NormalizedTable | None:
    if not table or len(table) < 2:
        return None

    header0 = [_clean_header(c) for c in table[0]]
    max_cols = max(len(r) for r in table[:2])

    header_rows = 1
    has_blank = any(not h for h in header0)
    if has_blank and len(table) >= 3 and _is_subheader_row(table[1]):
        header_rows = 2

    headers: list[str] = []
    if header_rows == 1:
        headers = header0 + [""] * (max_cols - len(header0))
    else:
        header1 = [_clean_header(c) for c in table[1]]
        header1 += [""] * (max_cols - len(header1))
        main = ""
        merged: list[str] = []
        for i in range(max_cols):
            h0 = header0[i] if i < len(header0) else ""
            h1 = header1[i] if i < len(header1) else ""
            if h0 and h0 not in _EMPTY_VALUES:
                main = h0
            if h1 and h1 not in _EMPTY_VALUES:
                merged.append(f"{main}{h1}" if main else h1)
            else:
                merged.append(main)
        headers = merged

    # 去重/补空：防止 dict 覆盖
    seen: dict[str, int] = {}
    uniq_headers: list[str] = []
    for h in headers:
        h = h or "col"
        if h not in seen:
            seen[h] = 1
            uniq_headers.append(h)
        else:
            seen[h] += 1
            uniq_headers.append(f"{h}_{seen[h]}")

    data_rows = table[header_rows:]
    rows: list[list[str]] = []
    for r in data_rows:
        row = [clean_cell(c) for c in r]
        if all((not c) or (c in _EMPTY_VALUES) for c in row):
            continue
        if len(row) < len(uniq_headers):
            row += [""] * (len(uniq_headers) - len(row))
        rows.append(row[: len(uniq_headers)])

    if not rows:
        return None
    return NormalizedTable(headers=uniq_headers, rows=rows)


def table_to_dicts(table: list[list[object]]) -> list[dict[str, str]]:
    nt = normalize_table(table)
    if not nt:
        return []
    out: list[dict[str, str]] = []
    for row in nt.rows:
        out.append({h: row[i] if i < len(row) else "" for i, h in enumerate(nt.headers)})
    return out

