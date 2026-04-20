#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import re
from collections import defaultdict
from decimal import Decimal, InvalidOperation, getcontext
from pathlib import Path

# 提高 Decimal 精度，避免浮点误差（特别是 value 是小数/科学计数法）
getcontext().prec = 50

KIND_ORDER = ["finetuning_At", "pretrain_At", "value_t", "value_tN"]
FILE_RE = re.compile(r"^lang_(\d+)__(finetuning_At|pretrain_At|value_tN|value_t)\.csv$")

def parse_decimal(s: str) -> Decimal:
    s = s.strip()
    try:
        return Decimal(s)
    except InvalidOperation:
        # 兜底：尝试 float 再转 Decimal（会损失精度，但至少不崩）
        return Decimal(str(float(s)))

def read_histogram_csv(csv_path: Path) -> dict[Decimal, int]:
    """
    读取两列(value, count)的csv，跳过首行表头，返回 {value: count}
    """
    hist: dict[Decimal, int] = defaultdict(int)
    with csv_path.open("r", newline="") as f:
        reader = csv.reader(f)
        # 跳过表头
        try:
            next(reader)
        except StopIteration:
            return dict(hist)

        for row_idx, row in enumerate(reader, start=2):
            if not row or len(row) < 2:
                continue
            v_str, c_str = row[0], row[1]
            if v_str is None or c_str is None:
                continue
            v = parse_decimal(v_str)
            c = int(float(c_str))  # 有些csv可能是"12.0"，这里兼容一下
            if c <= 0:
                continue
            hist[v] += c
    return dict(hist)

def top_percentile_cutoff_value(hist: dict[Decimal, int], top_ratio: float = 0.30):
    """
    hist: {value: count}
    按 value 从大到小排序，按 count 累加，找到累积达到 top_ratio*total 的 cutoff value
    返回 (cutoff_value, total_count, target_count, cum_at_cutoff)
    """
    if not hist:
        return None, 0, 0, 0

    total = sum(hist.values())
    target = total * top_ratio

    # 按 value 从大到小排序
    items = sorted(hist.items(), key=lambda x: x[0], reverse=True)

    cum = 0
    cutoff = items[-1][0]
    for v, c in items:
        cum += c
        if cum >= target:
            cutoff = v
            break
    return cutoff, total, target, cum

def main(root: Path, out_path: Path, top_ratio: float, show_topk: int):
    if not root.exists():
        raise FileNotFoundError(f"root not found: {root}")

    # agg[(j, kind)][value] += count
    agg = defaultdict(lambda: defaultdict(int))

    # 遍历 local_rank_*
    rank_dirs = sorted([p for p in root.iterdir() if p.is_dir() and p.name.startswith("local_rank_")])

    for rdir in rank_dirs:
        for f in rdir.iterdir():
            if not f.is_file() or f.suffix.lower() != ".csv":
                continue
            m = FILE_RE.match(f.name)
            if not m:
                continue
            j = int(m.group(1))
            kind = m.group(2)

            hist = read_histogram_csv(f)
            bucket = agg[(j, kind)]
            for v, c in hist.items():
                bucket[v] += c

    # 收集所有 j
    js = sorted({jk[0] for jk in agg.keys()})

    lines = []
    lines.append(f"Root: {root}")
    lines.append(f"Found local_rank dirs: {len(rank_dirs)}")
    lines.append(f"Top ratio (high-value side): {top_ratio:.2%}")
    lines.append("")

    if not js:
        lines.append("No matching CSV files found.")
        out_path.write_text("\n".join(lines), encoding="utf-8")
        print(f"[OK] report written to: {out_path}")
        return

    for j in js:
        lines.append("=" * 80)
        lines.append(f"j = {j}")
        lines.append("")

        # 每个 kind 单独统计
        for kind in KIND_ORDER:
            hist = agg.get((j, kind), {})
            cutoff, total, target, cum = top_percentile_cutoff_value(hist, top_ratio=top_ratio)

            if cutoff is None:
                lines.append(f"- {kind}: MISSING")
                continue

            # 额外展示一些分布信息
            unique_vals = len(hist)
            max_v = max(hist.keys())
            min_v = min(hist.keys())
            lines.append(
                f"- {kind}: total_count={total}, unique_values={unique_vals}, "
                f"value_range=[{min_v}, {max_v}]"
            )
            lines.append(
                f"  cutoff (top {top_ratio:.0%} reaches) = {cutoff} "
                f"(cum={cum}/{total}={cum/total:.2%}, target={target:.1f})"
            )

            if show_topk > 0:
                top_items = sorted(hist.items(), key=lambda x: x[0], reverse=True)[:show_topk]
                lines.append(f"  top {show_topk} values (desc):")
                for v, c in top_items:
                    lines.append(f"    {v}\tcount={c}")
            lines.append("")

        # 4种类型合并（同一个 j）
        merged = defaultdict(int)
        for kind in KIND_ORDER:
            hist = agg.get((j, kind), {})
            for v, c in hist.items():
                merged[v] += c
        cutoff, total, target, cum = top_percentile_cutoff_value(merged, top_ratio=top_ratio)
        if cutoff is not None:
            lines.append("  [MERGED 4 kinds]")
            lines.append(
                f"  merged_total_count={total}, unique_values={len(merged)}, "
                f"value_range=[{min(merged.keys())}, {max(merged.keys())}]"
            )
            lines.append(
                f"  merged_cutoff (top {top_ratio:.0%} reaches) = {cutoff} "
                f"(cum={cum}/{total}={cum/total:.2%}, target={target:.1f})"
            )
        else:
            lines.append("  [MERGED 4 kinds] EMPTY")
        lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[OK] report written to: {out_path}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=str, help="root path containing local_rank_* dirs")
    ap.add_argument("--out", type=str, default="report.txt", help="output report txt path")
    ap.add_argument("--top_ratio", type=float, default=0.30, help="top ratio on high-value side (default 0.30)")
    ap.add_argument("--show_topk", type=int, default=10, help="also print top-k values (default 10, 0 to disable)")
    args = ap.parse_args()

    main(Path(args.root), Path(args.out), args.top_ratio, args.show_topk)

# python bucket_report.py /wx-mix01/sppro/permanent/yuanzhang10/codes_rsluo/openpi06/z_bucket_csvs/20260204_5item_8dim --out report.txt --top_ratio 0.30 --show_topk 30
