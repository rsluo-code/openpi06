import argparse
import csv
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


FIELDS = [
    "loss",
    "learning_rate",
    "grad_norm",
    "gpu_mem_allocated_gb",
    "gpu_mem_reserved_gb",
    "gpu_mem_peak_allocated_gb",
    "gpu_mem_peak_reserved_gb",
    "cpu_percent",
    "cpu_mem_percent",
    "cpu_mem_used_gb",
    "process_rss_gb",
]


def _parse_value(field: str, value: str):
    if value == "":
        return float("nan")
    if field == "timestamp":
        return value
    if field in ("global_step", "interval_steps"):
        return int(value)
    return float(value)


def load_csv(path: pathlib.Path):
    rows = []
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({key: _parse_value(key, value) for key, value in row.items()})
    return rows


def plot_rows(rows, output_path: pathlib.Path):
    steps = [int(row["merged_step"]) for row in rows]
    metric_specs = [
        ("Loss", [("loss", "loss")]),
        ("Learning Rate", [("learning_rate", "lr")]),
        ("Grad Norm", [("grad_norm", "grad_norm")]),
        (
            "GPU Memory (GB)",
            [
                ("gpu_mem_allocated_gb", "allocated"),
                ("gpu_mem_reserved_gb", "reserved"),
                ("gpu_mem_peak_allocated_gb", "peak_alloc"),
                ("gpu_mem_peak_reserved_gb", "peak_reserved"),
            ],
        ),
        ("CPU Percent", [("cpu_percent", "cpu%")]),
        ("CPU Memory", [("cpu_mem_percent", "mem%"), ("cpu_mem_used_gb", "used_gb")]),
        ("Process RSS (GB)", [("process_rss_gb", "rss_gb")]),
    ]

    fig, axes = plt.subplots(len(metric_specs), 1, figsize=(16, 24), sharex=True)
    fig.suptitle("Merged Training Metrics", fontsize=16)

    for ax, (title, series_specs) in zip(axes, metric_specs, strict=True):
        for field, label in series_specs:
            values = [row.get(field, float("nan")) for row in rows]
            ax.plot(steps, values, label=label, linewidth=1.5)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        if len(series_specs) > 1:
            ax.legend(loc="best", fontsize=9)

    axes[-1].set_xlabel("Merged Step")
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def write_merged_csv(rows, output_path: pathlib.Path):
    fieldnames = list(rows[0].keys())
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", nargs="+", required=True, help="One or more training_metrics.csv files")
    parser.add_argument("--output-png", required=True, help="Output png path")
    parser.add_argument("--output-csv", help="Optional merged csv path")
    args = parser.parse_args()

    merged_rows = []
    step_offset = 0
    for csv_path_str in args.csv:
        csv_path = pathlib.Path(csv_path_str)
        rows = load_csv(csv_path)
        if not rows:
            continue
        local_max_step = 0
        for row in rows:
            merged_row = dict(row)
            merged_row["source_csv"] = str(csv_path)
            merged_row["merged_step"] = int(row["global_step"]) + step_offset
            merged_rows.append(merged_row)
            local_max_step = max(local_max_step, int(row["global_step"]))
        step_offset += local_max_step

    if not merged_rows:
        raise ValueError("No rows loaded from the provided CSV files")

    output_png = pathlib.Path(args.output_png)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    plot_rows(merged_rows, output_png)

    if args.output_csv:
        output_csv = pathlib.Path(args.output_csv)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        write_merged_csv(merged_rows, output_csv)


if __name__ == "__main__":
    main()
