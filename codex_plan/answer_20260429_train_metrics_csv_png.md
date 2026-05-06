PI06 PyTorch training metrics export update

Files changed
- `scripts/train_pytorch_pi06.py`
- `scripts/plot_training_metrics_csv.py`

What was added
- Main-process training now writes one aggregated row per `log_interval` to:
  - `<checkpoint_dir>/monitoring/training_metrics.csv`
- The same aggregated data is rendered to:
  - `<checkpoint_dir>/monitoring/training_metrics.png`

CSV fields
- `timestamp`
- `global_step`
- `interval_steps`
- `elapsed_sec`
- `time_per_step_sec`
- `loss`
- `learning_rate`
- `grad_norm`
- `gpu_mem_allocated_gb`
- `gpu_mem_reserved_gb`
- `gpu_mem_peak_allocated_gb`
- `gpu_mem_peak_reserved_gb`
- `cpu_percent`
- `cpu_mem_used_gb`
- `cpu_mem_total_gb`
- `cpu_mem_percent`
- `process_rss_gb`

PNG panels
- loss
- learning rate
- grad norm
- GPU memory
- CPU percent
- CPU memory
- process RSS

Runtime behavior
- CSV/PNG are updated every `config.log_interval`
- Existing CSV is loaded on resume, so resumed runs continue the same plot
- If `matplotlib` is unavailable, CSV still works and PNG rendering is skipped with a warning

Offline merge / replot
- Use:
  - `python scripts/plot_training_metrics_csv.py --csv <csv1> <csv2> ... --output-png <out.png> --output-csv <merged.csv>`
- The script concatenates runs in order and adds:
  - `source_csv`
  - `merged_step`
