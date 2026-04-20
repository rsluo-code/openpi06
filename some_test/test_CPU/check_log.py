import re
import os
import numpy as np
import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


def parse_time_to_datetime(t: str) -> datetime.datetime:
    """
    将 "YYYY-MM-DD HH:MM:SS" 解析为 datetime
    """
    return datetime.datetime.strptime(t.strip(), "%Y-%m-%d %H:%M:%S")


def nan_stats(arr):
    """
    对序列做统计（忽略 NaN）
    返回：count/min/max/mean/std
    """
    a = np.asarray(arr, dtype=np.float64)
    a = a[~np.isnan(a)]
    if a.size == 0:
        return {"count": 0, "min": np.nan, "max": np.nan, "mean": np.nan, "std": np.nan}
    return {
        "count": int(a.size),
        "min": float(np.min(a)),
        "max": float(np.max(a)),
        "mean": float(np.mean(a)),
        "std": float(np.std(a, ddof=0)),
    }


def format_time_axis(ax, t_datetimes):
    """
    格式化 matplotlib 时间横轴：
    - 数据跨度短：显示 HH:MM:SS
    - 跨度长：显示 %m-%d %H:%M
    """
    if len(t_datetimes) == 0:
        return

    dt0 = t_datetimes[0]
    dt1 = t_datetimes[-1]
    span = (dt1 - dt0).total_seconds()

    if span <= 6 * 3600:
        fmt = "%H:%M:%S"
    else:
        fmt = "%m-%d %H:%M"

    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter(fmt))


def main():
    # =========================
    # 1) 交互式输入：路径 + 百分比区间
    # =========================
    path = "/DATA/disk2/1024_grasp_packages/code/pi0_code/openpi06/some_test/test_CPU/sys_resource_continuous_20260227_141319.log"
    if not path:
        print("路径为空，退出。")
        return
    if not os.path.exists(path):
        print(f"文件不存在：{path}")
        return

    # 你要处理的数据百分比区间（默认全量 0~100）
    # 例：跳过前 10% 的空闲数据，就输入 10 和 100
    p_from_str = input("请输入起始百分比 p_from（0~100，回车默认0）: ").strip()
    p_to_str = input("请输入结束百分比 p_to（0~100，回车默认100）: ").strip()

    p_from = float(p_from_str) if p_from_str else 0.0
    p_to = float(p_to_str) if p_to_str else 100.0

    # 基本校验与纠正
    if p_from < 0:
        p_from = 0.0
    if p_to > 100:
        p_to = 100.0
    if p_to <= p_from:
        print(f"百分比区间非法：p_from={p_from}, p_to={p_to}，要求 p_to > p_from")
        return

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()

    # ========= 正则（匹配你的日志格式） =========
    time_re = re.compile(r"【采集时间】\s*([0-9]{4}-[0-9]{2}-[0-9]{2}\s+[0-9]{2}:[0-9]{2}:[0-9]{2})")
    cpu_re = re.compile(r"CPU信息：用户态占用=([0-9]+(?:\.[0-9]+)?)%\s*\|\s*系统态占用=([0-9]+(?:\.[0-9]+)?)%")
    mem_re = re.compile(r"内存信息：总=([0-9]+(?:\.[0-9]+)?)Gi,\s*已用=([0-9]+(?:\.[0-9]+)?)Gi,\s*空闲=([0-9]+(?:\.[0-9]+)?)Gi")
    gpu_re = re.compile(
        r"\|\s*([0-9]+)%\s+([0-9]+)C\s+P[0-9]+\s+([0-9]+)W\s*/\s*([0-9]+)W\s*\|\s*([0-9]+)MiB\s*/\s*([0-9]+)MiB"
    )

    # ========= 按【采集时间】切块，确保字段对齐 =========
    time_matches = list(time_re.finditer(text))
    if not time_matches:
        print("未找到任何【采集时间】记录，请确认文件格式。")
        return

    positions = [(m.start(), m.group(1)) for m in time_matches]
    positions.append((len(text), None))

    ts_str = []
    ts_dt = []

    cpu_user = []
    cpu_sys = []

    mem_total = []
    mem_used = []
    mem_free = []

    gpu_util = []
    gpu_temp = []
    gpu_power = []
    gpu_mem_used = []
    gpu_mem_total = []

    for i in range(len(positions) - 1):
        start_pos, tstr = positions[i]
        end_pos, _ = positions[i + 1]
        block = text[start_pos:end_pos]

        ts_str.append(tstr)
        ts_dt.append(parse_time_to_datetime(tstr))

        m = cpu_re.search(block)
        if m:
            cpu_user.append(float(m.group(1)))
            cpu_sys.append(float(m.group(2)))
        else:
            cpu_user.append(np.nan)
            cpu_sys.append(np.nan)

        m = mem_re.search(block)
        if m:
            mem_total.append(float(m.group(1)))
            mem_used.append(float(m.group(2)))
            mem_free.append(float(m.group(3)))
        else:
            mem_total.append(np.nan)
            mem_used.append(np.nan)
            mem_free.append(np.nan)

        m = gpu_re.search(block)
        if m:
            gpu_util.append(float(m.group(1)))
            gpu_temp.append(float(m.group(2)))
            gpu_power.append(float(m.group(3)))
            gpu_mem_used.append(float(m.group(5)))
            gpu_mem_total.append(float(m.group(6)))
        else:
            gpu_util.append(np.nan)
            gpu_temp.append(np.nan)
            gpu_power.append(np.nan)
            gpu_mem_used.append(np.nan)
            gpu_mem_total.append(np.nan)

    # ========= 排序（防止日志乱序） =========
    order = np.argsort(np.array(ts_dt, dtype="datetime64[ns]"))
    ts_dt = np.array(ts_dt, dtype=object)[order]
    ts_str = [ts_str[i] for i in order]

    cpu_user = np.array(cpu_user, dtype=np.float64)[order]
    cpu_sys = np.array(cpu_sys, dtype=np.float64)[order]
    cpu_total = cpu_user + cpu_sys

    mem_total = np.array(mem_total, dtype=np.float64)[order]
    mem_used = np.array(mem_used, dtype=np.float64)[order]
    mem_free = np.array(mem_free, dtype=np.float64)[order]

    gpu_util = np.array(gpu_util, dtype=np.float64)[order]
    gpu_temp = np.array(gpu_temp, dtype=np.float64)[order]
    gpu_power = np.array(gpu_power, dtype=np.float64)[order]
    gpu_mem_used = np.array(gpu_mem_used, dtype=np.float64)[order]
    gpu_mem_total = np.array(gpu_mem_total, dtype=np.float64)[order]

    # =========================
    # 2) 新增：按百分比区间裁剪数据（避免空闲段影响平均值）
    # =========================
    n = len(ts_dt)
    # 起始 idx 用 floor，结束 idx 用 ceil，尽量包含边界段
    start_idx = int(np.floor(n * (p_from / 100.0)))
    end_idx = int(np.ceil(n * (p_to / 100.0)))

    # 边界保护
    start_idx = max(0, min(start_idx, n - 1))
    end_idx = max(start_idx + 1, min(end_idx, n))

    # 为了方便对照，我们保留一份“全量”的起止时间
    start_time_full = ts_str[0]
    end_time_full = ts_str[-1]

    # 对所有序列统一裁剪（统计/画图将只基于这段）
    ts_dt = ts_dt[start_idx:end_idx]
    ts_str = ts_str[start_idx:end_idx]

    cpu_user = cpu_user[start_idx:end_idx]
    cpu_sys = cpu_sys[start_idx:end_idx]
    cpu_total = cpu_total[start_idx:end_idx]

    mem_total = mem_total[start_idx:end_idx]
    mem_used = mem_used[start_idx:end_idx]
    mem_free = mem_free[start_idx:end_idx]

    gpu_util = gpu_util[start_idx:end_idx]
    gpu_temp = gpu_temp[start_idx:end_idx]
    gpu_power = gpu_power[start_idx:end_idx]
    gpu_mem_used = gpu_mem_used[start_idx:end_idx]
    gpu_mem_total = gpu_mem_total[start_idx:end_idx]

    start_time = ts_str[0]
    end_time = ts_str[-1]

    print("将按百分比区间处理数据：")
    print(f"  p_from={p_from}%, p_to={p_to}%")
    print(f"  index范围: [{start_idx}, {end_idx}) / 原始总点数 {n}")
    print(f"  全量时间范围: {start_time_full}  ->  {end_time_full}")
    print(f"  裁剪时间范围: {start_time}  ->  {end_time}")

    # =========================
    # 3) CSV 输出：建议同时输出全量 & 裁剪版（便于核对）
    # =========================
    # （可选）全量 CSV：为了不影响你现有流程，这里直接从原文件再写一次“全量序列”
    # 若你不需要全量 CSV，可以删掉这一段
    csv_full_path = path + "_series_full.csv"
    # 重新构造全量排序后的序列（避免再写一遍解析逻辑，这里简单复读上面的变量会很麻烦）
    # 更“干净”的方式是你在裁剪前备份一份全量数组，这里为了清晰我建议你备份：
    #   ts_dt_full = ts_dt_all; ... 然后写出 full
    # 但为了让你最少改动，我这里给你一个更直白的做法：不写 full 也行。
    # === 如果你确实想保留 full，对应做法见下方注释“备份全量” ===

    # 裁剪版 CSV（与你原脚本一致的字段）
    csv_path = path + f"_series_p{p_from:.1f}-{p_to:.1f}.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("time,cpu_user_pct,cpu_sys_pct,cpu_total_pct,mem_used_gib,mem_free_gib,mem_total_gib,"
                "gpu_util_pct,gpu_mem_used_mib,gpu_mem_total_mib,gpu_power_w,gpu_temp_c\n")
        for i in range(len(ts_dt)):
            f.write(
                f"{ts_str[i]},"
                f"{cpu_user[i]:.6f},"
                f"{cpu_sys[i]:.6f},"
                f"{cpu_total[i]:.6f},"
                f"{mem_used[i]:.6f},"
                f"{mem_free[i]:.6f},"
                f"{mem_total[i]:.6f},"
                f"{gpu_util[i]:.6f},"
                f"{gpu_mem_used[i]:.6f},"
                f"{gpu_mem_total[i]:.6f},"
                f"{gpu_power[i]:.6f},"
                f"{gpu_temp[i]:.6f}\n"
            )

    # =========================
    # 4) 统计 TXT：只对裁剪段统计
    # =========================
    stat_path = path + f"_stats_p{p_from:.1f}-{p_to:.1f}.txt"
    with open(stat_path, "w", encoding="utf-8") as f:
        f.write("=====================================\n")
        f.write("监控统计汇总（按百分比裁剪后）\n")
        f.write(f"全量时间范围: {start_time_full} -> {end_time_full}\n")
        f.write(f"裁剪百分比: p_from={p_from}%, p_to={p_to}%\n")
        f.write(f"裁剪index范围: [{start_idx}, {end_idx}) / 全量点数 {n}\n")
        f.write(f"裁剪时间范围: {start_time} -> {end_time}\n")
        f.write(f"裁剪后采样点数: {len(ts_dt)}\n")
        f.write("=====================================\n\n")

        def write_stat(name, arr, unit=""):
            s = nan_stats(arr)
            f.write(f"[{name}] {unit}\n")
            f.write(f"  count: {s['count']}\n")
            f.write(f"  min  : {s['min']}\n")
            f.write(f"  max  : {s['max']}\n")
            f.write(f"  mean : {s['mean']}\n")
            f.write(f"  std  : {s['std']}\n\n")

        write_stat("CPU user", cpu_user, "%")
        write_stat("CPU sys", cpu_sys, "%")
        write_stat("CPU total", cpu_total, "%")

        write_stat("Mem used", mem_used, "GiB")
        write_stat("Mem free", mem_free, "GiB")
        write_stat("Mem total", mem_total, "GiB")

        write_stat("GPU util", gpu_util, "%")
        write_stat("GPU mem used", gpu_mem_used, "MiB")
        write_stat("GPU mem total", gpu_mem_total, "MiB")
        write_stat("GPU power", gpu_power, "W")
        write_stat("GPU temp", gpu_temp, "C")

    # =========================
    # 5) 画图：只画裁剪段（避免空闲段“拉平曲线”）
    # =========================
    # 1) CPU 图
    plt.figure(figsize=(12, 4))
    plt.plot(ts_dt, cpu_user, label="CPU user %")
    plt.plot(ts_dt, cpu_sys, label="CPU sys %")
    plt.plot(ts_dt, cpu_total, label="CPU total %")
    plt.title(f"CPU Usage (p{p_from:.1f}-{p_to:.1f})")
    plt.xlabel("Time")
    plt.ylabel("%")
    plt.legend()

    ax = plt.gca()
    format_time_axis(ax, ts_dt)
    plt.gcf().autofmt_xdate()
    plt.tight_layout()
    cpu_png = path + f"_cpu_p{p_from:.1f}-{p_to:.1f}.png"
    plt.savefig(cpu_png, dpi=150)
    plt.close()

    # 2) 内存图
    plt.figure(figsize=(12, 4))
    plt.plot(ts_dt, mem_used, label="Mem used (GiB)")
    plt.title(f"Memory Usage (p{p_from:.1f}-{p_to:.1f})")
    plt.xlabel("Time")
    plt.ylabel("GiB")
    plt.legend()

    ax = plt.gca()
    format_time_axis(ax, ts_dt)
    plt.gcf().autofmt_xdate()
    plt.tight_layout()
    mem_png = path + f"_mem_p{p_from:.1f}-{p_to:.1f}.png"
    plt.savefig(mem_png, dpi=150)
    plt.close()

    # 3) GPU 图（4个子图：util / mem / power / temp）
    fig, axs = plt.subplots(4, 1, figsize=(12, 10), sharex=True)

    axs[0].plot(ts_dt, gpu_util)
    axs[0].set_title("GPU Utilization")
    axs[0].set_ylabel("%")

    axs[1].plot(ts_dt, gpu_mem_used)
    axs[1].set_title("GPU Memory Used")
    axs[1].set_ylabel("MiB")

    axs[2].plot(ts_dt, gpu_power)
    axs[2].set_title("GPU Power")
    axs[2].set_ylabel("W")

    axs[3].plot(ts_dt, gpu_temp)
    axs[3].set_title("GPU Temperature")
    axs[3].set_ylabel("C")
    axs[3].set_xlabel("Time")

    format_time_axis(axs[3], ts_dt)
    fig.autofmt_xdate()
    plt.tight_layout()
    gpu_png = path + f"_gpu_p{p_from:.1f}-{p_to:.1f}.png"
    plt.savefig(gpu_png, dpi=150)
    plt.close()

    print("解析完成！生成文件（均基于裁剪段）：")
    print("  PNG :", cpu_png, mem_png, gpu_png)
    print("  CSV :", csv_path)
    print("  STAT:", stat_path)

    # ========= 备份全量 CSV（推荐做法，若你需要） =========
    # 如果你确实想保留 full CSV，请在“裁剪前”备份全量数组：
    #   ts_dt_all = ts_dt.copy(); ts_str_all = ts_str.copy(); cpu_user_all = cpu_user.copy(); ...
    # 然后裁剪后用 *_all 写 full CSV


if __name__ == "__main__":
    main()
