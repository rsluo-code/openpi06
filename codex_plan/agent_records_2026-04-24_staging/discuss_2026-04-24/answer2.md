# 2026-04-24 bucket 汇总脚本

本轮在 `z_bucket_csvs` 下新增了 rank 汇总和百分位统计脚本，用于把 8 卡各自输出的桶计数 CSV 合成一个总结果。

## 新增文件

- `z_bucket_csvs/merge_bucket_csvs.py`
- `z_bucket_csvs/run_merge_bucket_csvs.sh`

## 脚本行为

- 扫描指定结果目录下所有 `local_rank_*` 子目录。
- 合并以下 4 类桶计数 CSV：
  - `lang_24__finetuning_At.csv`
  - `lang_24__pretrain_At.csv`
  - `lang_24__value_t.csv`
  - `lang_24__value_tN.csv`
- 在输入目录下生成 `total_rank/`：
  - 合并后的 4 份总 CSV
  - `merge_manifest.csv`
  - `merged_from_ranks.txt`
  - `top_percent_thresholds.csv`
  - `top_percent_thresholds.txt`

## 百分位定义

- 这里按“从大到小 top 10%、20%、...、90%”计算阈值。
- 也就是对每个指标按 `value` 从大到小累加 `count`，第一次达到目标占比时的 `value` 就是该 top 百分比的阈值。

## 本次实际运行结果

输入目录：

```text
/home/rsluo/codes/openpi06/z_bucket_csvs/20260204_5item_8dim
```

输出目录：

```text
/home/rsluo/codes/openpi06/z_bucket_csvs/20260204_5item_8dim/total_rank
```

合并后 4 个指标总样本数一致，都是：

```text
1792800
```

阈值摘要：

```text
top_percent,finetuning,pretrain,value_t,value_tN
10,22,42,-80,-105
20,12,21,-115,-125
30,2,9,-155,-145
40,-3,1,-185,-175
50,-13,-4,-230,-225
60,-18,-10,-270,-270
70,-28,-21,-320,-325
80,-43,-36,-375,-375
90,-118,-69,-490,-480
```

## 验证

- `python3 -m py_compile z_bucket_csvs/merge_bucket_csvs.py`
- `bash -n z_bucket_csvs/run_merge_bucket_csvs.sh`
- `bash z_bucket_csvs/run_merge_bucket_csvs.sh`

都已通过。
