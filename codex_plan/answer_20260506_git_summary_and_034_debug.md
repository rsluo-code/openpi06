2026-05-06 git 总结与 034 调试记录

这份记录主要总结三部分内容：
- 最新的 034 / 035 PI06 validation 改造；
- 让 034 真正跑通时所做的运行时修复；
- 当前 `git status` 里那些还没有被前面文档完整覆盖的改动分类。

## 1. 034 / 035 的 PI06 validation 改造

涉及文件：
- `validation_1_hand_pi06_imgN.py`
- `validation_1_hand_pi06_imgN_multi.py`
- `034validation_1_hand_pi06.sh`
- `035validation_1_hand_pi06_multi.sh`

本次改动的核心是：
- 把 `validation_1_hand_pi06_imgN.py` 重构成“单条 episode 可复用”的形式；
- 提供统一的 `Args`；
- 提供可复用的 `_eval_single_episode(client, args, episode_dir)`；
- 新增 `validation_1_hand_pi06_imgN_multi.py`，支持多条 episode 批量跑。

`validation_1_hand_pi06_imgN_multi.py` 现在支持：
- `episode_dirs`
- `episode_dirs_file`
- `episode_glob`
- `prompt_types`

脚本侧：
- `034validation_1_hand_pi06.sh` 现在是单条 episode 入口；
- `035validation_1_hand_pi06_multi.sh` 是多条 episode 入口，作用上对应 `025`。

## 2. 034 真正执行时碰到的报错与修复

### 2.1 tyro 命令行参数解析不匹配

最开始 `034validation_1_hand_pi06.sh` 传的是普通参数：
- `--episode-dir`
- `--output-base-dir`
- `--model-time`
- `--model-step`
- `--model-dim`

但 `validation_1_hand_pi06_imgN.py` 当时使用的是：
- `tyro.cli(eval_isaac)`

这种写法会按函数参数形式解析，导致 shell 里这些普通参数不能直接识别，报错为：
- `Unrecognized options`

修复方式：
- 把入口改成直接解析 `Args`：
  - `eval_isaac(tyro.cli(Args))`
- `validation_1_hand_pi06_imgN_multi.py` 也做了同样处理。

这样以后 `034` 和 `035` 都可以直接用普通参数风格，不需要写成 `--args.xxx`。

### 2.2 chmod ptxas 在当前环境是只读路径

原来脚本里有：
- `chmod +x .../ptxas`

但当前环境里这个路径是只读的，所以会报：
- `Read-only file system`

修复方式：
- 改成：
  - `chmod ... 2>/dev/null || true`

这样不会因为这个环境噪音中断脚本。

### 2.3 matplotlib 默认缓存目录不可写

执行时 matplotlib 尝试使用：
- `/home/rsluo/.config/matplotlib`

但该目录在当前执行环境中不可写，所以会落到临时目录并打印警告。

修复方式：
- 在 `034` 和 `035` 中加入：
  - `export MPLCONFIGDIR=/tmp/matplotlib-openpi06`
  - `mkdir -p "$MPLCONFIGDIR"`

这样缓存目录固定可写，运行更干净。

### 2.4 websocket 连接在沙箱内被拦截

在沙箱内直接执行 `034` 时，`WebsocketClientPolicy(args.host, args.port)` 会失败。  
这不是仓库代码问题，而是当前执行环境不允许本地 socket 连接。

也就是说：
- 代码本身已经能工作；
- 只是沙箱里不能直接连你启动的 `031server_pi06.sh`。

### 2.5 最终真实运行结果

修完上面这些问题后，`034validation_1_hand_pi06.sh` 已经能在真实环境下连接 `031server_pi06.sh` 并完成推理。

输出文件为：
- `/home/rsluo/codes/openpi06/z_pi06_output/pi06_20260506_100000_8dim_sf包裹_right_episode_2025-11-07_095835_939_part_0_part_2_part_0.png`

## 3. 当前树里已经存在的训练监控导出功能

涉及文件：
- `scripts/train_pytorch_pi06.py`
- `scripts/plot_training_metrics_csv.py`
- `.gitignore`

当前行为：
- `train_pytorch_pi06.py` 会按 `log_interval` 输出训练监控到：
  - `<checkpoint_dir>/monitoring/training_metrics_<timestamp>.csv`
  - `<checkpoint_dir>/monitoring/training_metrics_<timestamp>.png`
- 训练启动时会把这两个路径直接打印出来。
- 只有主进程会写这些文件。
- 当 stderr 不是 TTY 时，会自动关闭进度条，避免日志里每一步都刷新。

`.gitignore` 里和这项工作相关的新增项是：
- `z_pi06_output/`
- `runs/`

## 4. 当前 `git status` 中可见、但概念上应分开的脚本和路径改动

下面这些改动虽然都在工作区里，但它们和 `034/035` 本身不是同一件事。

### 4.1 `003wx_train_torch_pi06.sh`

这里把：
- `sf_packages_rightarm_20260413`

改成了：
- `sf_packages_rightarm_20260429`

本质上是当前训练实验名切换。

### 4.2 `031server_pi06.sh`

这里把 PI06 server 对应的 checkpoint 改成了：
- `/data0/rsluo/pi06_torch/PI06_pretrain/sf_packages_rightarm_20260429/100000/`

这属于当前 server 所服务的模型路径切换。

### 4.3 `021server_valuenet.sh`

这里把 ValueNet server 对应的 checkpoint 改成了：
- `/data0/rsluo/pi06_torch/value_pretrain_16dim/sf_packages_rightarm_20260428/75000`

这同样属于当前运行时路径切换。

### 4.4 `src/openpi/models_pytorch/some_func.py`

这里把 task 24 的元信息从：
- `700 / 0.007`

改成了：
- `200 / 0.001`

这会影响 PI06 的实际运行行为，不只是注释或路径。  
因此它不应该被当作纯脚本改动看待，而应该明确判断它是：
- PI06 runtime tuning
或
- 当前实验的本地超参调整

这部分在前面的 `codex_plan` 文档里还没有被单独说明。

### 4.5 `src/openpi/models/tokenizer_print_test.py`

这里的改动更像本地调试：
- tokenizer 路径改到了本地文件；
- 测试文本改成了新的字符串；
- 很多 prompt 组合被注释掉。

它看起来不是主流程改动，更像单独的 tokenizer / prompt 调试文件。

## 5. 推荐的 commit 分组方式

不建议把当前所有改动一次性混成一个 commit。  
更合理的拆分方式如下：

### 5.1 PI06 / ValueNet 图像输入与 image_keys 主线改动

这一组包括：
- 四路图像接入；
- `image_keys` 可配置；
- dataset / policy / model / validation 的一致性处理；
- 与 `episode_first_head_img` 相关的主线改动。

### 5.2 PI06 训练监控导出

这一组包括：
- `scripts/train_pytorch_pi06.py`
- `scripts/plot_training_metrics_csv.py`
- `.gitignore`

也就是：
- CSV 导出
- PNG 总览图导出
- 非交互式日志下关闭进度条

### 5.3 PI06 imgN validation 重构

这一组包括：
- `validation_1_hand_pi06_imgN.py`
- `validation_1_hand_pi06_imgN_multi.py`
- `034validation_1_hand_pi06.sh`
- `035validation_1_hand_pi06_multi.sh`

也就是：
- 单条 episode 重构
- 多条 episode 支持
- 真实运行时的参数解析和环境兼容修复

### 5.4 当前实验脚本 / checkpoint 路径切换

这一组包括：
- `003wx_train_torch_pi06.sh`
- `021server_valuenet.sh`
- `031server_pi06.sh`

这类改动本质是：
- 训练脚本当前实验名
- server 当前服务的 checkpoint 路径

如果你不想把当前实验路径固化进仓库，这组可以不提交，或者单独提交。

### 5.5 可选的本地 debug 文件

例如：
- `src/openpi/models/tokenizer_print_test.py`

这类文件建议单独判断，通常不应默认混进主功能提交。

## 6. 总结

本次新增、并且需要特别注意的内容有两块：

1. `034 / 035` 现在已经完成了：
- 单条 / 多条结构拆分；
- 参数化；
- 真机运行验证；
- 真实报错修复。

2. 当前工作区里还有一些“不是同一层级”的改动：
- 有的是主线功能；
- 有的是运行时 checkpoint 路径切换；
- 有的是本地调试文件。

因此后续如果要 `git add` / `commit` / `push`，建议按上面的分组拆开，不要一次性全推。
