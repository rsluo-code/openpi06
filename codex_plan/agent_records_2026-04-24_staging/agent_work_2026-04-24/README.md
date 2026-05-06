# Agent Work 2026-04-24

本记录用于给下一个 agent 接手当前 `openpi06` 上的 ValueNet 训练、validation、server 联调，以及 `calAt/calVal` 计算链路。

当前最简洁的接手状态见同目录：

```text
CURRENT_STATUS.md
```

本轮工作的重点不在新训练任务启动，而在把 ValueNet 从三路图像扩展到四路图像 `episode_first_head_img` 后，把训练、server 推理、validation 可视化、和离线 `calAt/calVal` 这几条链路都补齐并验证到能继续推进。
