# 2026-04-24 025 视频布局和尺寸改造

本轮把 `025` 对应的 mp4 渲染从旧的 2x2 布局，改成了用户要求的 3 行 2 列布局，并把单个 panel 的宽高改成可以从 bash 脚本里配置。

## 修改点

- `validation_1_hand_valuenet.py`
  - 删除旧的 `render_quadrant_video(...)` 逻辑
  - 新增：
    - `_load_panel_image(...)`
    - `render_dashboard_video(...)`
  - 新视频布局为：

```text
第 1 行：head  | out_png
第 2 行：left  | out_png_At
第 3 行：right | out_png_It
```

  - 左列是逐帧视频，右列是三张静态 plot 图。
  - 所有格子都会 resize 到统一的 `(panel_height, panel_width)`。

- `validation_1_hand_valuenet.py -> Args`
  - 新增：
    - `video_panel_width`
    - `video_panel_height`

- `025validation_1_hand_valuenet_multi.sh`
  - 新增 bash 配置项：

```bash
VIDEO_PANEL_WIDTH="640"
VIDEO_PANEL_HEIGHT="360"
```

  - 并把它们传给 python：

```bash
--args.video-panel-width
--args.video-panel-height
```

## 当前效果

- 现在 mp4 不再沿用 head 原始尺寸。
- 你可以直接在 `025validation_1_hand_valuenet_multi.sh` 里改：
  - `VIDEO_PANEL_WIDTH`
  - `VIDEO_PANEL_HEIGHT`
- 当前默认先用：

```text
640 x 360
```

## 验证

- `python3 -m py_compile validation_1_hand_valuenet.py validation_1_hand_valuenet_multi.py`
- `bash -n 025validation_1_hand_valuenet_multi.sh`

都已通过。
