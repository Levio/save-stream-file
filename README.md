# save-camera-mp4

使用 `opencv-python` 持续录制 USB 相机视频，并按每 15 秒切片保存为 MP4 文件。  
支持相机断开后自动重连，重连成功后继续生成新文件。

## 安装依赖

```bash
pip install -e .
```

## 运行

```bash
python3 main.py
```

默认行为：
- 输出目录：`./recordings`
- 每段时长：15 秒
- 自动探测并优先选择相机可用的最高分辨率流（优先 MJPG）
- 默认目标分辨率参数：1920x1080（也会参与最高分辨率探测）
- 目标帧率：30 FPS
- 相机索引：macOS 默认自动扫描 `0..1`（减少无效索引报错）

## 常用参数

```bash
python3 main.py \
  --output-dir ./videos \
  --segment-seconds 15 \
  --width 1920 \
  --height 1080 \
  --fps 30 \
  --camera-index -1 \
  --scan-max-index 1 \
  --probe-read-attempts 2 \
  --probe-timeout-seconds 4 \
  --max-probe-profiles 30 \
  --reconnect-warmup-seconds 2
```

说明：
- `--camera-index -1` 表示自动扫描 USB 相机；如果知道索引，可指定如 `--camera-index 0`。
- 启动连接时会探测多组分辨率/帧率/像素格式，自动选择实际分辨率最高的可用流再开始保存。
- 为避免启动卡住，探测阶段有总超时（`--probe-timeout-seconds`）和最大探测数（`--max-probe-profiles`）保护。
- 断开后会优先用“上次成功的流配置”快速重连；失败时走轻量重连（目标配置 + 当前流），避免热插拔后长时间卡在探测。
- 日志会打印“请求参数”和“实际参数”，例如：`请求=1920x1080@30 MJPG, 实际=3840x2160@15.00 MJPG`。
- 程序会在连接后估算真实采集 fps，并用于写文件，避免出现“15 秒切片但播放仅几秒”的时长偏差。
- 如果读取失败达到阈值，程序会判断相机断开并自动等待重连。
- 按 `Ctrl+C` 可安全退出，已写入的视频文件会正常关闭。

探测失败排查：
- 先关闭占用摄像头的软件（视频会议、直播、录屏等）。
- 检查相机权限：`系统设置 -> 隐私与安全性 -> 相机`，给运行终端授权。
- 指定索引重试，例如：`python3 main.py --camera-index 0`。
- 如仍失败，可先降低探测负载：`--probe-timeout-seconds 2 --max-probe-profiles 10`。

## 打包为 macOS 可执行文件

在项目目录执行：

```bash
uv pip install --python .venv/bin/python pyinstaller
PYINSTALLER_CONFIG_DIR=.pyinstaller .venv/bin/pyinstaller \
  --noconfirm --clean --onefile --name camera-recorder main.py
```

打包产物：

```bash
./dist/camera-recorder
```

查看帮助：

```bash
./dist/camera-recorder --help
```
