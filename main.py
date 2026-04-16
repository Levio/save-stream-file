from __future__ import annotations

import argparse
import math
import signal
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

try:
    import cv2
except ModuleNotFoundError as exc:
    print("[ERROR] 未安装 opencv-python，请先执行: pip install -e .", file=sys.stderr)
    raise exc


@dataclass
class Config:
    output_dir: Path
    segment_seconds: int
    target_width: int
    target_height: int
    target_fps: float
    camera_index: int
    scan_max_index: int
    reconnect_interval: float
    max_read_failures: int
    codec_candidates: tuple[str, ...]
    capture_fourcc_candidates: tuple[str, ...]
    probe_resolutions: tuple[tuple[int, int], ...]
    probe_fps_candidates: tuple[float, ...]
    probe_read_attempts: int
    probe_timeout_seconds: float
    max_probe_profiles: int
    reconnect_warmup_seconds: float


@dataclass
class StreamProfile:
    request_fourcc: str
    request_width: int
    request_height: int
    request_fps: float
    actual_fourcc: str
    actual_width: int
    actual_height: int
    actual_fps: float


class Recorder:
    def __init__(self, config: Config) -> None:
        self.cfg = config
        self.running = True
        self.cap: cv2.VideoCapture | None = None
        self.cap_index: int | None = None
        self.writer: cv2.VideoWriter | None = None
        self.writer_codec: str | None = None
        self.segment_start_monotonic = 0.0
        self.segment_start_wall = 0.0
        self.read_failures = 0
        self.frame_count = 0
        self.segment_serial = 0
        self.current_frame_size: tuple[int, int] | None = None
        self.current_fps = self.cfg.target_fps
        self.last_good_index: int | None = None
        self.last_good_profile: StreamProfile | None = None
        self.scan_clamp_warned = False

    def normalize_fps(self, fps: float | int | None, fallback: float) -> float:
        value = float(fps) if fps is not None else float(fallback)
        if not math.isfinite(value) or value < 1.0:
            value = float(fallback)
        # Keep file playback speed stable: never exceed configured target fps.
        value = min(value, self.cfg.target_fps)
        return max(1.0, value)

    @staticmethod
    def fourcc_to_str(fourcc_value: float | int) -> str:
        try:
            code = int(fourcc_value)
        except (ValueError, TypeError):
            return "----"
        if code <= 0:
            return "----"
        chars = []
        for shift in (0, 8, 16, 24):
            ch = (code >> shift) & 0xFF
            chars.append(chr(ch) if 32 <= ch <= 126 else "-")
        return "".join(chars)

    def stream_score(self, profile: StreamProfile) -> tuple[int, float, int]:
        area = profile.actual_width * profile.actual_height
        fps_score = min(profile.actual_fps, self.cfg.target_fps)
        mjpg_bonus = 1 if profile.actual_fourcc == "MJPG" else 0
        return (area, fps_score, mjpg_bonus)

    def set_capture_profile(
        self,
        cap: cv2.VideoCapture,
        fourcc: str,
        width: int,
        height: int,
        fps: float,
    ) -> None:
        if fourcc:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(width))
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(height))
        cap.set(cv2.CAP_PROP_FPS, float(fps))
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass

    def read_frame_with_retry(
        self, cap: cv2.VideoCapture, attempts: int, delay_seconds: float = 0.03
    ) -> tuple[bool, object | None]:
        for _ in range(attempts):
            ok, frame = cap.read()
            if ok and frame is not None:
                return True, frame
            time.sleep(delay_seconds)
        return False, None

    def open_camera(self, index: int) -> cv2.VideoCapture | None:
        backends: list[int | None] = []
        if sys.platform == "darwin" and hasattr(cv2, "CAP_AVFOUNDATION"):
            backends.append(cv2.CAP_AVFOUNDATION)
        backends.append(None)

        for backend in backends:
            cap = cv2.VideoCapture(index, backend) if backend is not None else cv2.VideoCapture(index)
            try:
                cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 800)
                cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 800)
            except Exception:
                pass
            if cap.isOpened():
                return cap
            cap.release()
        return None

    def profile_from_frame(self, frame: object, cap: cv2.VideoCapture) -> StreamProfile:
        actual_h, actual_w = frame.shape[:2]
        actual_fps = self.normalize_fps(cap.get(cv2.CAP_PROP_FPS), self.cfg.target_fps)
        actual_fourcc = self.fourcc_to_str(cap.get(cv2.CAP_PROP_FOURCC))
        return StreamProfile(
            request_fourcc="AUTO",
            request_width=actual_w,
            request_height=actual_h,
            request_fps=actual_fps,
            actual_fourcc=actual_fourcc,
            actual_width=actual_w,
            actual_height=actual_h,
            actual_fps=actual_fps,
        )

    def probe_current_stream(
        self, cap: cv2.VideoCapture
    ) -> tuple[StreamProfile | None, object | None]:
        ok, frame = self.read_frame_with_retry(
            cap, attempts=max(8, self.cfg.probe_read_attempts), delay_seconds=0.08
        )
        if not ok or frame is None:
            return None, None

        return self.profile_from_frame(frame, cap), frame

    def try_reconnect_with_last_profile(
        self, cap: cv2.VideoCapture, camera_index: int
    ) -> tuple[StreamProfile | None, object | None]:
        if self.last_good_profile is None:
            return None, None
        if self.last_good_index is not None and camera_index != self.last_good_index:
            return None, None

        profile = self.last_good_profile
        req_fourcc = "" if profile.request_fourcc == "AUTO" else profile.request_fourcc
        self.set_capture_profile(
            cap,
            req_fourcc,
            profile.actual_width,
            profile.actual_height,
            profile.actual_fps,
        )
        attempts = max(8, int(self.cfg.reconnect_warmup_seconds / 0.08))
        ok, frame = self.read_frame_with_retry(cap, attempts=attempts, delay_seconds=0.08)
        if not ok or frame is None:
            return None, None
        live_profile = self.profile_from_frame(frame, cap)
        live_profile.request_fourcc = profile.request_fourcc
        live_profile.request_width = profile.request_width
        live_profile.request_height = profile.request_height
        live_profile.request_fps = profile.request_fps
        print(
            f"[INFO] 快速重连成功: index={camera_index}, 实际={live_profile.actual_width}x{live_profile.actual_height}@{live_profile.actual_fps:.2f} {live_profile.actual_fourcc}"
        )
        return live_profile, frame

    def try_simple_target_profile(
        self, cap: cv2.VideoCapture, camera_index: int
    ) -> tuple[StreamProfile | None, object | None]:
        # Reconnect phase: keep this lightweight to avoid long blocking on macOS.
        target_candidates = (
            (self.cfg.target_width, self.cfg.target_height, self.cfg.target_fps),
            (1920, 1080, min(self.cfg.target_fps, 30.0)),
            (1280, 720, min(self.cfg.target_fps, 30.0)),
            (640, 480, min(self.cfg.target_fps, 30.0)),
        )
        for fourcc in self.cfg.capture_fourcc_candidates:
            for width, height, fps in target_candidates:
                self.set_capture_profile(cap, fourcc, width, height, fps)
                ok, frame = self.read_frame_with_retry(cap, attempts=4, delay_seconds=0.06)
                if not ok or frame is None:
                    continue
                actual_profile = self.profile_from_frame(frame, cap)
                actual_profile.request_fourcc = fourcc if fourcc else "AUTO"
                actual_profile.request_width = width
                actual_profile.request_height = height
                actual_profile.request_fps = fps
                print(
                    f"[INFO] 轻量重连成功: index={camera_index}, 请求={width}x{height}@{fps:.0f} {actual_profile.request_fourcc}, "
                    f"实际={actual_profile.actual_width}x{actual_profile.actual_height}@{actual_profile.actual_fps:.2f} {actual_profile.actual_fourcc}"
                )
                return actual_profile, frame
        return None, None

    def probe_profile(
        self,
        cap: cv2.VideoCapture,
        fourcc: str,
        width: int,
        height: int,
        fps: float,
    ) -> StreamProfile | None:
        self.set_capture_profile(cap, fourcc, width, height, fps)
        ok, frame = self.read_frame_with_retry(cap, self.cfg.probe_read_attempts)
        if not ok or frame is None:
            return None

        actual_h, actual_w = frame.shape[:2]
        actual_fps = self.normalize_fps(cap.get(cv2.CAP_PROP_FPS), fps)
        actual_fourcc = self.fourcc_to_str(cap.get(cv2.CAP_PROP_FOURCC))

        if actual_w < 320 or actual_h < 240:
            return None

        return StreamProfile(
            request_fourcc=fourcc,
            request_width=width,
            request_height=height,
            request_fps=fps,
            actual_fourcc=actual_fourcc,
            actual_width=actual_w,
            actual_height=actual_h,
            actual_fps=actual_fps,
        )

    def select_best_stream_profile(
        self, cap: cv2.VideoCapture, camera_index: int
    ) -> tuple[StreamProfile | None, object | None]:
        fallback_profile, fallback_frame = self.probe_current_stream(cap)
        best: StreamProfile | None = fallback_profile
        started = time.monotonic()
        deadline = started + self.cfg.probe_timeout_seconds
        attempts = 0
        timed_out = False

        for fourcc in self.cfg.capture_fourcc_candidates:
            for width, height in self.cfg.probe_resolutions:
                for fps in self.cfg.probe_fps_candidates:
                    if attempts >= self.cfg.max_probe_profiles:
                        timed_out = True
                        break
                    if time.monotonic() >= deadline:
                        timed_out = True
                        break
                    attempts += 1
                    profile = self.probe_profile(cap, fourcc, width, height, fps)
                    if profile is None:
                        continue
                    if best is None or self.stream_score(profile) > self.stream_score(
                        best
                    ):
                        best = profile
                if timed_out:
                    break
            if timed_out:
                break

        if best is None:
            cost = time.monotonic() - started
            print(
                f"[WARN] 相机探测失败: index={camera_index}, attempts={attempts}, elapsed={cost:.2f}s. "
                "请检查相机权限、是否被其他软件占用，或手动指定 --camera-index。"
            )
            return None, None

        if best.request_fourcc == "AUTO" and fallback_frame is not None:
            cost = time.monotonic() - started
            print(
                f"[INFO] 相机探测完成: index={camera_index}, attempts={attempts}, "
                f"elapsed={cost:.2f}s, timeout={'yes' if timed_out else 'no'} (fallback=auto)"
            )
            return best, fallback_frame

        self.set_capture_profile(
            cap,
            best.request_fourcc,
            best.actual_width,
            best.actual_height,
            best.actual_fps,
        )
        ok, frame = self.read_frame_with_retry(
            cap, max(self.cfg.probe_read_attempts, 4), delay_seconds=0.05
        )
        if not ok:
            if fallback_profile is not None and fallback_frame is not None:
                cost = time.monotonic() - started
                print(
                    f"[WARN] 目标流复位失败，回退到默认流: index={camera_index}, elapsed={cost:.2f}s"
                )
                return fallback_profile, fallback_frame
            return None, None

        cost = time.monotonic() - started
        print(
            f"[INFO] 相机探测完成: index={camera_index}, attempts={attempts}, "
            f"elapsed={cost:.2f}s, timeout={'yes' if timed_out else 'no'}"
        )
        return best, frame

    def estimate_effective_fps(
        self, cap: cv2.VideoCapture, max_frames: int = 45, max_seconds: float = 1.6
    ) -> float | None:
        begin = time.monotonic()
        stamps: list[float] = []
        while len(stamps) < max_frames and (time.monotonic() - begin) < max_seconds:
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            stamps.append(time.monotonic())
        if len(stamps) < 6:
            return None
        elapsed = stamps[-1] - stamps[0]
        if elapsed <= 0:
            return None
        measured = (len(stamps) - 1) / elapsed
        return self.normalize_fps(measured, self.cfg.target_fps)

    def setup_signal_handlers(self) -> None:
        def _handle_stop(_signum: int, _frame: object) -> None:
            self.running = False

        signal.signal(signal.SIGINT, _handle_stop)
        signal.signal(signal.SIGTERM, _handle_stop)

    def run(self) -> None:
        self.cfg.output_dir.mkdir(parents=True, exist_ok=True)
        self.setup_signal_handlers()

        print(f"[INFO] 输出目录: {self.cfg.output_dir.resolve()}")
        print("[INFO] 程序启动，等待并连接 USB 相机...")

        try:
            while self.running:
                if self.cap is None:
                    self.try_connect_camera()
                    if self.cap is None:
                        time.sleep(self.cfg.reconnect_interval)
                        continue

                if not self.process_one_frame():
                    self.handle_camera_lost()
        finally:
            self.cleanup()

    def try_connect_camera(self) -> None:
        if self.cfg.camera_index >= 0:
            indices = [self.cfg.camera_index]
        else:
            max_index = self.cfg.scan_max_index
            if sys.platform == "darwin" and self.last_good_index in (None, 0, 1):
                if max_index > 1 and not self.scan_clamp_warned:
                    print("[INFO] macOS 自动扫描索引限制为 0..1，避免无效索引导致重连噪音。")
                    self.scan_clamp_warned = True
                max_index = min(max_index, 1)
            indices = list(range(max_index + 1))
            if self.last_good_index is not None and self.last_good_index in indices:
                indices.remove(self.last_good_index)
                indices.insert(0, self.last_good_index)

        reconnect_mode = self.last_good_profile is not None

        for idx in indices:
            cap = self.open_camera(idx)
            if cap is None:
                continue

            print(f"[INFO] 正在探测相机: index={idx}")

            quick_profile, quick_frame = self.try_reconnect_with_last_profile(cap, idx)
            if quick_profile is not None and quick_frame is not None:
                best, frame = quick_profile, quick_frame
            else:
                if reconnect_mode:
                    best, frame = self.try_simple_target_profile(cap, idx)
                    if best is None or frame is None:
                        best, frame = self.probe_current_stream(cap)
                        if best is not None and frame is not None:
                            print(
                                f"[INFO] 回退默认流重连: index={idx}, 实际={best.actual_width}x{best.actual_height}@{best.actual_fps:.2f} {best.actual_fourcc}"
                            )
                else:
                    best, frame = self.select_best_stream_profile(cap, idx)

            if best is None or frame is None:
                cap.release()
                continue

            height, width = frame.shape[:2]
            estimated_fps = self.estimate_effective_fps(cap)
            fps = self.normalize_fps(
                estimated_fps if estimated_fps is not None else best.actual_fps,
                self.cfg.target_fps,
            )
            best.actual_fps = fps

            self.cap = cap
            self.cap_index = idx
            self.current_frame_size = (width, height)
            self.current_fps = fps
            self.read_failures = 0
            self.last_good_index = idx
            self.last_good_profile = best
            print(
                f"[INFO] 相机已连接: index={idx}, 请求={best.request_width}x{best.request_height}@{best.request_fps:.0f} {best.request_fourcc}, "
                f"实际={width}x{height}@{fps:.2f} {best.actual_fourcc}"
            )
            return

    def process_one_frame(self) -> bool:
        assert self.cap is not None
        ok, frame = self.cap.read()
        if not ok or frame is None:
            self.read_failures += 1
            if self.read_failures >= self.cfg.max_read_failures:
                return False
            time.sleep(0.05)
            return True

        self.read_failures = 0
        now_mono = time.monotonic()
        now_wall = time.time()

        frame_h, frame_w = frame.shape[:2]
        frame_size = (frame_w, frame_h)

        if self.writer is None:
            if not self.open_new_segment(frame_size, now_mono, now_wall):
                time.sleep(0.2)
                return True
        elif now_mono - self.segment_start_monotonic >= self.cfg.segment_seconds:
            self.close_writer()
            if not self.open_new_segment(frame_size, now_mono, now_wall):
                time.sleep(0.2)
                return True
        elif self.current_frame_size != frame_size:
            self.close_writer()
            if not self.open_new_segment(frame_size, now_mono, now_wall):
                time.sleep(0.2)
                return True

        assert self.writer is not None
        self.writer.write(frame)
        self.frame_count += 1
        if self.frame_count % int(max(self.current_fps, 1) * 10) == 0:
            print(
                f"[INFO] 正在录制: cam={self.cap_index}, codec={self.writer_codec}, "
                f"segment_started={datetime.fromtimestamp(self.segment_start_wall).isoformat(timespec='seconds')}"
            )
        return True

    def build_segment_path(self, start_wall_ts: float, serial: int) -> Path:
        dt = datetime.fromtimestamp(start_wall_ts)
        stamp = dt.strftime("%Y%m%d_%H%M%S")
        idx = self.cap_index if self.cap_index is not None else -1
        name = f"{stamp}_cam{idx}_{serial:06d}.mp4"
        return self.cfg.output_dir / name

    def open_new_segment(
        self, frame_size: tuple[int, int], now_mono: float, now_wall: float
    ) -> bool:
        self.segment_serial += 1
        out_path = self.build_segment_path(now_wall, self.segment_serial)
        fps = self.normalize_fps(self.current_fps, self.cfg.target_fps)

        writer = None
        codec_used = None
        for codec in self.cfg.codec_candidates:
            fourcc = cv2.VideoWriter_fourcc(*codec)
            candidate = cv2.VideoWriter(str(out_path), fourcc, fps, frame_size)
            if candidate.isOpened():
                writer = candidate
                codec_used = codec
                break
            candidate.release()

        if writer is None:
            print(
                f"[WARN] 无法创建视频文件: {out_path.name}，可能是编码器不可用，稍后重试。"
            )
            return False

        self.writer = writer
        self.writer_codec = codec_used
        self.current_frame_size = frame_size
        self.segment_start_monotonic = now_mono
        self.segment_start_wall = now_wall
        print(
            f"[INFO] 开始新片段: {out_path.name}, codec={codec_used}, "
            f"size={frame_size[0]}x{frame_size[1]}, fps={fps:.2f}"
        )
        return True

    def handle_camera_lost(self) -> None:
        print("[WARN] 相机读取失败，判定为断开，等待重连...")
        self.close_writer()
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.cap_index = None
        self.read_failures = 0
        time.sleep(self.cfg.reconnect_interval)

    def close_writer(self) -> None:
        if self.writer is not None:
            self.writer.release()
            self.writer = None
            self.writer_codec = None

    def cleanup(self) -> None:
        self.close_writer()
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        print("[INFO] 已安全退出。")


def parse_args(argv: list[str]) -> Config:
    parser = argparse.ArgumentParser(
        description="按固定时长切片保存 USB 相机视频（支持拔插后自动继续）。"
    )
    default_scan_max_index = 1 if sys.platform == "darwin" else 3
    parser.add_argument("--output-dir", default="recordings", help="输出目录")
    parser.add_argument("--segment-seconds", type=int, default=15, help="每段时长(秒)")
    parser.add_argument("--width", type=int, default=1920, help="目标宽度")
    parser.add_argument("--height", type=int, default=1080, help="目标高度")
    parser.add_argument("--fps", type=float, default=30.0, help="目标帧率")
    parser.add_argument(
        "--camera-index",
        type=int,
        default=-1,
        help="-1 自动扫描，>=0 指定相机索引",
    )
    parser.add_argument(
        "--scan-max-index",
        type=int,
        default=default_scan_max_index,
        help="自动扫描时最大相机索引",
    )
    parser.add_argument(
        "--reconnect-interval",
        type=float,
        default=2.0,
        help="重连等待间隔(秒)",
    )
    parser.add_argument(
        "--max-read-failures",
        type=int,
        default=20,
        help="连续读帧失败次数阈值，超过后判定断开",
    )
    parser.add_argument(
        "--probe-read-attempts",
        type=int,
        default=2,
        help="探测每种流配置时的读帧重试次数",
    )
    parser.add_argument(
        "--probe-timeout-seconds",
        type=float,
        default=4.0,
        help="每个相机探测的最大耗时(秒)，避免启动卡住",
    )
    parser.add_argument(
        "--max-probe-profiles",
        type=int,
        default=30,
        help="每个相机最多探测的流配置数量",
    )
    parser.add_argument(
        "--reconnect-warmup-seconds",
        type=float,
        default=2.0,
        help="拔插后快速重连阶段等待相机出帧的时间(秒)",
    )
    args = parser.parse_args(argv)

    resolution_pool = {
        (3840, 2160),
        (2560, 1440),
        (1920, 1080),
        (1600, 1200),
        (1280, 720),
        (640, 480),
        (max(320, args.width), max(240, args.height)),
    }
    sorted_resolutions = tuple(
        sorted(resolution_pool, key=lambda size: (size[0] * size[1], size[0]), reverse=True)
    )

    target_fps = float(max(1.0, args.fps))
    ordered_fps: list[float] = []
    for fps in (target_fps, 30.0, 15.0):
        if fps not in ordered_fps:
            ordered_fps.append(fps)

    return Config(
        output_dir=Path(args.output_dir),
        segment_seconds=max(1, args.segment_seconds),
        target_width=max(320, args.width),
        target_height=max(240, args.height),
        target_fps=max(1.0, args.fps),
        camera_index=args.camera_index,
        scan_max_index=max(0, args.scan_max_index),
        reconnect_interval=max(0.2, args.reconnect_interval),
        max_read_failures=max(1, args.max_read_failures),
        codec_candidates=("avc1", "mp4v", "MJPG"),
        capture_fourcc_candidates=("MJPG", "YUYV", ""),
        probe_resolutions=sorted_resolutions,
        probe_fps_candidates=tuple(ordered_fps),
        probe_read_attempts=max(1, args.probe_read_attempts),
        probe_timeout_seconds=max(0.5, args.probe_timeout_seconds),
        max_probe_profiles=max(1, args.max_probe_profiles),
        reconnect_warmup_seconds=max(0.5, args.reconnect_warmup_seconds),
    )


def main(argv: list[str] | None = None) -> int:
    cfg = parse_args(argv if argv is not None else sys.argv[1:])
    recorder = Recorder(cfg)
    recorder.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
