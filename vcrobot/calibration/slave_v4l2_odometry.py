#!/usr/bin/env python3
"""Robot-side L515 IR+depth odometry server using V4L2 only.

This service deliberately does not import ROS or pyrealsense2.  It reads the
registered L515 GREY and Z16 nodes through two persistent ``v4l2-ctl`` streams
and exposes the same TCP/50240 protocol as ``slave_rgbd_odometry.py``.

V4L2 does not expose the RealSense factory intrinsics, so fx/fy/ppx/ppy are
required command-line inputs.  Supplying guessed values makes the metric pose
an uncalibrated diagnostic, not a calibration reference.
"""

import argparse
from collections import deque
import json
import math
from pathlib import Path
import shutil
import socket
import socketserver
import subprocess
import sys
import threading
import time

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from rgbd_visual_odometry import RGBDVisualOdometry


PROTOCOL_VERSION = 1


def parse_distortion(value):
    if value is None or not str(value).strip():
        return np.zeros(5, dtype=float)
    result = np.asarray([float(item) for item in str(value).split(",")],
                        dtype=float)
    if len(result) not in (4, 5, 8, 12, 14):
        raise ValueError("distortion must contain 4, 5, 8, 12 or 14 values")
    if not np.all(np.isfinite(result)):
        raise ValueError("distortion values must be finite")
    return result


def decode_frame(data, width, height, pixel_format):
    """Decode one raw V4L2 frame without copying when possible."""
    if pixel_format == "GREY":
        dtype = np.dtype("u1")
    elif pixel_format == "Z16":
        dtype = np.dtype("<u2")
    else:
        raise ValueError(f"unsupported raw pixel format: {pixel_format}")
    expected = int(width) * int(height) * dtype.itemsize
    if len(data) != expected:
        raise ValueError(f"expected {expected} bytes, received {len(data)}")
    return np.frombuffer(data, dtype=dtype).reshape(int(height), int(width))


class RawV4L2Stream:
    """Continuously drain one raw V4L2 node into a small timestamped ring."""

    def __init__(self, device, width, height, fps, pixel_format,
                 ring_frames=12, executable=None):
        self.device = str(device)
        self.width = int(width)
        self.height = int(height)
        self.fps = float(fps)
        self.pixel_format = str(pixel_format).strip().upper()
        if self.pixel_format not in ("GREY", "Z16"):
            raise ValueError("pixel_format must be GREY or Z16")
        if self.width < 1 or self.height < 1 or self.fps <= 0:
            raise ValueError("width, height and fps must be positive")
        if not Path(self.device).exists():
            raise RuntimeError(f"V4L2 device not found: {self.device}")
        self.executable = executable or shutil.which("v4l2-ctl")
        if not self.executable:
            raise RuntimeError("v4l2-ctl is required (install v4l-utils)")
        self.frame_bytes = (self.width * self.height *
                            (2 if self.pixel_format == "Z16" else 1))
        self.cv = threading.Condition()
        self.frames = deque(maxlen=max(2, int(ring_frames)))
        self.error = None
        self.stopping = threading.Event()
        self.stderr_tail = deque(maxlen=30)
        self.process = subprocess.Popen(
            self.command(), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            stdin=subprocess.DEVNULL, bufsize=0)
        self.reader = threading.Thread(target=self._reader_loop, daemon=True)
        self.stderr_reader = threading.Thread(
            target=self._stderr_loop, daemon=True)
        self.reader.start()
        self.stderr_reader.start()

    def command(self):
        fps_text = (str(int(self.fps)) if self.fps.is_integer()
                    else str(self.fps))
        return [
            self.executable,
            "--device", self.device,
            "--set-fmt-video",
            (f"width={self.width},height={self.height},"
             f"pixelformat={self.pixel_format}"),
            "--set-parm", fps_text,
            "--stream-mmap=4",
            "--stream-to=-",
            "--silent",
        ]

    def _read_exact(self):
        if self.process.stdout is None:
            return None
        chunks = []
        remaining = self.frame_bytes
        while remaining and not self.stopping.is_set():
            chunk = self.process.stdout.read(remaining)
            if not chunk:
                return None
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks) if remaining == 0 else None

    def _reader_loop(self):
        try:
            while not self.stopping.is_set():
                raw = self._read_exact()
                if raw is None:
                    break
                timestamp_ns = time.monotonic_ns()
                image = decode_frame(raw, self.width, self.height,
                                     self.pixel_format).copy()
                with self.cv:
                    self.frames.append((timestamp_ns, image))
                    self.cv.notify_all()
        except Exception as exc:
            with self.cv:
                self.error = str(exc)
                self.cv.notify_all()
        finally:
            with self.cv:
                if self.error is None and not self.stopping.is_set():
                    detail = "; ".join(self.stderr_tail)
                    self.error = "V4L2 stream ended" + (
                        f": {detail}" if detail else "")
                self.cv.notify_all()

    def _stderr_loop(self):
        if self.process.stderr is None:
            return
        for raw in iter(self.process.stderr.readline, b""):
            text = raw.decode("utf-8", errors="replace").strip()
            if text:
                with self.cv:
                    self.stderr_tail.append(text)

    def next_frame(self, after_ns, deadline):
        with self.cv:
            while True:
                for timestamp_ns, image in self.frames:
                    if timestamp_ns > after_ns:
                        return timestamp_ns, image.copy()
                if self.error:
                    raise RuntimeError(
                        f"{self.device} {self.pixel_format}: {self.error}")
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(
                        f"timed out waiting for {self.device} "
                        f"{self.pixel_format} frame")
                self.cv.wait(remaining)

    def wait_frames(self, count, timeout_s=10.0):
        timestamp_ns = -1
        deadline = time.monotonic() + float(timeout_s)
        for _ in range(max(1, int(count))):
            timestamp_ns, _ = self.next_frame(timestamp_ns, deadline)

    def close(self):
        self.stopping.set()
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=2.0)
        for thread in (self.reader, self.stderr_reader):
            thread.join(timeout=1.0)
        for pipe in (self.process.stdout, self.process.stderr):
            if pipe is not None:
                pipe.close()


class L515V4L2Source:
    """Pair registered L515 IR and depth frames captured while stationary."""

    def __init__(self, ir_device, depth_device, width, height, fps,
                 depth_scale_m, fx, fy, ppx, ppy, distortion,
                 warmup_frames=30, max_pair_skew_ms=60.0):
        values = (depth_scale_m, fx, fy, ppx, ppy, max_pair_skew_ms)
        if any(not math.isfinite(float(value)) for value in values):
            raise ValueError("camera parameters must be finite")
        if depth_scale_m <= 0 or fx <= 0 or fy <= 0:
            raise ValueError("depth scale, fx and fy must be positive")
        self.lock = threading.Lock()
        self.width = int(width)
        self.height = int(height)
        self.depth_scale_m = float(depth_scale_m)
        self.max_pair_skew_ns = int(float(max_pair_skew_ms) * 1e6)
        self.camera_matrix = np.array([
            [float(fx), 0.0, float(ppx)],
            [0.0, float(fy), float(ppy)],
            [0.0, 0.0, 1.0],
        ])
        self.distortion = np.asarray(distortion, dtype=float)
        self.intrinsics = {
            "width": self.width, "height": self.height,
            "fx": float(fx), "fy": float(fy),
            "ppx": float(ppx), "ppy": float(ppy),
            "model": "opencv_manual_v4l2",
            "coeffs": self.distortion.tolist(),
            "status": "MANUAL_REQUIRES_VALIDATION",
            "depth_scale_m": self.depth_scale_m,
            "ir_device": str(ir_device), "depth_device": str(depth_device),
        }
        self.ir = None
        self.depth = None
        try:
            self.ir = RawV4L2Stream(
                ir_device, width, height, fps, "GREY")
            self.depth = RawV4L2Stream(
                depth_device, width, height, fps, "Z16")
            self.ir.wait_frames(warmup_frames)
            self.depth.wait_frames(warmup_frames)
        except Exception:
            self.close()
            raise

    def capture(self, timeout_ms=5000):
        with self.lock:
            deadline = time.monotonic() + float(timeout_ms) / 1000.0
            marker = time.monotonic_ns()
            ir_ts, ir = self.ir.next_frame(marker, deadline)
            depth_ts, depth_raw = self.depth.next_frame(marker, deadline)
            # Pipe-read timestamps are only a pairing heuristic. Advancing the
            # older stream is sufficient for stopped-between-samples trials.
            while abs(ir_ts - depth_ts) > self.max_pair_skew_ns:
                if ir_ts < depth_ts:
                    ir_ts, ir = self.ir.next_frame(ir_ts, deadline)
                else:
                    depth_ts, depth_raw = self.depth.next_frame(
                        depth_ts, deadline)
            depth_m = depth_raw.astype(np.float32) * self.depth_scale_m
            return ir, depth_m, max(ir_ts, depth_ts)

    def close(self):
        for stream in (self.depth, self.ir):
            if stream is not None:
                stream.close()


class OdometryService:
    def __init__(self, source, min_inliers=20, max_rmse_px=3.0):
        self.source = source
        self.lock = threading.Lock()
        self.estimator = RGBDVisualOdometry(
            source.camera_matrix, source.distortion,
            min_inliers=min_inliers, max_rmse_px=max_rmse_px)
        self.reference_timestamp_ns = None

    def request(self, message):
        kind = message.get("type")
        if kind == "hello":
            return {"type": "hello", "version": PROTOCOL_VERSION,
                    "robot": socket.gethostname(),
                    "sensor": "L515 V4L2 IR+Z16 visual odometry",
                    "intrinsics": self.source.intrinsics}
        if kind == "status":
            return {"type": "status", "ready": True,
                    "has_reference": self.reference_timestamp_ns is not None,
                    "reference_timestamp_ns": self.reference_timestamp_ns,
                    "intrinsics": self.source.intrinsics}
        if kind == "reset_reference":
            with self.lock:
                ir, depth, timestamp_ns = self.source.capture(
                    message.get("timeout_ms", 5000))
                features = self.estimator.set_reference(ir, depth)
                self.reference_timestamp_ns = timestamp_ns
            return {"type": "reference", "timestamp_ns": timestamp_ns,
                    "features": features}
        if kind == "estimate":
            with self.lock:
                if self.reference_timestamp_ns is None:
                    raise RuntimeError(
                        "reset_reference is required before estimate")
                ir, _, timestamp_ns = self.source.capture(
                    message.get("timeout_ms", 5000))
                result = self.estimator.estimate(ir).to_dict()
            return {"type": "estimate",
                    "reference_timestamp_ns": self.reference_timestamp_ns,
                    "timestamp_ns": timestamp_ns, **result}
        raise ValueError(f"unknown message type: {kind}")


class Handler(socketserver.StreamRequestHandler):
    def handle(self):
        for raw in self.rfile:
            try:
                message = json.loads(raw)
                reply = self.server.service.request(message)
            except Exception as exc:
                reply = {"type": "error", "error": str(exc)}
            self.wfile.write((json.dumps(reply, separators=(",", ":")) +
                              "\n").encode())
            self.wfile.flush()


class Server(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


def positive_finite(value):
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise argparse.ArgumentTypeError("value must be finite and positive")
    return result


def finite(value):
    result = float(value)
    if not math.isfinite(result):
        raise argparse.ArgumentTypeError("value must be finite")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--listen", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=50240)
    parser.add_argument("--ir-device", default="/dev/video0")
    parser.add_argument("--depth-device", default="/dev/video2")
    # These portrait dimensions are what this specific L515 advertises via
    # V4L2. Do not silently swap them to the librealsense 640x480 convention.
    parser.add_argument("--width", type=int, default=480)
    parser.add_argument("--height", type=int, default=640)
    parser.add_argument("--fps", type=positive_finite, default=30.0)
    parser.add_argument("--depth-scale-m", type=positive_finite, required=True,
                        help="metres represented by one Z16 increment")
    parser.add_argument("--fx", type=positive_finite, required=True)
    parser.add_argument("--fy", type=positive_finite, required=True)
    parser.add_argument("--ppx", type=finite, required=True)
    parser.add_argument("--ppy", type=finite, required=True)
    parser.add_argument("--distortion", default="0,0,0,0,0",
                        help="comma-separated OpenCV distortion coefficients")
    parser.add_argument("--warmup-frames", type=int, default=30)
    parser.add_argument("--max-pair-skew-ms", type=positive_finite,
                        default=60.0)
    parser.add_argument("--min-inliers", type=int, default=20)
    parser.add_argument("--max-rmse-px", type=positive_finite, default=3.0)
    args = parser.parse_args(argv)
    if args.width < 1 or args.height < 1:
        parser.error("width and height must be positive")
    try:
        distortion = parse_distortion(args.distortion)
    except ValueError as exc:
        parser.error(str(exc))
    source = L515V4L2Source(
        args.ir_device, args.depth_device, args.width, args.height, args.fps,
        args.depth_scale_m, args.fx, args.fy, args.ppx, args.ppy, distortion,
        args.warmup_frames, args.max_pair_skew_ms)
    server = Server((args.listen, args.port), Handler)
    server.service = OdometryService(
        source, args.min_inliers, args.max_rmse_px)
    print("L515 V4L2 IR+depth odometry listening on "
          f"{args.listen}:{args.port}", flush=True)
    print("Camera parameters are MANUAL_REQUIRES_VALIDATION", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
        server.server_close()
        source.close()


if __name__ == "__main__":
    main()
