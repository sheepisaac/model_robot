#!/usr/bin/env python3
"""Robot-side L515 RGB-D visual-odometry server (no motor access).

Run this instead of any other process that owns the L515.  It exposes only
read-only camera operations on TCP/50240: status, reset_reference and estimate.
"""

import argparse
import json
from pathlib import Path
import socket
import socketserver
import sys
import threading
import time

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from rgbd_visual_odometry import RGBDVisualOdometry


PROTOCOL_VERSION = 1


class L515Source:
    def __init__(self, width=640, height=480, fps=30, warmup_frames=30):
        try:
            import pyrealsense2 as rs
        except ImportError as exc:
            raise RuntimeError("install pyrealsense2 on the robot") from exc
        self.rs = rs
        self.lock = threading.Lock()
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        profile = self.pipeline.start(config)
        self.align = rs.align(rs.stream.color)
        self.depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
        for _ in range(int(warmup_frames)):
            self.pipeline.wait_for_frames(5000)
        color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
        intrinsics = color_profile.get_intrinsics()
        self.intrinsics = {
            "width": intrinsics.width, "height": intrinsics.height,
            "fx": intrinsics.fx, "fy": intrinsics.fy,
            "ppx": intrinsics.ppx, "ppy": intrinsics.ppy,
            "model": str(intrinsics.model),
            "coeffs": [float(item) for item in intrinsics.coeffs],
        }
        self.camera_matrix = np.array([
            [intrinsics.fx, 0.0, intrinsics.ppx],
            [0.0, intrinsics.fy, intrinsics.ppy],
            [0.0, 0.0, 1.0],
        ])
        self.distortion = np.asarray(intrinsics.coeffs, dtype=float)

    def capture(self, timeout_ms=5000):
        with self.lock:
            frames = self.align.process(
                self.pipeline.wait_for_frames(int(timeout_ms)))
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            if not depth_frame or not color_frame:
                raise RuntimeError("aligned L515 depth/color frame is incomplete")
            return (np.asanyarray(color_frame.get_data()).copy(),
                    np.asanyarray(depth_frame.get_data()).astype(np.float32) *
                    self.depth_scale,
                    time.monotonic_ns())

    def close(self):
        self.pipeline.stop()


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
                    "sensor": "L515 RGB-D visual odometry",
                    "intrinsics": self.source.intrinsics}
        if kind == "status":
            return {"type": "status", "ready": True,
                    "has_reference": self.reference_timestamp_ns is not None,
                    "reference_timestamp_ns": self.reference_timestamp_ns,
                    "intrinsics": self.source.intrinsics}
        if kind == "reset_reference":
            with self.lock:
                color, depth, timestamp_ns = self.source.capture(
                    message.get("timeout_ms", 5000))
                features = self.estimator.set_reference(color, depth)
                self.reference_timestamp_ns = timestamp_ns
            return {"type": "reference", "timestamp_ns": timestamp_ns,
                    "features": features}
        if kind == "estimate":
            with self.lock:
                if self.reference_timestamp_ns is None:
                    raise RuntimeError("reset_reference is required before estimate")
                color, _, timestamp_ns = self.source.capture(
                    message.get("timeout_ms", 5000))
                result = self.estimator.estimate(color).to_dict()
            return {"type": "estimate", "reference_timestamp_ns":
                    self.reference_timestamp_ns,
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


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--listen", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=50240)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--warmup-frames", type=int, default=30)
    parser.add_argument("--min-inliers", type=int, default=20)
    parser.add_argument("--max-rmse-px", type=float, default=3.0)
    args = parser.parse_args(argv)
    source = L515Source(args.width, args.height, args.fps, args.warmup_frames)
    server = Server((args.listen, args.port), Handler)
    server.service = OdometryService(
        source, args.min_inliers, args.max_rmse_px)
    print(f"L515 RGB-D odometry listening on {args.listen}:{args.port}", flush=True)
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
