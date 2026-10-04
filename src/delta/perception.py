# -*- coding: utf-8 -*-
"""视觉侧桥接：在另一个 Python 环境里跑分割，把作物掩膜读回本环境。

本仓库不 import ultralytics：改进版权重要靠 sys.path 前置的 fork 包才能反序列化，
那套代码是 AGPL 且拖着 torch。跨环境只走一条文件契约——worker 写出 .npz，这里读回数组。
"""
from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np

WORKER = Path(__file__).resolve().parents[2] / "scripts" / "perception_worker.py"


def _resolve(name, explicit, default=""):
    return explicit or os.environ.get(name) or default


def _tmp_name(suffix):
    """只借临时目录生成一个不占位的文件名：mkstemp 会创建空文件，
    而 np.savez 遇到已存在的同名文件会改名写入。"""
    return Path(tempfile.gettempdir()) / f"delta-{os.getpid()}-{os.urandom(4).hex()}{suffix}"


def crop_mask(image, *, python_exe=None, weights=None, ultralytics_root=None,
              conf=0.25, imgsz=512, device=None, out=None):
    """田间图 -> 作物二值掩膜。

    image  : BGR 数组或图片路径（worker 只吃路径，数组会先落成临时 PNG）
    out    : 给定则把掩膜存到该 .npz 并保留，省掉重复推理；否则用临时文件
    返回    {"crop_mask": (H, W) uint8 0/255, "meta": worker 输出的统计}
    """
    py = _resolve("DELTA_PERCEPTION_PYTHON", python_exe)
    w = _resolve("DELTA_HLB_WEIGHTS", weights)
    root = _resolve("DELTA_ULTRALYTICS_ROOT", ultralytics_root)
    if not py:
        raise ValueError("需要感知侧 Python 解释器：--python 或环境变量 DELTA_PERCEPTION_PYTHON")
    if not WORKER.exists():
        raise FileNotFoundError(f"找不到 worker 脚本: {WORKER}")

    keep = out is not None
    if keep:
        npz = Path(out)
        npz.parent.mkdir(parents=True, exist_ok=True)
    else:
        npz = Path(_tmp_name(".npz"))

    src_is_temp = isinstance(image, np.ndarray)
    src = Path(_tmp_name(".png")) if src_is_temp else Path(image)
    if src_is_temp:
        cv2.imwrite(str(src), image)  # PNG 无损，再编码不会改变像素

    cmd = [py, str(WORKER), "--image", str(src), "--out", str(npz),
           "--conf", str(conf), "--imgsz", str(imgsz)]
    if w:
        cmd += ["--weights", w]
    if root:
        cmd += ["--ultralytics-root", root]
    if device is not None:
        cmd += ["--device", str(device)]

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              encoding="utf-8", errors="replace")
        if proc.returncode != 0:
            raise RuntimeError(
                f"感知侧推理失败 (exit {proc.returncode}):\n{proc.stderr.strip()[-2000:]}")
        meta = json.loads(proc.stdout.strip().splitlines()[-1])
        mask = np.load(npz)["crop_mask"]
    finally:
        if not keep:
            npz.unlink(missing_ok=True)
        if src_is_temp:
            src.unlink(missing_ok=True)

    return {"crop_mask": mask, "meta": meta}
