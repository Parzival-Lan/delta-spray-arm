# -*- coding: utf-8 -*-
"""感知侧 worker：在装有 torch 的环境里运行，图片 -> crop_mask.npz + stdout 一行 JSON。

不要在本仓库的 delta 环境里跑它。ultralytics 必须是 sys.path 前置的 fork(8.3.15)，
pip 版会覆盖 fork，导致 HLB 权重里的自定义模块无法反序列化。
"""
import argparse
import json
import os
import sys
import time

import cv2
import numpy as np


def main():
    ap = argparse.ArgumentParser(description="crop mask inference (run in the perception env)")
    ap.add_argument("--image", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--weights", default=os.environ.get("DELTA_HLB_WEIGHTS", ""))
    ap.add_argument("--ultralytics-root", default=os.environ.get("DELTA_ULTRALYTICS_ROOT", ""))
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=512)
    ap.add_argument("--device", default=None)
    a = ap.parse_args()

    if not a.weights:
        sys.exit("需要权重路径：--weights 或环境变量 DELTA_HLB_WEIGHTS")

    if a.ultralytics_root:
        sys.path.insert(0, a.ultralytics_root)
    import ultralytics
    if a.ultralytics_root:
        got = os.path.abspath(ultralytics.__file__)
        want = os.path.abspath(a.ultralytics_root)
        if not got.startswith(want):
            sys.exit(f"ultralytics 未走 fork: {got} (期望在 {want} 下)")

    from ultralytics import YOLO

    img = cv2.imread(a.image)
    if img is None:
        sys.exit(f"图片读取失败: {a.image}")
    h, w = img.shape[:2]

    device = a.device
    if device is None:
        import torch
        device = 0 if torch.cuda.is_available() else "cpu"

    t0 = time.time()
    model = YOLO(a.weights)
    r = model.predict(source=img, imgsz=a.imgsz, conf=a.conf,
                      device=device, retina_masks=True, verbose=False)[0]

    mask = np.zeros((h, w), np.uint8)
    class_count, confs = {}, []
    if r.masks is not None and len(r.masks) > 0:
        data = r.masks.data.cpu().numpy()
        if data.shape[1:] != (h, w):
            data = np.stack([cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
                             for m in data])
        mask = (data > 0.5).any(axis=0).astype(np.uint8) * 255
        names = getattr(model, "names", {})
        for c, cf in zip(r.boxes.cls.cpu().numpy().astype(int),
                         r.boxes.conf.cpu().numpy().astype(float)):
            key = names.get(int(c), str(int(c))) if isinstance(names, dict) else str(int(c))
            class_count[key] = class_count.get(key, 0) + 1
            confs.append(round(float(cf), 4))
    np.savez_compressed(a.out, crop_mask=mask)

    print(json.dumps({
        "ultralytics": {"version": ultralytics.__version__, "path": ultralytics.__file__},
        "weights": a.weights, "device": str(device), "imgsz": a.imgsz, "conf": a.conf,
        "image": os.path.abspath(a.image), "image_shape": [int(h), int(w)],
        "n_instances": len(confs), "class_count": class_count, "confs": confs,
        "elapsed_ms": round((time.time() - t0) * 1000, 1),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
