# -*- coding: utf-8 -*-
"""图片 -> 处方图 JSON。W4 的交付物，也是 W5 端到端脚本的输入契约。

控制侧与感知侧在同一环境里跑不起来（fork ultralytics + torch 在另一个环境），
所以这里默认调 worker 推理；已有掩膜时用 --mask 跳过推理，纯 CPU 也能复算处方。
"""
import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from delta import perception, spray_map  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="generate a variable-rate prescription map")
    ap.add_argument("image")
    ap.add_argument("--out", default=None)
    ap.add_argument("--mask", default=None, help="已有 crop_mask 的 .npz；给了就跳过推理")
    ap.add_argument("--python", default=None, help="感知侧 Python 解释器（含 torch）")
    ap.add_argument("--weights", default=None)
    ap.add_argument("--ultralytics-root", default=None, help="fork 版 ultralytics 的上级目录")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=512)
    ap.add_argument("--device", default=None)
    ap.add_argument("--fov-side", type=float, default=spray_map.FOV_SIDE_M)
    ap.add_argument("--grid", type=int, default=spray_map.GRID_SIZE)
    ap.add_argument("--process-size", type=int, default=768,
                    help="决策分辨率边长：裁方后降采样到此尺寸再推理与决策。"
                         "768 是当前工控机的性能妥协，换边缘设备只改这里（0=不降采样）")
    ap.add_argument("--buffer-m", type=float, default=spray_map.PROTECTION_BUFFER_M)
    ap.add_argument("--threshold", type=float, default=spray_map.PROTECTION_THRESHOLD)
    ap.add_argument("--exg", type=float, default=spray_map.EXG_THRESHOLD)
    ap.add_argument("--flip-rows", action="store_true", help="图像第 0 行对应 +y 一侧")
    ap.add_argument("--min-level", type=int, default=2,
                    help="低于此等级的格不进 points（默认 2，即 L0 保护区与 L1 跳过都不喷）")
    a = ap.parse_args()

    img = cv2.imread(a.image)
    if img is None:
        sys.exit(f"图片读取失败: {a.image}")
    img = spray_map.center_square(img)
    source_side = int(min(img.shape[:2]))
    img = spray_map.working_resolution(img, a.process_size)

    t0 = time.perf_counter()
    if a.mask:
        crop, meta = np.load(a.mask)["crop_mask"], {"skipped_inference": True, "mask": str(a.mask)}
    else:
        r = perception.crop_mask(img, python_exe=a.python, weights=a.weights,
                                 ultralytics_root=a.ultralytics_root, conf=a.conf,
                                 imgsz=a.imgsz, device=a.device)
        crop, meta = r["crop_mask"], r["meta"]
    infer_ms = round((time.perf_counter() - t0) * 1000)

    res = spray_map.generate_spray_map(
        img, crop, grid_size=a.grid, fov_side_m=a.fov_side, buffer_m=a.buffer_m,
        protection_threshold=a.threshold, exg_threshold=a.exg)

    centers = spray_map.grid_centers(a.grid, a.fov_side, a.flip_rows)
    points = [{"i": int(i), "j": int(j),
               "level": int(res["spray_map"][i, j]),
               "score": round(float(res["score_map"][i, j]), 6),
               "x_m": round(float(centers[i, j, 0]), 6),
               "y_m": round(float(centers[i, j, 1]), 6)}
              for i in range(a.grid) for j in range(a.grid)
              if int(res["spray_map"][i, j]) >= max(a.min_level, 2)]

    out = Path(a.out) if a.out else Path(f"{Path(a.image).stem}.prescription.json")
    payload = {
        "source_image": str(Path(a.image).resolve()),
        "source_side_px": source_side,
        "process_size_px": int(min(img.shape[:2])),
        "grid_size": a.grid, "fov_side_m": a.fov_side,
        "cell_side_m": round(a.fov_side / a.grid, 6),
        "buffer_m": a.buffer_m, "protection_threshold": a.threshold,
        "exg_threshold": a.exg, "flip_rows": bool(a.flip_rows),
        "min_level": a.min_level,
        "spray_map": res["spray_map"].tolist(),
        "score_map": [[round(float(v), 6) for v in row] for row in res["score_map"]],
        "density_map": [[round(float(v), 6) for v in row] for row in res["density_map"]],
        "protection_map": [[round(float(v), 6) for v in row] for row in res["protection_map"]],
        "stats": {str(k): v for k, v in res["stats"].items()},
        "points": points,
        "perception": meta,
        "timings_ms": {"inference": infer_ms},
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(spray_map.format_matrix(res["spray_map"], res["stats"], Path(a.image).name))
    print(f"\n需喷点 {len(points)} 个 (等级 >= {a.min_level})，缓冲核 {res['stats']['protection_buffer_px']} px")
    print(f"写出: {out}")


if __name__ == "__main__":
    main()
