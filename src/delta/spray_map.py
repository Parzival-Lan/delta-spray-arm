# -*- coding: utf-8 -*-
"""5x5 变量施药决策：田间图 + 作物掩膜 -> L0-L4 等级矩阵与格心坐标。

纯函数层：只依赖 numpy 与 cv2，不做推理、不读配置文件、不落盘。
物理尺寸一律用米表达，像素换算集中在 protection_buffer_px 一处。
"""
from __future__ import annotations

import numpy as np
import cv2

FOV_SIDE_M = 0.60
GRID_SIZE = 5
PROTECTION_BUFFER_M = 0.005
PROTECTION_THRESHOLD = 0.02
EXG_THRESHOLD = 20.0

DOSE_NAMES = {0: "L0", 1: "L1", 2: "L2", 3: "L3", 4: "L4"}


def center_square(image):
    """裁掉非方形帧的中心最大正方形。

    5x5 分区与"视场为正方形"是同一个假设的两个结果，所以必须在打分之前完成。
    """
    h, w = image.shape[:2]
    if h == w:
        return image
    s = min(h, w)
    y0, x0 = (h - s) // 2, (w - s) // 2
    return image[y0:y0 + s, x0:x0 + s]


def extract_green_mask(image, exg_threshold=EXG_THRESHOLD):
    """Raw EXG + 固定阈值 + HSV 绿色校验。

    用固定阈值而非 Otsu：Otsu 阈值随图变化，跨图不可比。HSV 校验剔除亮土、反光、枯叶。
    """
    f = image.astype(np.float32)
    b, g, r = cv2.split(f)
    exg = 2 * g - r - b

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h_ok = (hsv[:, :, 0] >= 30) & (hsv[:, :, 0] <= 90)
    s_ok = hsv[:, :, 1] >= 10

    mask = (exg >= exg_threshold) & h_ok & s_ok
    return mask.astype(np.float32)


def determine_dose(density):
    """加权杂草密度 -> 剂量等级。L1 为跳过；L0 由保护区判定，不在此函数内。"""
    if density < 0.01:
        return 1
    if density < 0.1:
        return 2
    if density < 0.3:
        return 3
    return 4


def protection_buffer_px(side, fov_side_m=FOV_SIDE_M, buffer_m=PROTECTION_BUFFER_M):
    """把物理缓冲宽度换算成膨胀核像素数。

    旧实现锚在 "15 px @ 3072 px = 1 m 视场"，视场一改就系统性偏差；这里以米为准。
    """
    return max(2, int(round(buffer_m / fov_side_m * side)))


def build_protection_mask(crop_mask, buffer_px):
    binary = (np.asarray(crop_mask) > 0).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (buffer_px, buffer_px))
    return (cv2.dilate(binary, kernel, iterations=1) > 0).astype(np.float32)


def generate_spray_map(image, crop_mask=None, *, grid_size=GRID_SIZE,
                       fov_side_m=FOV_SIDE_M, buffer_m=PROTECTION_BUFFER_M,
                       protection_threshold=PROTECTION_THRESHOLD,
                       exg_threshold=EXG_THRESHOLD):
    """由一张方形图像计算 grid_size x grid_size 的施药决策。

    image      : BGR (H, W, 3)，需已裁成正方形
    crop_mask  : (H, W)，>0 为作物；None 表示画面内无作物

    score_map / density_map 只由像素决定，与 fov_side_m 无关；视场大小只通过缓冲核
    宽度影响 L0 保护区判定。
    """
    h, w = image.shape[:2]
    cell_h, cell_w = h // grid_size, w // grid_size

    green_mask = extract_green_mask(image, exg_threshold)
    has_crops = crop_mask is not None and np.any(np.asarray(crop_mask) > 0)

    if has_crops:
        crop = (np.asarray(crop_mask) > 0).astype(np.float32)
        if crop.shape[:2] != (h, w):
            crop = cv2.resize(crop, (w, h), interpolation=cv2.INTER_NEAREST)
        weed_mask = green_mask * (1 - crop)
        buffer_px = protection_buffer_px(min(h, w), fov_side_m, buffer_m)
        protection_mask = build_protection_mask(crop, buffer_px)
    else:
        crop = np.zeros((h, w), np.float32)
        weed_mask = green_mask.copy()
        buffer_px = 0
        protection_mask = np.zeros_like(green_mask)

    n = grid_size
    spray_map = np.zeros((n, n), np.uint8)
    density_map = np.zeros((n, n), np.float32)
    score_map = np.zeros((n, n), np.float32)
    protection_map = np.zeros((n, n), np.float32)
    crop_density_map = np.zeros((n, n), np.float32)

    # 径向衰减权重：格心 1.0 递减到角上 0.05，压低格边界处杂草对邻格得分的贡献
    yy, xx = np.ogrid[:cell_h, :cell_w]
    cy, cx = cell_h / 2, cell_w / 2
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    max_dist = np.sqrt(cx ** 2 + cy ** 2)
    weight_mask = np.clip(1.0 - 0.95 * (dist / max_dist), 0.05, 1.0)
    weight_sum = float(np.sum(weight_mask))

    for i in range(n):
        for j in range(n):
            y1, y2 = i * cell_h, (i + 1) * cell_h
            x1, x2 = j * cell_w, (j + 1) * cell_w

            cell_weed = weed_mask[y1:y2, x1:x2]
            density_map[i, j] = float(np.sum(cell_weed) / cell_weed.size)
            score_map[i, j] = float(np.sum(cell_weed * weight_mask)) / weight_sum

            cell_prot = protection_mask[y1:y2, x1:x2]
            protection_map[i, j] = float(np.sum(cell_prot) / cell_prot.size)

            if has_crops:
                cell_crop = crop[y1:y2, x1:x2]
                crop_density_map[i, j] = float(np.sum(cell_crop) / cell_crop.size)
                spray_map[i, j] = (0 if protection_map[i, j] > protection_threshold
                                   else determine_dose(score_map[i, j]))
            else:
                spray_map[i, j] = determine_dose(score_map[i, j])

    stats = {k: int((spray_map == k).sum()) for k in range(5)}
    stats["total"] = n * n
    stats["spray_cells"] = stats[2] + stats[3] + stats[4]
    stats.update(has_crops=bool(has_crops), protection_buffer_px=buffer_px,
                 fov_side_m=float(fov_side_m), grid_size=n)

    return {"spray_map": spray_map, "density_map": density_map, "score_map": score_map,
            "protection_map": protection_map, "crop_density_map": crop_density_map,
            "weed_mask": weed_mask, "protection_mask": protection_mask,
            "green_mask": green_mask, "stats": stats}


def grid_centers(grid_size=GRID_SIZE, fov_side_m=FOV_SIDE_M, flip_rows=False):
    """格索引 (i, j) -> 视场平面内的格心坐标 (x, y)，单位 m，原点 = 视场中心。

    i 沿图像向下、j 沿图像向右。flip_rows=True 时行序取反（图像第 0 行对应 +y 一侧）；
    方向由相机与臂的安装关系实测决定，见 notes/00-conventions.md 第 8 节。
    """
    cell = fov_side_m / grid_size
    along = (np.arange(grid_size) + 0.5) * cell - fov_side_m / 2
    rows = -along if flip_rows else along
    x = np.broadcast_to(along, (grid_size, grid_size))
    y = np.broadcast_to(rows[:, None], (grid_size, grid_size))
    return np.stack([x, y], axis=-1).astype(np.float64)


def format_matrix(spray_map, stats=None, basename=""):
    """5x5 等级的可读文本。"""
    n = len(spray_map)
    lines = [f"图像: {basename}"] if basename else []
    lines.append(f"施药等级矩阵 ({n}x{n}, 行=自上而下, 列=自左而右):")
    lines += ["  " + " ".join(f"{DOSE_NAMES[int(v)]:>2}" for v in row) for row in spray_map]
    if stats:
        lines += ["", f"总格数 : {stats['total']}",
                  f"喷药格 : {stats['spray_cells']} (L2+L3+L4)",
                  "L0 保护: {}   L1 跳过: {}   L2 轻: {}   L3 中: {}   L4 重: {}".format(
                      *[stats[k] for k in range(5)])]
    return "\n".join(lines)
