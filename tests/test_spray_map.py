# -*- coding: utf-8 -*-
"""L0 决策层的性质测试。

不依赖真实照片，也不依赖感知侧环境：全部用合成图，纯 CPU。
"""
import numpy as np
import pytest

from delta import spray_map

GREEN = (0, 255, 0)      # BGR: exg=510, HSV hue=60 -> 判绿植
SOIL = (40, 60, 90)      # BGR: exg=-10 -> 判非绿植


def flat(side, color):
    img = np.zeros((side, side, 3), np.uint8)
    img[:, :] = color
    return img


def test_dose_boundaries():
    assert spray_map.determine_dose(0.0099) == 1
    assert spray_map.determine_dose(0.01) == 2
    assert spray_map.determine_dose(0.0999) == 2
    assert spray_map.determine_dose(0.1) == 3
    assert spray_map.determine_dose(0.2999) == 3
    assert spray_map.determine_dose(0.3) == 4


def test_buffer_px_from_physical_width():
    f = spray_map.protection_buffer_px
    assert f(3072, fov_side_m=0.60, buffer_m=0.005) == 26
    assert f(1080, fov_side_m=0.60, buffer_m=0.005) == 9
    assert f(768, fov_side_m=0.60, buffer_m=0.005) == 6      # 当前部署决策分辨率
    assert f(512, fov_side_m=0.60, buffer_m=0.005) == 4      # 缓冲核仍可分辨的下限附近
    assert f(480, fov_side_m=0.60, buffer_m=0.005) == 4
    assert f(64, fov_side_m=0.60, buffer_m=0.005) == 2          # 下限兜住
    # 迁移路径：旧标定 15px@3072 在 1 m 视场下可被米制参数精确复现
    assert f(3072, fov_side_m=1.0, buffer_m=15 / 3072) == 15


def test_working_resolution_only_downsamples():
    big = np.zeros((1920, 1920, 3), np.uint8)
    assert spray_map.working_resolution(big, 768).shape[:2] == (768, 768)
    small = np.zeros((600, 600, 3), np.uint8)
    assert spray_map.working_resolution(small, 768) is small     # 不放大
    assert spray_map.working_resolution(big, 0) is big           # 0 表示关闭
    rect = np.zeros((1080, 1920, 3), np.uint8)
    assert spray_map.working_resolution(rect, 768).shape[:2] == (432, 768)


def test_all_weed_field_is_heavy():
    res = spray_map.generate_spray_map(flat(200, GREEN), None)
    assert (res["spray_map"] == 4).all()
    assert res["stats"][0] == 0 and res["stats"]["spray_cells"] == 25
    assert not res["stats"]["has_crops"]


def test_bare_soil_field_is_skipped():
    res = spray_map.generate_spray_map(flat(200, SOIL), None)
    assert (res["spray_map"] == 1).all()
    assert res["stats"]["spray_cells"] == 0


def test_crop_covered_field_becomes_all_l0():
    img = flat(200, GREEN)
    res = spray_map.generate_spray_map(img, np.full((200, 200), 255, np.uint8))
    assert (res["spray_map"] == 0).all()
    assert res["stats"]["spray_cells"] == 0


def test_weed_is_subtracted_from_green_under_crop():
    """作物区域内的绿植不该算杂草：同一片绿植，盖作物后得分应下降。"""
    img = flat(100, GREEN)
    open_res = spray_map.generate_spray_map(img, None)
    crop = np.zeros((100, 100), np.uint8)
    crop[:, :] = 255
    covered = spray_map.generate_spray_map(img, crop)
    assert np.isclose(covered["density_map"][0, 0], 0.0)
    assert open_res["density_map"][0, 0] > 0.9


def test_score_and_density_ignore_fov_side():
    rng = np.random.default_rng(7)
    side = 600                       # 够大才能让两种视场算出不同的缓冲核
    green = rng.random((side, side)) < 0.35
    img = np.tile(np.array(SOIL, np.uint8), (side, side, 1))
    img[green] = GREEN
    crop = np.zeros((side, side), np.uint8)
    crop[:, :90] = 255

    a = spray_map.generate_spray_map(img, crop, fov_side_m=0.60)
    b = spray_map.generate_spray_map(img, crop, fov_side_m=1.00)
    np.testing.assert_array_equal(a["score_map"], b["score_map"])
    np.testing.assert_array_equal(a["density_map"], b["density_map"])

    assert a["stats"]["protection_buffer_px"] != b["stats"]["protection_buffer_px"]
    changed = np.argwhere(a["spray_map"] != b["spray_map"])
    prot_changed = np.argwhere(a["protection_map"] != b["protection_map"])
    for i, j in changed:                        # 等级只能因保护区判定而变
        assert any((i == p[0] and j == p[1]) for p in prot_changed)


def test_mask_resized_to_image():
    res = spray_map.generate_spray_map(flat(100, GREEN), np.full((50, 50), 255, np.uint8))
    assert res["spray_map"].shape == (5, 5)
    assert res["protection_mask"].shape == (100, 100)


def test_center_square_crop():
    out = spray_map.center_square(np.zeros((480, 640, 3), np.uint8))
    assert out.shape[:2] == (480, 480)
    same = flat(200, GREEN)
    assert spray_map.center_square(same) is same


def test_grid_centers_geometry():
    c = spray_map.grid_centers(5, 0.60)
    assert c.shape == (5, 5, 2)
    assert np.allclose(c[2, 2], [0.0, 0.0])
    expect = [-0.24, -0.12, 0.0, 0.12, 0.24]
    assert [round(float(v), 6) for v in c[0, :, 0]] == expect      # x 沿列 j 变化
    assert [round(float(v), 6) for v in c[:, 0, 1]] == expect      # y 沿行 i 变化
    # 最远格心的水平半径：角格格心是 0.24*sqrt(2)，不是视场边角的 0.30*sqrt(2)
    assert np.isclose(np.hypot(*c[0, 0]), 0.24 * np.sqrt(2))


def test_grid_centers_flip_rows_negates_y_only():
    a = spray_map.grid_centers(5, 0.60, flip_rows=False)
    b = spray_map.grid_centers(5, 0.60, flip_rows=True)
    np.testing.assert_array_equal(a[:, :, 0], b[:, :, 0])
    np.testing.assert_allclose(a[:, :, 1], -b[:, :, 1])


def test_format_matrix_roundtrip():
    res = spray_map.generate_spray_map(flat(100, GREEN), None)
    text = spray_map.format_matrix(res["spray_map"], res["stats"], "x.jpg")
    assert text.count("L4") >= 25
    assert "喷药格 : 25" in text
