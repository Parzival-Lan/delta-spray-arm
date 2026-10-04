# -*- coding: utf-8 -*-
"""运动学测试。基准数值来自 notes/01-model.md 第 8 节（由 MATLAB 那套 IK 公式独立复算）。"""
import numpy as np
import pytest

from delta.kinematics import (BRANCH_MINUS, BRANCH_PLUS, DeltaGeometry, Unreachable,
                              fk, fk_solutions, ik, ik_newton, ik_solutions,
                              jacobian, leg_margin, within_limits)

MM = 1e-3
DEG = np.pi / 180


@pytest.fixture
def geom():
    from delta.kinematics import SAMPLE
    return SAMPLE


def qdeg(v):
    return np.rad2deg(np.asarray(v, dtype=float))


# ---------- 与 MATLAB 基准对齐 ----------

@pytest.mark.parametrize("p_mm,expect", [
    ((0, 0, -600), (22.96, 22.96, 22.96)),
    ((0, 0, -800), (46.17, 46.17, 46.17)),
    ((300, 300, -800), (47.88, 72.15, 90.00)),
])
def test_ik_matches_matlab_baseline(geom, p_mm, expect):
    got = qdeg(ik(np.asarray(p_mm) * MM, geom))
    assert np.allclose(got, expect, atol=0.02), got


def test_other_branch_is_far_away_and_out_of_limits(geom):
    sols = ik_solutions(np.array([0.0, 0.0, -600.0]) * MM, geom)
    assert np.allclose(qdeg(sols[:, 1]), 22.96, atol=0.02)
    assert np.all(qdeg(sols[:, 0]) < -180.0)          # theta- 在 -202 度附近
    assert not within_limits(sols[:, 0], geom)


def test_target_box_corner_sits_on_theta_max_limit(geom):
    q = ik(np.array([300.0, 300.0, -800.0]) * MM, geom)
    assert q[2] == pytest.approx(geom.theta_max, abs=1e-6)   # 零裕度：撞限位而非几何不可达
    assert within_limits(q, geom)
    beyond = ik(np.array([305.0, 305.0, -800.0]) * MM, geom)   # 解仍存在
    assert not within_limits(beyond, geom)                      # 但越出限位


def test_unreachable_point_raises(geom):
    with pytest.raises(Unreachable):
        ik(np.array([2000.0, 0.0, -100.0]) * MM, geom)


# ---------- FK / IK 往返 ----------

def random_q(geom, rng, n=100):
    return rng.uniform(geom.theta_min, geom.theta_max, size=(n, 3))


@pytest.mark.parametrize("seed", [1, 7, 2026])
def test_fk_ik_roundtrip(geom, seed):
    rng = np.random.default_rng(seed)
    for q in random_q(geom, rng, 50):
        p = fk(q, geom)
        back = ik(p, geom)
        assert np.allclose(back, q, atol=1e-9), (q, back)


def test_fk_returns_two_solutions_generically(geom):
    sols = fk_solutions(np.array([-10.0, 20.0, 45.0]) * DEG, geom)
    assert len(sols) == 2
    assert np.linalg.norm(sols[0] - sols[1]) > 1e-6      # 两个不同分支，不是重根


def test_fk_near_anchor_selects_branch(geom):
    q = np.array([-10.0, 20.0, 45.0]) * DEG
    sols = fk_solutions(q, geom)
    low = min(sols, key=lambda s: s[2])
    assert fk(q, geom, near=low) == pytest.approx(low, abs=1e-12)
    assert fk(q, geom) == pytest.approx(low, abs=1e-12)   # 默认取更低那个


def test_ik_newton_agrees_with_closed_form(geom):
    rng = np.random.default_rng(3)
    for q in random_q(geom, rng, 30):
        p = fk(q, geom)
        assert ik_newton(p, geom, q0=q) == pytest.approx(q, abs=1e-9)


def test_ik_newton_from_zero_converges_to_lower_branch(geom):
    rng = np.random.default_rng(5)
    q = random_q(geom, rng, 1)[0]
    p = fk(q, geom, near=None)
    qn = ik_newton(p, geom, q0=np.zeros(3))
    assert np.allclose(ik(p, geom), qn, atol=1e-9) or np.allclose(
        ik(p, geom, branch=BRANCH_MINUS), qn, atol=1e-9)


# ---------- 雅可比与裕度 ----------

def test_jacobian_matches_finite_difference(geom):
    rng = np.random.default_rng(11)
    q = random_q(geom, rng, 1)[0]
    p0 = fk(q, geom)
    d = 1e-7
    fwd = np.empty((3, 3))
    for i in range(3):
        e = np.zeros(3); e[i] = d
        fwd[:, i] = (fk(q + e, geom, near=p0) - fk(q - e, geom, near=p0)) / (2 * d)
    assert fwd == pytest.approx(np.linalg.inv(jacobian(q, p0, geom)), rel=1e-6)


def test_leg_margin_zero_when_crank_and_rod_collinear(geom):
    """定义性测试：曲柄与从动臂共线必须给 0，垂直必须给 1。"""
    q = np.array([20.0, -5.0, 40.0]) * DEG
    i = 0
    B = geom.crank_tip(q, i)
    u = geom.unit_vectors[i]
    crank_dir = np.array([np.cos(q[i]) * u[0], np.cos(q[i]) * u[1], -np.sin(q[i])])
    w = geom.crank_tangent(q, i)
    for vec, expect in ((crank_dir, 0.0), (w / np.linalg.norm(w), 1.0)):
        p = B + geom.L2 * vec
        p[:2] -= geom.r * u                      # P_i = p + r u_i  =>  p = P_i - r u_i
        assert leg_margin(q, p, geom)[i] == pytest.approx(expect, abs=1e-12)


def test_leg_margin_is_branch_independent_at_same_point(geom):
    p = np.array([200.0, -100.0, -600.0]) * MM
    qa = ik(p, geom, branch=BRANCH_PLUS)
    qb = ik(p, geom, branch=BRANCH_MINUS)
    assert leg_margin(qa, p, geom) == pytest.approx(leg_margin(qb, p, geom), abs=1e-12)


def test_transmission_margin_terrain_in_the_work_box(geom):
    """600x600 框、四个作业深度上的最紧传动裕度（粗网格，全部限位内）。"""
    xs = np.linspace(-0.3, 0.3, 7)
    worst = {}
    for z in (-0.35, -0.50, -0.65, -0.80):
        vals = []
        for x in xs:
            for y in xs:
                p = np.array([x, y, z])
                q = ik(p, geom)
                assert within_limits(q, geom)
                vals.append(leg_margin(q, p, geom).min())
        worst[round(z, 2)] = min(vals)
    assert worst[-0.35] > 0.50 and worst[-0.65] > 0.60
    assert worst[-0.80] < 0.25          # 最深一层的角点是全框最紧处，会触发告警


# ---------- 约定本身 ----------

def test_geometry_from_mm_and_branch_switch(geom):
    g = DeltaGeometry.from_mm(138.1201643948795, 133.11878320186256,
                              406.6849938773964, 582.0832901594389)
    assert g.R == pytest.approx(0.1381201643948795)
    assert g.D == pytest.approx(5.0014 * MM, abs=1e-7)
    assert g.with_branch(BRANCH_MINUS).branch == BRANCH_MINUS
    assert geom.branch == BRANCH_PLUS


def test_platform_and_crank_points_satisfy_rod_length(geom):
    rng = np.random.default_rng(13)
    q = random_q(geom, rng, 20)
    for qi in q:
        p = fk(qi, geom)
        for i in range(3):
            v = geom.platform_joint(p, i) - geom.crank_tip(qi, i)
            assert np.linalg.norm(v) == pytest.approx(geom.L2, abs=1e-12)
