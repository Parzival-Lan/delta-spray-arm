# -*- coding: utf-8 -*-
"""Delta 并联臂运动学：闭式 IK、三球交点 FK、雅可比与传动裕度。

坐标取向见 notes/00-conventions.md 第 1、4 节（与 MATLAB `optimization/` 一致：Z 向上为正，
动平台在下方故 z 为负）。结构参数取 notes/01-model.md 第 5 节的 GA 结果。

几何记号（第 i 臂，i = 0..2）：
    u_i   = (cos alpha_i, sin alpha_i)              该臂的径向单位向量
    A_i   = (R u_i, 0)                              曲柄轴心
    B_i   = ((R + L1 cos q_i) u_i, -L1 sin q_i)     曲柄柄端
    P_i   = (p + r u_i, p_z)                        动平台球铰
    |P_i - B_i| = L2                                从动臂长度约束
姿态恒为常量（R_matrix = I），所以只处理平移，本模块不含旋转部分。
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

MM = 1e-3
DEG = np.pi / 180
BRANCH_PLUS = 1
BRANCH_MINUS = -1


class Unreachable(ValueError):
    """目标点在该臂上无实数解（从动臂够不着）。"""

    def __init__(self, point, leg, defect):
        super().__init__(f"p={np.round(point, 6).tolist()} 第 {leg} 臂无实数解 "
                         f"(A^2+B^2-C^2 = {defect:.3e} < 0)")
        self.point, self.leg, self.defect = point, leg, defect


@dataclass(frozen=True)
class DeltaGeometry:
    R: float                      # 静平台外接圆半径
    r: float                      # 动平台外接圆半径
    L1: float                     # 主动臂（曲柄）
    L2: float                     # 从动臂（平行四边形支链）
    theta_min: float = -35.551414903564975 * DEG
    theta_max: float = 90.0 * DEG
    alpha: tuple = (0.0, 2.0 * np.pi / 3, 4.0 * np.pi / 3)
    branch: int = BRANCH_PLUS     # 全工程锁定的 IK 解支，见 00-conventions 第 5 节

    @classmethod
    def from_mm(cls, R, r, L1, L2, theta_min_deg=-35.551414903564975,
                theta_max_deg=90.0, **kw):
        return cls(R * MM, r * MM, L1 * MM, L2 * MM,
                   theta_min_deg * DEG, theta_max_deg * DEG, **kw)

    @property
    def D(self) -> float:
        """R - r：静、动平台外接圆半径之差，运动学公式里唯一进入的半径组合。"""
        return self.R - self.r

    @property
    def unit_vectors(self) -> np.ndarray:
        a = np.asarray(self.alpha)
        return np.stack([np.cos(a), np.sin(a)], axis=-1)

    def with_branch(self, branch: int) -> "DeltaGeometry":
        return replace(self, branch=branch)

    def crank_tip(self, q, i: int) -> np.ndarray:
        u = self.unit_vectors[i]
        return np.array([(self.R + self.L1 * np.cos(q[i])) * u[0],
                         (self.R + self.L1 * np.cos(q[i])) * u[1],
                         -self.L1 * np.sin(q[i])])

    def crank_tangent(self, q, i: int) -> np.ndarray:
        """dB_i/dq_i：曲柄柄端速度方向，雅可比与传动裕度都只用到它。"""
        u = self.unit_vectors[i]
        return self.L1 * np.array([-np.sin(q[i]) * u[0], -np.sin(q[i]) * u[1], -np.cos(q[i])])

    def platform_joint(self, p, i: int) -> np.ndarray:
        u = self.unit_vectors[i]
        return np.array([p[0] + self.r * u[0], p[1] + self.r * u[1], p[2]])


SAMPLE = DeltaGeometry.from_mm(138.1201643948795, 133.11878320186256,
                               406.6849938773964, 582.0832901594389)
"""样机结构参数：Delta_v0/optimization/delta_result.mat 的 GA 结果（2026-06-21）。"""


def ik_solutions(p, geom: DeltaGeometry) -> np.ndarray:
    """位置 -> 每臂两支关节角，形状 (3, 2)，列序 [BRANCH_MINUS, BRANCH_PLUS]。

    每臂独立求解（姿态恒定 ⇒ 三球约束解耦）。把 |P_i - B_i| = L2 展开成
        b_cos·cos(q) + a_sin·sin(q) = c
    的形式后用辅助角法，其中
        a_sin = p_z,  b_cos = D - proj_i,  proj_i = p_xy · u_i
        c = (L2^2 - D^2 - L1^2 - |p|^2 + 2 D proj_i) / (2 L1)
    判别式 a_sin^2 + b_cos^2 - c^2 < 0 即该臂无实数解。
    """
    p = np.asarray(p, dtype=float)
    u = geom.unit_vectors
    proj = u @ p[:2]
    a_sin = np.full(3, p[2])
    b_cos = geom.D - proj
    c = (geom.L2 ** 2 - geom.D ** 2 - geom.L1 ** 2 - p @ p + 2 * geom.D * proj) / (2 * geom.L1)

    defect = a_sin ** 2 + b_cos ** 2 - c ** 2
    out = np.full((3, 2), np.nan)
    ok = defect >= 0
    base = np.arctan2(a_sin[ok], b_cos[ok])
    off = np.arctan2(np.sqrt(defect[ok]), c[ok])
    out[ok, 0] = base - off
    out[ok, 1] = base + off
    return out


def ik(p, geom: DeltaGeometry = SAMPLE, *, branch: int | None = None) -> np.ndarray:
    """位置 -> 锁定分支上的关节角 (3,)。无实数解时抛 Unreachable。

    注意：本函数**不**判关节限位。限位是规划问题而不是解的存在性问题，
    混在一起会掩盖"解存在但被电机限位挡住"这一类完全不同的故障，
    需要时单独调用 within_limits()。
    """
    sols = ik_solutions(p, geom)
    col = 1 if (geom.branch if branch is None else branch) == BRANCH_PLUS else 0
    if np.any(np.isnan(sols[:, col])):
        raise Unreachable(p, int(np.argmax(np.isnan(sols[:, col]))), float("nan"))
    return sols[:, col]


def within_limits(q, geom: DeltaGeometry = SAMPLE, tol: float = 1e-9) -> bool:
    q = np.asarray(q, dtype=float)
    return bool(np.all(q >= geom.theta_min - tol) and np.all(q <= geom.theta_max + tol))


def ik_newton(p, geom: DeltaGeometry = SAMPLE, q0=None, *, tol=1e-13, max_iter=60):
    """数值 IK（逐臂标量 Newton），与闭式解互为交叉验证。

    因姿态恒定时三臂解耦，这里不需要完整的阻尼最小二乘：每臂一个标量方程
        f_i(q_i) = |P_i - B_i(q_i)|^2 - L2^2,   f_i' = -2 (P_i - B_i) . w_i
    q0 提供初值，也就自然提供分支——W2 里"用上一帧的解作初值 + 分支一致性检查"
    直接走这条路径即可。
    """
    p = np.asarray(p, dtype=float)
    q = np.array([0.0, 0.0, 0.0]) if q0 is None else np.asarray(q0, dtype=float).copy()
    for i in range(3):
        P = geom.platform_joint(p, i)
        for _ in range(max_iter):
            v = P - geom.crank_tip(q, i)
            f = v @ v - geom.L2 ** 2
            df = -2.0 * (v @ geom.crank_tangent(q, i))
            if df == 0.0:
                raise Unreachable(p, i, float("nan"))
            step = f / df
            q[i] -= step
            if abs(step) < tol:
                break
        else:
            raise Unreachable(p, i, float(f))
    return q


def fk_solutions(q, geom: DeltaGeometry):
    """关节角 -> 动平台中心的所有实数解（最多 2 个）。

    三球交点法。令 G_i = B_i - r u_i，则 |p - G_i| = L2：p 到三个固定点等距。
    两两相减只得 **2 个** 独立线性方程（第 3 个恒线性相关），几何上是一条直线；
    再回代一个球面方程得到一元二次，两个根即 FK 的两个分支。
    optimization_mimo/fk_delta.m 拼 3x3 线性方程组去解，那个矩阵秩恒为 2，故判废。
    """
    q = np.asarray(q, dtype=float)
    u = geom.unit_vectors
    G = np.stack([(geom.D + geom.L1 * np.cos(q[i])) * u[i] for i in range(3)])
    G = np.hstack([G, (-geom.L1 * np.sin(q))[:, None]])

    M = 2.0 * np.stack([G[0] - G[1], G[0] - G[2]])
    d = (G[0] @ G[0]) - np.array([G[1] @ G[1], G[2] @ G[2]])
    if np.linalg.matrix_rank(M) < 2:
        return []

    p0 = np.linalg.lstsq(M, d, rcond=None)[0]                  # 直线上离原点最近的点
    n = np.linalg.svd(M)[2][-1]
    n = n / np.linalg.norm(n)
    if n @ (G[0] - p0) < 0:
        n = -n

    g = p0 - G[0]
    b = n @ g
    c = g @ g - geom.L2 ** 2
    disc = b * b - c
    if disc < 0:
        return []
    s = np.sqrt(disc)
    return [p0 + (-b - s) * n, p0 + (-b + s) * n]


def fk(q, geom: DeltaGeometry = SAMPLE, *, near=None, prefer_low=True):
    """关节角 -> 动平台中心 (3,)。

    near 给定时取最接近它的解——轨迹跟随必须走这条路，否则会在两分支间跳变。
    未给定时取 z 更低的那个（样机作业区间在静平台下方，`calculate_workspace.m`
    的 `P(3) < 0` 判据是同一件事）。
    """
    sols = fk_solutions(q, geom)
    if not sols:
        raise Unreachable(np.asarray([np.nan] * 3), -1, float("nan"))
    if near is not None:
        return min(sols, key=lambda s: np.linalg.norm(s - np.asarray(near, dtype=float)))
    return min(sols, key=lambda s: s[2]) if prefer_low else max(sols, key=lambda s: s[2])


def leg_margin(q, p, geom: DeltaGeometry = SAMPLE) -> np.ndarray:
    """每臂传动裕度 = |sin(μ_i)|，μ_i 是该臂曲柄与从动臂的夹角，形状 (3,)。

    μ → 0 或 180°（两杆共线）就是死点/边界奇异：此时曲柄的切向速度方向 w_i 与从动臂
    垂直，绕该方向无法驱动末端，逆雅可比第 i 行的分母 b_i = (P_i-B_i).w_i 归零，
    速度增益发散。μ = 90° 时 |sin μ| = 1，传动最好。

    注意叉积算的是 cos μ（在 μ=90° 处归零，恰好把最优位置当成奇异），所以这里必须用点积。

    `p` 必须是与 `q` 自洽的位形（规划器给的目标点，或 fk(q)）。并联臂上「所有臂都取
    另一支」的 q 一般不对应任何真实位形，别指望 fk 替你补。同一 p 上单臂的两支解
    给出相同裕度——两支关于 B_i–P_i 连线镜像，三角形三边不变。
    """
    q, p = np.asarray(q, dtype=float), np.asarray(p, dtype=float)
    out = np.empty(3)
    for i in range(3):
        v = geom.platform_joint(p, i) - geom.crank_tip(q, i)
        w = geom.crank_tangent(q, i)
        out[i] = abs(v @ w) / (np.linalg.norm(v) * np.linalg.norm(w))
    return out


def jacobian(q, p, geom: DeltaGeometry = SAMPLE) -> np.ndarray:
    """3x3 逆雅可比：末端的平移速度 -> 三个关节角速度，即 qdot = J @ p_dot。

    由 |P_i - B_i| = L2 对时间求导得 (P_i - B_i) . p_dot = (P_i - B_i) . w_i * qdot_i，
    故第 i 行 = (P_i - B_i)^T / ((P_i - B_i) . w_i)。姿态恒定时只需处理平移三个自由度。
    奇异判据用 leg_margin()，不要用 det(J)：det 的量纲随尺度变，阈值没法通用。
    """
    q, p = np.asarray(q, dtype=float), np.asarray(p, dtype=float)
    J = np.empty((3, 3))
    for i in range(3):
        v = geom.platform_joint(p, i) - geom.crank_tip(q, i)
        J[i] = v / (v @ geom.crank_tangent(q, i))
    return J
