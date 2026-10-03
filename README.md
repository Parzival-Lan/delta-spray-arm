# delta-spray-arm

Delta 并联臂视觉靶向喷药闭环：真实田间照片 → 分割与处方图 → 轨迹规划 → 控制器 → 自写数值仿真。

姿态在作业全程恒为常量，本仓库不含姿态规划；仿真为自写 RK4 数值积分，不使用物理引擎。

## 架构

```
真实田间照片 → YOLO11-HLB 分割 → 作物掩膜 + 缓冲带 + EXG 杂草
   → 5×5 距离加权打分 → L0–L4 处方图
   → homography（相机视场 80×80cm ↔ 臂工作空间）
   → 喷施点序列 (x, y, level)
   → 轨迹规划（直线插补 + S 曲线 + 格间 blend）
   → 控制器（PD + 重力前馈 → 计算力矩）
   → 自写数值仿真（RK4）→ 可视化 + 喷阀时序
```

<!-- TODO(前期准备): 换成 mermaid 或 drawio 架构图 -->

## 目录

| 路径 | 内容 |
|---|---|
| `src/delta/kinematics.py` | FK / IK / 雅可比 |
| `src/delta/dynamics.py` | `M(q)q̈ = τ − C q̇ − g` 简化模型 |
| `src/delta/trajectory.py` | 五次多项式 / S 曲线 / 直线插补 / 栅格路径 |
| `src/delta/controller.py` | PD + 重力前馈 / 计算力矩 |
| `src/delta/simulator.py` | RK4 积分器，`sim_dt` 与 `ctrl_dt` 分离 |
| `src/delta/viz.py` | matplotlib 3D 线框 |
| `src/delta/perception.py` | 分割 → 掩膜 |
| `src/delta/spray_map.py` | 打分 → L0–L4 处方图 → 喷施点序列 |
| `notes/` | 坐标系约定、模型说明、各模块 `why-*.md` 决策记录 |

## 复现

<!-- TODO(W5): scripts/run_e2e.py 一条命令跑完后补齐 -->

## 约定

单位、坐标系与姿态约定见 [`notes/00-conventions.md`](notes/00-conventions.md)。
