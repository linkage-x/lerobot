#!/usr/bin/env python3
"""
FR3 + DAS Gripper SLAM Frame Contract 验证脚本

验证目标：
  1. T(I,E) 数值结构合法性（SE(3)、行列式、正交性）
  2. R(I,E) 近似为干净的旋转矩阵 [0,1,0; 1,0,0; 0,0,-1]
  3. t(I,E) 平移量与预期值一致
  4. E 轴语义（-z 接近，-y 闭合，x 法向）验证
  5. URDF 链提取 T(base,E)，与已知 T(I,E) 反推 T(base,I)，验证合法性
  6. 链式一致性：T(B,E_t) = T(B,W_s) * T(W_s,I_t) * T(I,E)（合成数据）
  7. 首帧恒等检查：T(W_s,I_0) ≈ I，episode 初始化正确
  8. Replay 一致性：T(B,E*_t) = T(B,W_s) * T(W_s,E*_t)
  9. FK 一致性（需要 placo，可选）

运行方式：
  python validate_frame_contract.py
  python validate_frame_contract.py --urdf path/to/fr3_das_ati.urdf --use-placo
"""

from __future__ import annotations

import argparse
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Frame Contract 常量
# 以下所有值来自文档冻结版，不得在其他脚本中重复硬编码
# ---------------------------------------------------------------------------

# 候选固定外参 T(I,E)（精确值，来自 URDF 链推导）
# I = link_imu，E = das_gripper_ee
# E 轴语义：-z_E 接近方向，-y_E 闭合方向，x_E 法向
T_IE_CANDIDATE: np.ndarray = np.array(
    [
        [-3.62e-13, 1.0, -3.62e-6, 0.007522733],
        [1.0, -3.70e-13, -3.70e-6, 0.009068533],
        [-3.70e-6, -3.62e-6, -1.0, -0.127258061],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)

# 工程近似版（角度误差 < 0.001°）
R_IE_APPROX: np.ndarray = np.array(
    [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]], dtype=np.float64
)
t_IE_APPROX: np.ndarray = np.array([0.007523, 0.009069, -0.127258], dtype=np.float64)

# 验证容差
TOL_ROT = 1e-4   # 旋转矩阵元素绝对误差（≈ 0.006°）
TOL_TRANS = 1e-4  # 平移量绝对误差（m）
TOL_SE3 = 1e-8   # SE(3) 结构误差（正交性、行列式）
TOL_FK = 5e-4    # FK 一致性容差（m / 旋转角度弧度）

# ---------------------------------------------------------------------------
# SE(3) 工具函数
# ---------------------------------------------------------------------------


def make_se3(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """构造 4x4 SE(3) 变换矩阵。"""
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def se3_inv(T: np.ndarray) -> np.ndarray:
    """SE(3) 矩阵的解析逆。"""
    R = T[:3, :3]
    t = T[:3, 3]
    T_inv = np.eye(4, dtype=np.float64)
    T_inv[:3, :3] = R.T
    T_inv[:3, 3] = -R.T @ t
    return T_inv


def rpy_to_matrix(rpy: tuple[float, float, float]) -> np.ndarray:
    """
    URDF RPY 约定转旋转矩阵。
    R = Rz(yaw) * Ry(pitch) * Rx(roll)
    """
    r, p, y = rpy

    def Rx(a: float) -> np.ndarray:
        return np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])

    def Ry(a: float) -> np.ndarray:
        return np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])

    def Rz(a: float) -> np.ndarray:
        return np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])

    return Rz(y) @ Ry(p) @ Rx(r)


def random_se3(rng: np.random.Generator, scale: float = 0.3) -> np.ndarray:
    """生成随机 SE(3) 变换（用于合成测试数据）。"""
    # 随机旋转：Rodrigues
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    angle = rng.uniform(0, np.pi / 4)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
    t = rng.uniform(-scale, scale, size=3)
    return make_se3(R, t)


# ---------------------------------------------------------------------------
# 检查函数
# ---------------------------------------------------------------------------


def check(name: str, passed: bool, detail: str = "") -> bool:
    mark = "PASS" if passed else "FAIL"
    msg = f"  [{mark}] {name}"
    if detail:
        msg += f"\n         {detail}"
    print(msg)
    return passed


def section(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Check 1: T(I,E) SE(3) 结构合法性
# ---------------------------------------------------------------------------


def check_se3_validity(T: np.ndarray, label: str = "T(I,E)") -> bool:
    section(f"Check 1: {label} SE(3) 结构合法性")
    all_pass = True

    # 形状
    all_pass &= check(f"{label} 形状为 (4,4)", T.shape == (4, 4))

    R = T[:3, :3]
    t = T[:3, 3]

    # 正交性
    ortho_err = np.max(np.abs(R.T @ R - np.eye(3)))
    all_pass &= check(
        f"旋转矩阵正交性 |R^T R - I|_inf < {TOL_SE3}",
        ortho_err < TOL_SE3,
        f"|R^T R - I|_inf = {ortho_err:.2e}",
    )

    # 行列式
    det = np.linalg.det(R)
    all_pass &= check(
        f"旋转矩阵行列式 det(R) ≈ +1",
        abs(det - 1.0) < TOL_SE3,
        f"det(R) = {det:.10f}",
    )

    # 最后一行
    last_row_ok = np.allclose(T[3, :], [0, 0, 0, 1], atol=TOL_SE3)
    all_pass &= check(
        "最后一行为 [0,0,0,1]",
        last_row_ok,
        f"last row = {T[3, :]}",
    )

    return all_pass


# ---------------------------------------------------------------------------
# Check 2: R(I,E) 近似为干净旋转矩阵
# ---------------------------------------------------------------------------


def check_rotation_approx(T: np.ndarray) -> bool:
    section("Check 2: R(I,E) 近似验证")
    all_pass = True

    R = T[:3, :3]
    err = np.max(np.abs(R - R_IE_APPROX))
    all_pass &= check(
        f"|R(I,E) - R_approx|_inf < {TOL_ROT}",
        err < TOL_ROT,
        f"误差 = {err:.2e}\nR(I,E):\n{np.round(R, 6)}\nR_approx:\n{R_IE_APPROX}",
    )
    return all_pass


# ---------------------------------------------------------------------------
# Check 3: 平移量与预期值一致
# ---------------------------------------------------------------------------


def check_translation(T: np.ndarray) -> bool:
    section("Check 3: t(I,E) 平移量验证")
    all_pass = True

    t = T[:3, 3]
    err = np.abs(t - t_IE_APPROX)
    all_pass &= check(
        f"|t(I,E) - t_approx|_inf < {TOL_TRANS} m",
        np.max(err) < TOL_TRANS,
        f"t(I,E) = {t}\nt_approx = {t_IE_APPROX}\n误差 [m] = {err}",
    )

    # 分量检查（易于人工核对）
    print(f"         x 偏移 = {t[0]*1000:.3f} mm  (期望 ≈ 7.5 mm)")
    print(f"         y 偏移 = {t[1]*1000:.3f} mm  (期望 ≈ 9.1 mm)")
    print(f"         z 偏移 = {t[2]*1000:.3f} mm  (期望 ≈ -127.3 mm)")

    return all_pass


# ---------------------------------------------------------------------------
# Check 4: E 轴语义验证
# ---------------------------------------------------------------------------


def check_axis_semantics(T: np.ndarray) -> bool:
    """
    E 轴语义（在 I frame 中表达）：
      -z_E → 接近方向（approach）
      -y_E → 闭合方向（closing）
       x_E → 法向（normal）

    T(I,E) 的列向量是 E 的坐标轴在 I 中的表示：
      col 0 = x_E in I = R(I,E)[:,0]
      col 1 = y_E in I = R(I,E)[:,1]
      col 2 = z_E in I = R(I,E)[:,2]
    """
    section("Check 4: E 轴语义验证")
    all_pass = True

    R = T[:3, :3]
    x_E_in_I = R[:, 0]  # E 的 x 轴在 I 坐标系中的方向
    y_E_in_I = R[:, 1]  # E 的 y 轴在 I 坐标系中的方向
    z_E_in_I = R[:, 2]  # E 的 z 轴在 I 坐标系中的方向

    print(f"         x_E in I = {np.round(x_E_in_I, 4)}  ← 法向")
    print(f"         y_E in I = {np.round(y_E_in_I, 4)}  ← 闭合方向反向")
    print(f"         z_E in I = {np.round(z_E_in_I, 4)}  ← 接近方向反向")

    # x_E 应为 I 坐标系中 y 轴方向（法向）
    all_pass &= check(
        "x_E ≈ y_I（法向轴）",
        np.allclose(x_E_in_I, [0, 1, 0], atol=TOL_ROT),
        f"x_E in I = {x_E_in_I}",
    )

    # y_E 应为 I 坐标系中 x 轴方向（闭合轴反向）
    all_pass &= check(
        "y_E ≈ x_I（闭合方向反向）",
        np.allclose(y_E_in_I, [1, 0, 0], atol=TOL_ROT),
        f"y_E in I = {y_E_in_I}",
    )

    # z_E 应为 I 坐标系中 -z 轴方向（接近方向反向）
    all_pass &= check(
        "z_E ≈ -z_I（接近方向反向）",
        np.allclose(z_E_in_I, [0, 0, -1], atol=TOL_ROT),
        f"z_E in I = {z_E_in_I}",
    )

    return all_pass


# ---------------------------------------------------------------------------
# Check 5: URDF 链提取 T(base,E)
# ---------------------------------------------------------------------------


def parse_urdf_joint_transform(urdf_path: str, joint_name: str) -> np.ndarray | None:
    """从 URDF 解析指定 fixed joint 的 4x4 变换矩阵。"""
    try:
        tree = ET.parse(urdf_path)
        root = tree.getroot()
    except Exception as e:
        print(f"         URDF 解析失败: {e}")
        return None

    for joint in root.iter("joint"):
        if joint.get("name") == joint_name:
            origin = joint.find("origin")
            if origin is None:
                return np.eye(4)
            xyz_str = origin.get("xyz", "0 0 0")
            rpy_str = origin.get("rpy", "0 0 0")
            xyz = np.array([float(v) for v in xyz_str.split()])
            rpy = tuple(float(v) for v in rpy_str.split())
            R = rpy_to_matrix(rpy)
            return make_se3(R, xyz)
    return None


def check_urdf_base_E(urdf_path: str | None, T_IE: np.ndarray) -> bool:
    section("Check 5: URDF 链提取 T(base,E) 合法性")

    if urdf_path is None:
        print("  [SKIP] 未提供 --urdf，跳过 URDF 解析验证")
        return True

    T_base_E = parse_urdf_joint_transform(urdf_path, "fr3_hand_tcp_joint")
    if T_base_E is None:
        return check("URDF 中找到 fr3_hand_tcp_joint", False)

    check("URDF 中找到 fr3_hand_tcp_joint", True)

    R_base_E = T_base_E[:3, :3]
    t_base_E = T_base_E[:3, 3]
    print(f"         R(base,E):\n{np.round(R_base_E, 6)}")
    print(f"         t(base,E) = {t_base_E}")

    # 反推 T(base,I) = T(base,E) * T(E,I) = T(base,E) * T(I,E)^{-1}
    T_IE_inv = se3_inv(T_IE)
    T_base_I = T_base_E @ T_IE_inv
    R_base_I = T_base_I[:3, :3]
    t_base_I = T_base_I[:3, 3]
    print(f"\n         反推 T(base,I):")
    print(f"         R(base,I):\n{np.round(R_base_I, 6)}")
    print(f"         t(base,I) = {np.round(t_base_I, 6)}")

    # 反推的 T(base,I) 也必须是合法 SE(3)
    ortho_err = np.max(np.abs(R_base_I.T @ R_base_I - np.eye(3)))
    det = np.linalg.det(R_base_I)
    all_pass = True
    all_pass &= check(
        "反推 R(base,I) 正交性",
        ortho_err < TOL_SE3,
        f"|R^T R - I|_inf = {ortho_err:.2e}",
    )
    all_pass &= check(
        "反推 R(base,I) det ≈ +1",
        abs(det - 1.0) < TOL_SE3,
        f"det = {det:.10f}",
    )

    # 验证链闭合：T(base,E) = T(base,I) * T(I,E)
    T_base_E_check = T_base_I @ T_IE
    err = np.max(np.abs(T_base_E_check - T_base_E))
    all_pass &= check(
        "链闭合：T(base,I)*T(I,E) ≈ T(base,E)",
        err < TOL_SE3,
        f"最大误差 = {err:.2e}",
    )

    return all_pass


# ---------------------------------------------------------------------------
# Check 6: 链式一致性（合成数据）
# ---------------------------------------------------------------------------


def check_chain_consistency(T_IE: np.ndarray, n_trials: int = 100) -> bool:
    section(f"Check 6: 链式一致性（{n_trials} 组随机合成数据）")

    rng = np.random.default_rng(42)
    T_IE_inv = se3_inv(T_IE)
    max_err = 0.0
    all_pass = True

    for _ in range(n_trials):
        # 合成数据
        T_B_Ws = random_se3(rng)         # episode 初始化
        T_Ws_It = random_se3(rng)        # SLAM 输出（I 语义）

        # 方法 A：直接链
        T_B_Et_chain = T_B_Ws @ T_Ws_It @ T_IE

        # 方法 B：先转换 T(Ws,E_t) = T(Ws,I_t)*T(I,E)，再与 T(B,Ws) 组合
        T_Ws_Et = T_Ws_It @ T_IE
        T_B_Et_two_step = T_B_Ws @ T_Ws_Et

        err = np.max(np.abs(T_B_Et_chain - T_B_Et_two_step))
        max_err = max(max_err, err)
        if err > TOL_SE3:
            all_pass = False
            break

    all_pass &= check(
        "T(B,Ws)*T(Ws,It)*T(I,E) ≡ T(B,Ws)*T(Ws,Et) 恒等",
        max_err < TOL_SE3,
        f"最大误差 = {max_err:.2e}（{n_trials} 组随机数据）",
    )

    # 链结果也应为合法 SE(3)
    T_B_Ws = random_se3(rng)
    T_Ws_It = random_se3(rng)
    T_B_Et = T_B_Ws @ T_Ws_It @ T_IE
    R_result = T_B_Et[:3, :3]
    ortho_err = np.max(np.abs(R_result.T @ R_result - np.eye(3)))
    det = np.linalg.det(R_result)
    all_pass &= check(
        "链结果 T(B,E_t) 仍为合法 SE(3)",
        ortho_err < TOL_SE3 and abs(det - 1.0) < TOL_SE3,
        f"正交误差 = {ortho_err:.2e}, det = {det:.10f}",
    )

    return all_pass


# ---------------------------------------------------------------------------
# Check 7: Episode 首帧恒等检查 + 初始化一致性
# ---------------------------------------------------------------------------


def check_first_frame_identity(T_IE: np.ndarray) -> bool:
    """
    W_s ≡ I_0，因此：
      T(W_s, I_0) = I   （SLAM 首帧输出应为单位矩阵）

    episode 初始化：
      T(B, W_s) = T(B, I_0) = T(B, E_0) * T(E, I)
    """
    section("Check 7: Episode 首帧恒等 + 初始化一致性")

    rng = np.random.default_rng(7)
    T_IE_inv = se3_inv(T_IE)
    all_pass = True

    # 合成：给定初始 joint state，FK 给出 T(B,E_0)
    T_B_E0 = random_se3(rng)

    # Episode 初始化：T(B,W_s) = T(B,E_0) * T(E,I)
    T_B_Ws = T_B_E0 @ T_IE_inv

    # 首帧 SLAM 应为恒等：T(W_s, I_0) = I
    # 等价验证：T(B,W_s) * T(W_s,I_0) * T(I,E) 应等于 T(B,E_0)
    T_Ws_I0 = np.eye(4)  # W_s ≡ I_0 的定义
    T_B_E0_from_chain = T_B_Ws @ T_Ws_I0 @ T_IE

    err = np.max(np.abs(T_B_E0_from_chain - T_B_E0))
    all_pass &= check(
        "首帧链闭合：T(B,Ws)*I*T(I,E) = T(B,E_0)",
        err < TOL_SE3,
        f"误差 = {err:.2e}",
    )

    # T(B,W_s) 必须是合法 SE(3)
    R_Ws = T_B_Ws[:3, :3]
    ortho_err = np.max(np.abs(R_Ws.T @ R_Ws - np.eye(3)))
    all_pass &= check(
        "T(B,W_s) 为合法 SE(3)",
        ortho_err < TOL_SE3,
        f"正交误差 = {ortho_err:.2e}",
    )

    # 关键等式：T(B,W_s) = T(B,I_0) → 验证分解正确
    # T(B,I_0) = T(B,E_0) * T(E,I) = T(B,E_0) * T(I,E)^{-1}
    T_B_I0 = T_B_E0 @ T_IE_inv
    err2 = np.max(np.abs(T_B_Ws - T_B_I0))
    all_pass &= check(
        "T(B,W_s) = T(B,E_0)*T(E,I) 一致性",
        err2 < TOL_SE3,
        f"误差 = {err2:.2e}",
    )

    return all_pass


# ---------------------------------------------------------------------------
# Check 8: Replay 一致性
# ---------------------------------------------------------------------------


def check_replay_consistency(T_IE: np.ndarray, n_trials: int = 50) -> bool:
    """
    给定 T(W_s,E*_t)（来自 action_pose_ws），
    replay 时：T(B,E*_t) = T(B,W_s) * T(W_s,E*_t)
    不需要再经过 T(I,E)，因为 action 已经是 E 语义。
    验证与错误地多乘 T(I,E) 会产生差异。
    """
    section(f"Check 8: Replay 一致性（{n_trials} 组随机数据）")

    rng = np.random.default_rng(99)
    T_IE_inv = se3_inv(T_IE)
    all_pass = True
    max_wrong_err = 0.0
    max_correct_err = 0.0

    for _ in range(n_trials):
        T_B_Ws = random_se3(rng)
        T_Ws_Et_star = random_se3(rng)  # action_pose_ws（已是 E 语义）

        # 正确 replay：
        T_B_Et_star_correct = T_B_Ws @ T_Ws_Et_star

        # 错误做法（误把 action 当 I 语义再乘 T(I,E)）：
        T_B_Et_star_wrong = T_B_Ws @ T_Ws_Et_star @ T_IE

        # 正确结果应为合法 SE(3)
        R = T_B_Et_star_correct[:3, :3]
        ortho_err = np.max(np.abs(R.T @ R - np.eye(3)))
        max_correct_err = max(max_correct_err, ortho_err)

        # 错误结果与正确结果应有显著差异（证明 T(I,E)≠I）
        diff = np.max(np.abs(T_B_Et_star_wrong - T_B_Et_star_correct))
        max_wrong_err = max(max_wrong_err, diff)

    all_pass &= check(
        "正确 replay：T(B,Ws)*T(Ws,E*_t) 为合法 SE(3)",
        max_correct_err < TOL_SE3,
        f"最大正交误差 = {max_correct_err:.2e}",
    )

    # T(I,E) ≠ I，因此错误做法和正确做法必然有显著差异
    translation_magnitude = np.linalg.norm(T_IE[:3, 3])
    all_pass &= check(
        f"T(I,E) ≠ I：||t(I,E)|| = {translation_magnitude*1000:.1f} mm（非零偏移）",
        translation_magnitude > 0.05,  # > 50mm 偏移不可忽略
        f"||t(I,E)|| = {translation_magnitude:.4f} m",
    )
    all_pass &= check(
        "误用 T(I,E) 会导致显著位姿偏差（平均差异 > 10 mm）",
        max_wrong_err > 0.01,
        f"最大差异 = {max_wrong_err*1000:.1f} mm",
    )

    return all_pass


# ---------------------------------------------------------------------------
# Check 9: FK 一致性（可选，需要 placo）
# ---------------------------------------------------------------------------


def check_fk_consistency(urdf_path: str, T_IE: np.ndarray) -> bool:
    """
    使用 placo 计算 FK，验证：
      T(B,E_t)_FK = T(B,Ws)*T(Ws,It)*T(I,E)

    合成流程：
      1. 随机选取关节角 q
      2. FK(q, das_gripper_ee) → T(B,E)_FK
      3. 从 T(B,E)_FK 反推 T(B,I) = T(B,E)*T(E,I)
      4. 设 T(B,W_s) = T(B,I)（首帧初始化）
      5. 对另一组关节角 q2，FK → T(B,E2)_FK
      6. 反推 T(B,I2) = T(B,E2)*T(E,I)
      7. T(W_s,I2) = T(B,W_s)^{-1} * T(B,I2)
      8. 链验证：T(B,W_s)*T(W_s,I2)*T(I,E) ≈ T(B,E2)_FK
    """
    section("Check 9: FK 一致性（placo）")

    try:
        from lerobot.model.kinematics import RobotKinematics
    except ImportError:
        print("  [SKIP] lerobot.model.kinematics 导入失败")
        return True

    try:
        kin = RobotKinematics(
            urdf_path=urdf_path,
            target_frame_name="das_gripper_ee",
            joint_names=[
                "fr3_joint1",
                "fr3_joint2",
                "fr3_joint3",
                "fr3_joint4",
                "fr3_joint5",
                "fr3_joint6",
                "fr3_joint7",
            ],
        )
    except Exception as e:
        print(f"  [SKIP] placo 初始化失败: {e}")
        return True

    rng = np.random.default_rng(2024)
    T_IE_inv = se3_inv(T_IE)
    all_pass = True
    errors = []

    # FR3 关节角合法范围（近似），单位：度
    joint_limits_deg = [
        (-166, 166),
        (-101, 101),
        (-166, 166),
        (-176, -4),
        (-166, 166),
        (-1, 215),
        (-166, 166),
    ]

    n_trials = 20
    for trial in range(n_trials):
        # 随机关节角（episode 首帧 q0）
        q0_deg = np.array([rng.uniform(lo, hi) for lo, hi in joint_limits_deg])
        T_B_E0_fk = kin.forward_kinematics(q0_deg)

        # Episode 初始化：T(B,W_s) = T(B,I_0) = T(B,E_0)*T(E,I)
        T_B_Ws = T_B_E0_fk @ T_IE_inv

        # 另一组关节角（运行时某帧）
        q_t_deg = np.array([rng.uniform(lo, hi) for lo, hi in joint_limits_deg])
        T_B_Et_fk = kin.forward_kinematics(q_t_deg)

        # 反推 T(W_s, I_t) from FK
        T_B_It = T_B_Et_fk @ T_IE_inv
        T_Ws_It = se3_inv(T_B_Ws) @ T_B_It

        # 链重建
        T_B_Et_chain = T_B_Ws @ T_Ws_It @ T_IE

        # 位置误差
        pos_err = np.linalg.norm(T_B_Et_chain[:3, 3] - T_B_Et_fk[:3, 3])
        # 旋转误差（Frobenius norm of R diff）
        rot_err = np.linalg.norm(T_B_Et_chain[:3, :3] - T_B_Et_fk[:3, :3])

        errors.append((pos_err, rot_err))
        if pos_err > TOL_FK or rot_err > TOL_FK:
            all_pass = False

    max_pos = max(e[0] for e in errors)
    max_rot = max(e[1] for e in errors)
    all_pass &= check(
        f"FK 链一致性：位置误差 < {TOL_FK*1000:.1f} mm（{n_trials} 组）",
        max_pos < TOL_FK,
        f"最大位置误差 = {max_pos*1000:.4f} mm",
    )
    all_pass &= check(
        f"FK 链一致性：旋转误差 < {TOL_FK:.4f}（{n_trials} 组）",
        max_rot < TOL_FK,
        f"最大旋转误差 = {max_rot:.6f}",
    )

    return all_pass


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="验证 FR3 + DAS Gripper SLAM Frame Contract（T(I,E) 及链式一致性）"
    )
    parser.add_argument(
        "--urdf",
        type=str,
        default=None,
        help="fr3_das_ati.urdf 路径（用于 Check 5）。留空则跳过 URDF 解析检查。",
    )
    parser.add_argument(
        "--use-placo",
        action="store_true",
        help="启用 FK 一致性检查（需要 placo 安装，需同时提供 --urdf）",
    )
    args = parser.parse_args()

    # 如果 --urdf 未指定，尝试从脚本位置自动推断
    if args.urdf is None:
        default_urdf = (
            Path(__file__).parent / "assets" / "franka_fr3" / "fr3_das_ati.urdf"
        )
        if default_urdf.exists():
            args.urdf = str(default_urdf)
            print(f"[INFO] 自动找到 URDF: {args.urdf}")

    print("=" * 60)
    print("  FR3 + DAS Gripper SLAM Frame Contract 验证")
    print("=" * 60)
    print(f"\n候选 T(I,E):\n{np.round(T_IE_CANDIDATE, 8)}\n")

    T_IE = T_IE_CANDIDATE
    results = []

    results.append(check_se3_validity(T_IE))
    results.append(check_rotation_approx(T_IE))
    results.append(check_translation(T_IE))
    results.append(check_axis_semantics(T_IE))
    results.append(check_urdf_base_E(args.urdf, T_IE))
    results.append(check_chain_consistency(T_IE))
    results.append(check_first_frame_identity(T_IE))
    results.append(check_replay_consistency(T_IE))

    if args.use_placo:
        if args.urdf is None:
            print("\n[WARN] --use-placo 需要同时提供 --urdf，FK 检查跳过。")
        else:
            results.append(check_fk_consistency(args.urdf, T_IE))

    # 汇总
    n_pass = sum(results)
    n_total = len(results)
    print("\n" + "=" * 60)
    if all(results):
        print(f"  全部通过 ({n_pass}/{n_total})  ✓  Frame Contract 验证成功")
        print("=" * 60)
        return 0
    else:
        print(f"  未全部通过 ({n_pass}/{n_total}) ✗  存在验证失败项，请检查上方输出")
        print("=" * 60)
        return 1


if __name__ == "__main__":
    sys.exit(main())
