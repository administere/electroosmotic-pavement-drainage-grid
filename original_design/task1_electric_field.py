#!/usr/bin/env python3
"""
任务一：二维电场分布与电流密度验证
=============================================
方法：有限差分法（FDM）求解稳态电流场 ∇·(σ∇V) = 0
计算域：横向一个周期 2d=0.8m，深度 0.2m（水稳层）
边界条件：阳极 +48V，阴极 0V，其余边界绝缘
求解器：逐次超松弛（SOR）迭代法

作者：计算物理工程师
日期：2026-07-22
"""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
from matplotlib import cm
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 1. 设计参数定义
# ============================================================
V_anode = 48.0        # 阳极电压 [V]
V_cathode = 0.0       # 阴极电压 [V]
d = 0.4                # 电极间距（阳-阴中心距）[m]
Lx = 2 * d             # 计算域宽度（一个周期）[m] = 0.8 m
Lz = 0.2               # 计算域深度 [m]
sigma_med = 0.05       # 水稳层电导率 [S/m]
w_electrode = 0.02     # 电极宽度 [m]（碳纤维束宽度）
J_target = 0.3         # 目标面电流密度 [A/m²]

# 电极位置（沿顶部 z=0）：
# 左阳极：x ∈ [0, w_electrode]，中心在 x=0
# 阴极：  x ∈ [d-w_electrode, d+w_electrode]，中心在 x=d=0.4m
# 右阳极：x ∈ [Lx-w_electrode, Lx]，中心在 x=Lx=0.8m

# ============================================================
# 2. 有限差分网格设置
# ============================================================
Nx = 201               # x方向网格点数（奇数以便电极居中）
Nz = 51                # z方向网格点数
dx = Lx / (Nx - 1)     # x方向步长 [m]
dz = Lz / (Nz - 1)     # z方向步长 [m]

x = np.linspace(0, Lx, Nx)
z = np.linspace(0, Lz, Nz)
X, Z = np.meshgrid(x, z, indexing='ij')  # X[i,j], Z[i,j]

print(f"网格: {Nx}×{Nz}, dx={dx:.4f}m, dz={dz:.4f}m")

# ============================================================
# 3. 边界条件标记
# ============================================================
# is_dirichlet[i,j] = True 表示该点电压固定
# V_init[i,j] 存储初始/边界值
is_dirichlet = np.zeros((Nx, Nz), dtype=bool)
V_init = np.zeros((Nx, Nz))

# 顶边界 (j=0)：电极位置为 Dirichlet，间隙为 Neumann
anode_left_idx = int(w_electrode / dx)       # 左阳极右边缘索引
cathode_left_idx = int((d - w_electrode) / dx)   # 阴极左边缘索引
cathode_right_idx = int((d + w_electrode) / dx)  # 阴极右边缘索引
anode_right_idx = int((Lx - w_electrode) / dx)   # 右阳极左边缘索引

for i in range(Nx):
    if i <= anode_left_idx:
        # 左阳极
        is_dirichlet[i, 0] = True
        V_init[i, 0] = V_anode
    elif cathode_left_idx <= i <= cathode_right_idx:
        # 中间阴极
        is_dirichlet[i, 0] = True
        V_init[i, 0] = V_cathode
    elif i >= anode_right_idx:
        # 右阳极
        is_dirichlet[i, 0] = True
        V_init[i, 0] = V_anode
    # 其余顶边界点为 Neumann (∂V/∂z=0)，在求解器中处理

print(f"左阳极: x ∈ [0, {anode_left_idx*dx:.3f}] m, 点数 {anode_left_idx+1}")
print(f"阴极:   x ∈ [{cathode_left_idx*dx:.3f}, {cathode_right_idx*dx:.3f}] m, "
      f"点数 {cathode_right_idx-cathode_left_idx+1}")
print(f"右阳极: x ∈ [{anode_right_idx*dx:.3f}, {Lx:.3f}] m, 点数 {Nx-anode_right_idx}")
print(f"顶部间隙(Neumann)点数: {Nx - (anode_left_idx+1) - (cathode_right_idx-cathode_left_idx+1) - (Nx-anode_right_idx)}")

# ============================================================
# 4. SOR 迭代求解器
# ============================================================
def solve_laplace_sor(V, is_dirichlet, dx, dz, omega=1.85, tol=1e-8, max_iter=50000):
    """
    用逐次超松弛法(SOR)求解 ∇²V = 0

    参数:
        V: 初始电压数组（含 Dirichlet 边界值）
        is_dirichlet: Dirichlet 边界标记
        dx, dz: 网格步长
        omega: 松弛因子（1<ω<2 为超松弛）
        tol: 收敛容差
        max_iter: 最大迭代次数

    返回:
        V: 求解后的电压分布
        residuals: 每次迭代的最大残差列表
    """
    Nx, Nz = V.shape
    alpha = (dx / dz) ** 2  # 各向异性系数
    residuals = []

    for iteration in range(max_iter):
        V_old = V.copy()
        max_residual = 0.0

        for i in range(Nx):
            for j in range(Nz):
                if is_dirichlet[i, j]:
                    continue

                # 获取邻居电压（处理边界）
                # 左邻居
                if i == 0:
                    V_left = V[1, j]  # Neumann: ∂V/∂x=0 → V[-1]=V[1]
                else:
                    V_left = V[i-1, j]

                # 右邻居
                if i == Nx - 1:
                    V_right = V[Nx-2, j]  # Neumann
                else:
                    V_right = V[i+1, j]

                # 下邻居
                if j == Nz - 1:
                    V_down = V[i, Nz-2]  # Neumann
                else:
                    V_down = V[i, j+1]

                # 上邻居
                if j == 0:
                    if is_dirichlet[i, 0]:
                        V_up = V[i, 0]  # 不应到达此处
                    else:
                        V_up = V[i, 1]  # Neumann: ∂V/∂z=0 → ghost = interior
                else:
                    V_up = V[i, j-1]

                # SOR 更新公式
                # 标准5点差分: V_new = (V_left+V_right+α*(V_up+V_down))/(2+2α)
                V_gs = (V_left + V_right + alpha * (V_up + V_down)) / (2 + 2*alpha)
                V_new = (1 - omega) * V[i, j] + omega * V_gs

                residual = abs(V_new - V[i, j])
                if residual > max_residual:
                    max_residual = residual

                V[i, j] = V_new

        residuals.append(max_residual)

        if iteration % 2000 == 0:
            print(f"  迭代 {iteration:5d}, 最大残差 = {max_residual:.2e}")

        if max_residual < tol:
            print(f"  收敛于迭代 {iteration+1}, 残差 = {max_residual:.2e}")
            break
    else:
        print(f"  达到最大迭代次数 {max_iter}, 最终残差 = {max_residual:.2e}")

    return V, residuals

# 初始化
V = V_init.copy()
print("\n开始 SOR 迭代求解...")
V, residuals = solve_laplace_sor(V, is_dirichlet, dx, dz, omega=1.85, tol=1e-8)

# ============================================================
# 5. 计算电场强度与电流密度
# ============================================================
# E = -∇V, 使用中心差分（内部）和单侧差分（边界）
Ex = np.zeros((Nx, Nz))
Ez = np.zeros((Nx, Nz))

for i in range(Nx):
    for j in range(Nz):
        # Ex = -∂V/∂x
        if i == 0:
            Ex[i, j] = -(V[1, j] - V[0, j]) / dx  # 前向差分
        elif i == Nx - 1:
            Ex[i, j] = -(V[Nx-1, j] - V[Nx-2, j]) / dx  # 后向差分
        else:
            Ex[i, j] = -(V[i+1, j] - V[i-1, j]) / (2 * dx)  # 中心差分

        # Ez = -∂V/∂z
        if j == 0:
            Ez[i, j] = -(V[i, 1] - V[i, 0]) / dz  # 前向差分
        elif j == Nz - 1:
            Ez[i, j] = -(V[i, Nz-1] - V[i, Nz-2]) / dz  # 后向差分
        else:
            Ez[i, j] = -(V[i, j+1] - V[i, j-1]) / (2 * dz)  # 中心差分

# 电场强度幅值
E_mag = np.sqrt(Ex**2 + Ez**2)

# 电流密度 (Ohm's law: J = σE)
Jx = sigma_med * Ex
Jz = sigma_med * Ez
J_mag = sigma_med * E_mag

# ============================================================
# 6. 关键指标计算
# ============================================================
# 6.1 电极间平均电场强度
# 沿顶部电极连线 (z≈0) 的平均水平电场
top_mask = np.zeros(Nx, dtype=bool)
for i in range(Nx):
    if not is_dirichlet[i, 0]:
        top_mask[i] = True

# 电极间间隙区域的电场
gap_mask = np.zeros((Nx, Nz), dtype=bool)
for i in range(Nx):
    for j in range(Nz):
        if not is_dirichlet[i, j]:
            gap_mask[i, j] = True

# 整个域的平均 |E|
E_avg_domain = np.mean(E_mag[gap_mask])
E_avg_top = np.mean(E_mag[top_mask, 0]) if np.any(top_mask) else V_anode / d

# 沿 x 方向（电极连线方向）的电场均匀性
# 取 z = dz/2 (靠近顶面的第一层内部点)沿 x 的 E 分布
j_near_top = 1  # 靠近顶面的内部点
E_x_profile = E_mag[:, j_near_top]
E_x_top = np.sqrt(Ex[:, 0]**2 + Ez[:, 0]**2)

# 6.2 平均电流密度
J_avg_domain = sigma_med * E_avg_domain

# 6.3 电极间电场均匀性评价
# 计算电极间区域（非电极覆盖区）E场的变异系数
gap_x_mask = np.ones(Nx, dtype=bool)
for i in range(Nx):
    if is_dirichlet[i, 0]:
        gap_x_mask[i] = False
E_gap_values = E_mag[gap_x_mask, 0]
E_gap_mean = np.mean(E_gap_values) if len(E_gap_values) > 0 else 0
E_gap_std = np.std(E_gap_values) if len(E_gap_values) > 0 else 0
E_gap_cv = E_gap_std / E_gap_mean if E_gap_mean > 0 else 0  # 变异系数

# 6.4 计算流过电极的总电流（验证电流密度）
# 阳极注入电流 = ∫ J·n dS，在2D中为线积分
# 左阳极：通过其底面的电流（z=0处向下）
I_anode_left = 0.0
for i in range(anode_left_idx + 1):
    I_anode_left += Jz[i, 0] * dx  # Jz向下为正（电流注入介质）
I_anode_left = abs(I_anode_left)  # 单位宽度电流 [A/m]

# 右阳极：
I_anode_right = 0.0
for i in range(anode_right_idx, Nx):
    I_anode_right += Jz[i, 0] * dx
I_anode_right = abs(I_anode_right)

# 阴极收集电流
I_cathode = 0.0
for i in range(cathode_left_idx, cathode_right_idx + 1):
    I_cathode += Jz[i, 0] * dx  # 流入阴极的电流
I_cathode = abs(I_cathode)

I_total_per_meter = (I_anode_left + I_anode_right)  # 每米长度总电流 [A/m]
# 面电流密度（考虑周期2d=0.8m内有两对阳-阴极）
J_areal = I_total_per_meter / Lx  # [A/m²] 每平米格栅电流

print(f"\n{'='*50}")
print(f"电场与电流密度计算结果")
print(f"{'='*50}")
print(f"域内平均 |E| = {E_avg_domain:.2f} V/m")
print(f"顶面(间隙)平均 |E| = {E_avg_top:.2f} V/m")
print(f"V/d 理论值 = {V_anode/d:.2f} V/m")
print(f"电极间隙 E 场变异系数 = {E_gap_cv:.3f}")
print(f"域内平均 |J| = {J_avg_domain:.3f} A/m²")
print(f"计算面电流密度 = {J_areal:.3f} A/m²")
print(f"目标面电流密度 = {J_target:.3f} A/m²")
print(f"偏差 = {(J_areal - J_target)/J_target*100:.1f}%")

# ============================================================
# 7. 可视化 (4个子图)
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('任务一：二维电场分布与电流密度验证', fontsize=14, fontweight='bold')

# --- 子图1：电势分布等值线图 ---
ax1 = axes[0, 0]
# 填充等值线
levels = 25
cf1 = ax1.contourf(X, Z, V, levels=levels, cmap='RdYlBu_r', extend='both')
# 等值线
ct1 = ax1.contour(X, Z, V, levels=levels, colors='k', linewidths=0.3, alpha=0.5)
ax1.clabel(ct1, inline=True, fontsize=6, fmt='%.0f')
# 标记电极位置
# 左阳极
ax1.add_patch(plt.Rectangle((0, -0.003), w_electrode, 0.006,
                             facecolor='red', edgecolor='darkred', alpha=0.8))
ax1.text(w_electrode/2, -0.012, 'Anode\n+48V', ha='center', fontsize=8, color='red', fontweight='bold')
# 阴极
ax1.add_patch(plt.Rectangle((d-w_electrode, -0.003), 2*w_electrode, 0.006,
                             facecolor='blue', edgecolor='darkblue', alpha=0.8))
ax1.text(d, -0.012, 'Cathode\n0V', ha='center', fontsize=8, color='blue', fontweight='bold')
# 右阳极
ax1.add_patch(plt.Rectangle((Lx-w_electrode, -0.003), w_electrode, 0.006,
                             facecolor='red', edgecolor='darkred', alpha=0.8))
ax1.text(Lx-w_electrode/2, -0.012, 'Anode\n+48V', ha='center', fontsize=8, color='red', fontweight='bold')

ax1.set_xlabel('横向位置 x [m]', fontsize=10)
ax1.set_ylabel('深度 z [m]', fontsize=10)
ax1.set_title('电势分布 V(x,z) [V]', fontsize=11)
ax1.invert_yaxis()  # z向下为正
cbar1 = fig.colorbar(cf1, ax=ax1, shrink=0.85, pad=0.02)
cbar1.set_label('电势 [V]', fontsize=9)
ax1.set_aspect('equal')

# --- 子图2：电场强度云图 + 矢量图 ---
ax2 = axes[0, 1]
# 电场强度云图
cf2 = ax2.contourf(X, Z, E_mag, levels=30, cmap='plasma')
# 电场矢量（降采样以提高可读性）
skip_x = 6
skip_z = 2
X_sub = X[::skip_x, ::skip_z]
Z_sub = Z[::skip_x, ::skip_z]
Ex_sub = Ex[::skip_x, ::skip_z]
Ez_sub = Ez[::skip_x, ::skip_z]
E_mag_sub = E_mag[::skip_x, ::skip_z]
# 归一化矢量方向，长度表示幅值
ax2.quiver(X_sub, Z_sub, Ex_sub, Ez_sub, E_mag_sub,
           cmap='Greys', alpha=0.7, scale=800, width=0.003)

# 标记电极
ax2.add_patch(plt.Rectangle((0, -0.003), w_electrode, 0.006, facecolor='red', alpha=0.8))
ax2.add_patch(plt.Rectangle((d-w_electrode, -0.003), 2*w_electrode, 0.006, facecolor='blue', alpha=0.8))
ax2.add_patch(plt.Rectangle((Lx-w_electrode, -0.003), w_electrode, 0.006, facecolor='red', alpha=0.8))

ax2.set_xlabel('横向位置 x [m]', fontsize=10)
ax2.set_ylabel('深度 z [m]', fontsize=10)
ax2.set_title('电场强度 |E| [V/m] 及矢量图', fontsize=11)
ax2.invert_yaxis()
cbar2 = fig.colorbar(cf2, ax=ax2, shrink=0.85, pad=0.02)
cbar2.set_label('|E| [V/m]', fontsize=9)
ax2.set_aspect('equal')

# --- 子图3：电流密度分布 ---
ax3 = axes[1, 0]
cf3 = ax3.contourf(X, Z, J_mag, levels=30, cmap='YlOrRd')
# 电流密度矢量（降采样）
Jx_sub = Jx[::skip_x, ::skip_z]
Jz_sub = Jz[::skip_x, ::skip_z]
J_mag_sub = J_mag[::skip_x, ::skip_z]
ax3.quiver(X_sub, Z_sub, Jx_sub, Jz_sub, J_mag_sub,
           cmap='Greys', alpha=0.7, scale=0.5, width=0.003)

ax3.add_patch(plt.Rectangle((0, -0.003), w_electrode, 0.006, facecolor='red', alpha=0.8))
ax3.add_patch(plt.Rectangle((d-w_electrode, -0.003), 2*w_electrode, 0.006, facecolor='blue', alpha=0.8))
ax3.add_patch(plt.Rectangle((Lx-w_electrode, -0.003), w_electrode, 0.006, facecolor='red', alpha=0.8))

ax3.set_xlabel('横向位置 x [m]', fontsize=10)
ax3.set_ylabel('深度 z [m]', fontsize=10)
ax3.set_title('电流密度 |J| [A/m²] 及流向', fontsize=11)
ax3.invert_yaxis()
cbar3 = fig.colorbar(cf3, ax=ax3, shrink=0.85, pad=0.02)
cbar3.set_label('|J| [A/m²]', fontsize=9)
ax3.set_aspect('equal')

# --- 子图4：顶部沿x方向的电场和电流分布 ---
ax4 = axes[1, 1]
# 沿顶部 (z=0) 的 |E| 分布
ax4.plot(x, E_x_top, 'b-', linewidth=1.5, label='|E| at z=0')
ax4.plot(x, J_mag[:, 0], 'r-', linewidth=1.5, label='|J| at z=0')

# 标记电极区域
ax4.axvspan(0, w_electrode, alpha=0.15, color='red', label='Anode region')
ax4.axvspan(d-w_electrode, d+w_electrode, alpha=0.15, color='blue', label='Cathode region')
ax4.axvspan(Lx-w_electrode, Lx, alpha=0.15, color='red')

ax4.axhline(y=E_avg_top, color='b', linestyle='--', alpha=0.5, label=f'Mean |E|={E_avg_top:.1f} V/m')
ax4.axhline(y=J_target, color='g', linestyle=':', linewidth=1.5,
            label=f'Target J={J_target} A/m²')

ax4.set_xlabel('横向位置 x [m]', fontsize=10)
ax4.set_ylabel('幅值', fontsize=10)
ax4.set_title('顶面(z=0)沿x方向电场与电流密度', fontsize=11)
ax4.legend(loc='upper right', fontsize=7)
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('task1_electric_field.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"\n图片已保存: task1_electric_field.png")

# --- 单独电势等值线图（用于报告） ---
fig2, ax = plt.subplots(figsize=(10, 6))
cf = ax.contourf(X, Z, V, levels=30, cmap='RdYlBu_r', extend='both')
ct = ax.contour(X, Z, V, levels=15, colors='k', linewidths=0.4)
ax.clabel(ct, inline=True, fontsize=7, fmt='%.0f')
ax.add_patch(plt.Rectangle((0, -0.004), w_electrode, 0.008, facecolor='red', edgecolor='darkred'))
ax.add_patch(plt.Rectangle((d-w_electrode, -0.004), 2*w_electrode, 0.008, facecolor='blue', edgecolor='darkblue'))
ax.add_patch(plt.Rectangle((Lx-w_electrode, -0.004), w_electrode, 0.008, facecolor='red', edgecolor='darkred'))
ax.text(w_electrode/2, -0.014, 'Anode +48V', ha='center', fontsize=9, color='darkred', fontweight='bold')
ax.text(d, -0.014, 'Cathode 0V', ha='center', fontsize=9, color='darkblue', fontweight='bold')
ax.text(Lx-w_electrode/2, -0.014, 'Anode +48V', ha='center', fontsize=9, color='darkred', fontweight='bold')
ax.set_xlabel('Transverse position x [m]', fontsize=11)
ax.set_ylabel('Depth z [m]', fontsize=11)
ax.set_title('Electric Potential Distribution V(x,z)', fontsize=12)
ax.invert_yaxis()
cbar = fig.colorbar(cf, ax=ax)
cbar.set_label('Potential [V]', fontsize=10)
ax.set_aspect('equal')
plt.tight_layout()
plt.savefig('task1_potential.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"图片已保存: task1_potential.png")

# --- 单独电场矢量图 ---
fig3, ax = plt.subplots(figsize=(10, 6))
cf3 = ax.contourf(X, Z, E_mag, levels=30, cmap='plasma')
skip = 4
ax.quiver(X[::skip, ::skip], Z[::skip, ::skip],
          Ex[::skip, ::skip], Ez[::skip, ::skip],
          E_mag[::skip, ::skip], cmap='Greys', alpha=0.7, scale=1200, width=0.004)
ax.add_patch(plt.Rectangle((0, -0.004), w_electrode, 0.008, facecolor='red', edgecolor='darkred'))
ax.add_patch(plt.Rectangle((d-w_electrode, -0.004), 2*w_electrode, 0.008, facecolor='blue', edgecolor='darkblue'))
ax.add_patch(plt.Rectangle((Lx-w_electrode, -0.004), w_electrode, 0.008, facecolor='red', edgecolor='darkred'))
ax.set_xlabel('Transverse position x [m]', fontsize=11)
ax.set_ylabel('Depth z [m]', fontsize=11)
ax.set_title('Electric Field Magnitude |E| and Vectors', fontsize=12)
ax.invert_yaxis()
cbar = fig.colorbar(cf3, ax=ax)
cbar.set_label('|E| [V/m]', fontsize=10)
ax.set_aspect('equal')
plt.tight_layout()
plt.savefig('task1_efield.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"图片已保存: task1_efield.png")

# ============================================================
# 8. 输出汇总
# ============================================================
print(f"\n{'='*60}")
print(f"任务一 汇总结果")
print(f"{'='*60}")
print(f"  电极间电场强度（域平均）:   {E_avg_domain:.2f} V/m")
print(f"  电极间电场强度（顶面平均）: {E_avg_top:.2f} V/m")
print(f"  V/d 理论近似值:              {V_anode/d:.2f} V/m")
print(f"  电场均匀性（变异系数CV）:    {E_gap_cv:.3f} "
      f"({'良好' if E_gap_cv < 0.3 else '较差'})")
print(f"  计算面电流密度:              {J_areal:.3f} A/m²")
print(f"  目标面电流密度:              {J_target:.3f} A/m²")
print(f"  偏差:                        {(J_areal-J_target)/J_target*100:.1f}%")
print(f"  域内平均 |J|:                {J_avg_domain:.3f} A/m²")
print(f"  单位宽度总电流:              {I_total_per_meter:.3f} A/m")
print(f"  结论: 实际电流密度 "
      f"{'接近' if abs(J_areal-J_target)/J_target < 0.5 else '显著偏离'}目标值")

# 保存关键数据供其他任务使用
np.savez('task1_results.npz',
         E_avg_domain=E_avg_domain,
         E_avg_top=E_avg_top,
         J_areal=J_areal,
         J_avg_domain=J_avg_domain,
         V_d_ratio=V_anode/d,
         I_total_per_meter=I_total_per_meter)

print(f"\n关键数据已保存至: task1_results.npz")
print(f"任务一脚本执行完毕。")
