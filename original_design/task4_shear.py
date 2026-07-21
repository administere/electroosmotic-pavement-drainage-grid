#!/usr/bin/env python3
"""
任务四：层间抗剪安全系数验算
=============================================
方法：弹性半空间理论（Boussinesq解）计算圆形均布荷载
      在格栅层位产生的应力，验算层间抗剪安全系数。

理论背景：
  - 双轮组轴载 P=100kN，简化为单轮当量圆均布荷载
  - 轮压 p=0.7MPa，当量圆半径 δ=0.15m
  - 格栅埋深 z=0.12m
  - 基于弹性半空间理论计算 σ_z, σ_r, σ_θ
  - 最大剪应力 τ_max = max(|σ_i - σ_j|)/2
  - 抗剪强度 τ_allow = c + σ_n·tan(φ)
  - 安全系数 FS = τ_allow / τ_max

参考文献：
  - Boussinesq (1885), Application des potentiels...
  - Foster & Ahlvin (1954), Stresses in pavements
  - Yoder & Witczak (1975), Principles of Pavement Design

作者：计算物理工程师
日期：2026-07-22
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 1. 设计参数定义
# ============================================================
P_axle = 100.0            # 双轮组轴载 [kN]
p_tire = 0.7              # 轮胎接地压力 [MPa]
delta = 0.15              # 单轮当量圆半径 [m]
z_grid = 0.12             # 格栅埋深 [m]（= h_asphalt）
c_cohesion = 40.0         # 界面粘聚力 [kPa]
phi_friction = 25.0       # 内摩擦角 [°]
phi_rad = np.radians(phi_friction)
FS_required = 1.5         # 要求安全系数

# 沥青泊松比（典型值，温度较高时偏大）
nu_asphalt = 0.35

print(f"{'='*50}")
print(f"层间抗剪安全系数验算")
print(f"{'='*50}")
print(f"  轴载 P = {P_axle} kN")
print(f"  轮压 p = {p_tire} MPa")
print(f"  当量圆半径 δ = {delta} m")
print(f"  格栅埋深 z = {z_grid} m")
print(f"  界面粘聚力 c = {c_cohesion} kPa")
print(f"  内摩擦角 φ = {phi_friction}°")

# ============================================================
# 2. 弹性半空间应力计算
# ============================================================
# 圆形均布荷载下弹性半空间的应力解（Foster & Ahlvin, 1954）
# 在荷载中心线下方(r=0)的应力：

# 无量纲深度
a = z_grid / delta  # = 0.12/0.15 = 0.8

# 竖向应力 σ_z（荷载中心线下方）
# σ_z/p = 1 - (a³) / (1 + a²)^(3/2)
sigma_z_ratio = 1.0 - a**3 / (1.0 + a**2)**1.5
sigma_z = sigma_z_ratio * p_tire  # [MPa]

# 径向应力 σ_r = σ_θ（荷载中心线下方，轴对称）
# σ_r/p = (1+2ν)/2 - (1+ν)*a/√(1+a²) + a³/(2*(1+a²)^(3/2))
# 或使用以下标准公式：
# σ_r/p = 0.5*[(1+2ν) - 2(1+ν)*a/√(1+a²) + a³/(1+a²)^(3/2)]
term1 = (1 + 2*nu_asphalt) / 2.0
term2 = (1 + nu_asphalt) * a / np.sqrt(1 + a**2)
term3 = a**3 / (2 * (1 + a**2)**1.5)
sigma_r_ratio = term1 - term2 + term3
sigma_r = sigma_r_ratio * p_tire  # [MPa]

# 环向应力 σ_θ = σ_r（轴对称）
sigma_theta = sigma_r

print(f"\n  荷载中心线下方 z={z_grid}m 处应力:")
print(f"    归一化深度 a = z/δ = {a:.3f}")
print(f"    竖向应力 σ_z = {sigma_z:.4f} MPa ({sigma_z_ratio*100:.1f}% of p)")
print(f"    径向应力 σ_r = {sigma_r:.4f} MPa")
print(f"    环向应力 σ_θ = {sigma_theta:.4f} MPa")

# ============================================================
# 3. 最大剪应力计算
# ============================================================
# 在轴对称条件下 (σ_r = σ_θ)，主应力为 σ_z, σ_r, σ_θ
# 最大剪应力为三个主应力差的一半中的最大值

# 三个主应力
sigma_1 = max(sigma_z, sigma_r, sigma_theta)
sigma_3 = min(sigma_z, sigma_r, sigma_theta)
sigma_2 = sigma_z + sigma_r + sigma_theta - sigma_1 - sigma_3

# 剪应力
tau_13 = abs(sigma_1 - sigma_3) / 2.0
tau_12 = abs(sigma_1 - sigma_2) / 2.0
tau_23 = abs(sigma_2 - sigma_3) / 2.0

tau_max = max(tau_13, tau_12, tau_23)
tau_max_kPa = tau_max * 1000  # 转换为 kPa

print(f"\n  主应力:")
print(f"    σ₁ = {sigma_1:.4f} MPa")
print(f"    σ₂ = {sigma_2:.4f} MPa")
print(f"    σ₃ = {sigma_3:.4f} MPa")
print(f"  剪应力:")
print(f"    τ₁₃ = {tau_13*1000:.1f} kPa")
print(f"    τ₁₂ = {tau_12*1000:.1f} kPa")
print(f"    τ₂₃ = {tau_23*1000:.1f} kPa")
print(f"    τ_max = {tau_max_kPa:.1f} kPa")

# 另外，计算八面体剪应力作为参考
tau_oct = np.sqrt((sigma_z - sigma_r)**2 + (sigma_r - sigma_theta)**2 +
                   (sigma_theta - sigma_z)**2) / 3.0
print(f"    八面体剪应力 τ_oct = {tau_oct*1000:.1f} kPa")

# ============================================================
# 4. 抗剪强度与安全系数
# ============================================================
# τ_allow = c + σ_n * tan(φ)
# σ_n 取竖向应力 σ_z（层间法向应力）
sigma_n = sigma_z  # [MPa]
sigma_n_kPa = sigma_z * 1000  # [kPa]

tau_allow = c_cohesion + sigma_n_kPa * np.tan(phi_rad)  # [kPa]

FS = tau_allow / tau_max_kPa

print(f"\n  抗剪强度计算:")
print(f"    法向应力 σ_n = {sigma_n_kPa:.1f} kPa")
print(f"    粘聚力分量 c = {c_cohesion} kPa")
print(f"    摩擦分量 σ_n·tan(φ) = {sigma_n_kPa*np.tan(phi_rad):.1f} kPa")
print(f"    抗剪强度 τ_allow = {tau_allow:.1f} kPa")
print(f"    最大剪应力 τ_max = {tau_max_kPa:.1f} kPa")
print(f"    安全系数 FS = τ_allow/τ_max = {FS:.3f}")
print(f"    要求安全系数 FS_req = {FS_required}")
print(f"    判断: {'✓ 通过' if FS >= FS_required else '✗ 不通过'}")

# ============================================================
# 5. 深度方向应力与安全系数分布
# ============================================================
z_range = np.linspace(0.02, 0.4, 200)
FS_vs_z = np.zeros_like(z_range)
sigma_z_vs_z = np.zeros_like(z_range)
tau_max_vs_z = np.zeros_like(z_range)
tau_allow_vs_z = np.zeros_like(z_range)

for i, z_i in enumerate(z_range):
    a_i = z_i / delta

    # σ_z
    sigma_z_i = p_tire * (1.0 - a_i**3 / (1.0 + a_i**2)**1.5)

    # σ_r
    sigma_r_i = p_tire * ((1 + 2*nu_asphalt)/2.0
                          - (1 + nu_asphalt)*a_i/np.sqrt(1 + a_i**2)
                          + a_i**3/(2*(1 + a_i**2)**1.5))

    # 主应力排序
    sigmas = sorted([sigma_z_i, sigma_r_i, sigma_r_i], reverse=True)
    tau_max_i = (sigmas[0] - sigmas[2]) / 2.0

    sigma_n_i = sigma_z_i * 1000  # kPa
    tau_allow_i = c_cohesion + sigma_n_i * np.tan(phi_rad)

    sigma_z_vs_z[i] = sigma_z_i * 1000
    tau_max_vs_z[i] = tau_max_i * 1000
    tau_allow_vs_z[i] = tau_allow_i
    FS_vs_z[i] = tau_allow_i / (tau_max_i * 1000) if tau_max_i > 0 else 99

# ============================================================
# 6. 参数敏感性分析
# ============================================================
# 6.1 泊松比对安全系数的影响
nu_range = np.linspace(0.2, 0.49, 50)
FS_vs_nu = np.zeros_like(nu_range)
for i, nu_i in enumerate(nu_range):
    a_val = z_grid / delta
    sz = p_tire * (1.0 - a_val**3 / (1.0 + a_val**2)**1.5)
    sr = p_tire * ((1+2*nu_i)/2.0 - (1+nu_i)*a_val/np.sqrt(1+a_val**2)
                    + a_val**3/(2*(1+a_val**2)**1.5))
    sigs = sorted([sz, sr, sr], reverse=True)
    tmax = (sigs[0] - sigs[2]) / 2.0
    sn = sz * 1000
    tallow = c_cohesion + sn * np.tan(phi_rad)
    FS_vs_nu[i] = tallow / (tmax * 1000) if tmax > 0 else 99

# 6.2 内摩擦角对安全系数的影响
phi_range = np.linspace(15, 40, 50)
FS_vs_phi = np.zeros_like(phi_range)
for i, phi_i in enumerate(phi_range):
    a_val = z_grid / delta
    sz = p_tire * (1.0 - a_val**3 / (1.0 + a_val**2)**1.5)
    sr = p_tire * ((1+2*nu_asphalt)/2.0 - (1+nu_asphalt)*a_val/np.sqrt(1+a_val**2)
                    + a_val**3/(2*(1+a_val**2)**1.5))
    sigs = sorted([sz, sr, sr], reverse=True)
    tmax = (sigs[0] - sigs[2]) / 2.0
    sn = sz * 1000
    tallow = c_cohesion + sn * np.tan(np.radians(phi_i))
    FS_vs_phi[i] = tallow / (tmax * 1000) if tmax > 0 else 99

# ============================================================
# 7. 改进建议
# ============================================================
print(f"\n{'='*50}")
print(f"参数优化分析")
print(f"{'='*50}")

if FS < FS_required:
    # 需要多大的粘聚力才能达到FS=1.5
    c_required = FS_required * tau_max_kPa - sigma_n_kPa * np.tan(phi_rad)
    print(f"  所需最小粘聚力: c_min = {c_required:.1f} kPa (当前 {c_cohesion} kPa)")
    print(f"  所需粘聚力增量: Δc = {c_required - c_cohesion:.1f} kPa")

    # 需要多大的内摩擦角
    phi_required = np.degrees(np.arctan(
        (FS_required * tau_max_kPa - c_cohesion) / sigma_n_kPa
    ))
    if sigma_n_kPa > 0:
        print(f"  所需最小内摩擦角: φ_min = {phi_required:.1f}° (当前 {phi_friction}°)")
    else:
        print(f"  无法通过增加内摩擦角满足要求（法向应力为零）")

    # 若降低埋深...
    # 寻找临界深度使得FS=1.5
    def fs_at_depth(z_val):
        a_v = z_val / delta
        sz_v = p_tire * (1.0 - a_v**3 / (1.0 + a_v**2)**1.5)
        sr_v = p_tire * ((1+2*nu_asphalt)/2.0
                         - (1+nu_asphalt)*a_v/np.sqrt(1+a_v**2)
                         + a_v**3/(2*(1+a_v**2)**1.5))
        sigs_v = sorted([sz_v, sr_v, sr_v], reverse=True)
        tmax_v = (sigs_v[0] - sigs_v[2]) / 2.0
        sn_v = sz_v * 1000
        tallow_v = c_cohesion + sn_v * np.tan(phi_rad)
        if tmax_v <= 0:
            return 99.0
        return tallow_v / (tmax_v * 1000)

    # 寻找使FS>=1.5的深度范围
    safe_depths = []
    for z_test in z_range:
        if fs_at_depth(z_test) >= FS_required:
            safe_depths.append(z_test)
    if safe_depths:
        print(f"  满足FS≥{FS_required}的深度范围: {min(safe_depths)*100:.1f} ~ "
              f"{max(safe_depths)*100:.1f} cm")
    else:
        print(f"  在当前参数下，任意深度均无法满足FS≥{FS_required}")

# ============================================================
# 8. 可视化
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 11))
fig.suptitle('任务四：层间抗剪安全系数验算', fontsize=14, fontweight='bold')

# --- 子图1：应力沿深度分布 ---
ax1 = axes[0, 0]
ax1.plot(sigma_z_vs_z, z_range * 100, 'b-', linewidth=2, label='Normal stress σ_z')
ax1.plot(tau_max_vs_z, z_range * 100, 'r-', linewidth=2, label='Max shear τ_max')
ax1.plot(tau_allow_vs_z, z_range * 100, 'g-', linewidth=2, label='Shear strength τ_allow')
ax1.axhline(y=z_grid * 100, color='gray', linestyle='--', linewidth=1.5,
            label=f'Grid depth z={z_grid*100:.0f} cm')
ax1.axvline(x=0, color='black', linewidth=0.5)

# 标注当前深度
idx_grid = np.argmin(np.abs(z_range - z_grid))
ax1.plot(sigma_z_vs_z[idx_grid], z_grid*100, 'bo', markersize=10)
ax1.plot(tau_max_vs_z[idx_grid], z_grid*100, 'ro', markersize=10)
ax1.plot(tau_allow_vs_z[idx_grid], z_grid*100, 'go', markersize=10)

ax1.set_xlabel('Stress [kPa]', fontsize=10)
ax1.set_ylabel('Depth [cm]', fontsize=10)
ax1.set_title('Stress Distribution vs Depth', fontsize=11)
ax1.legend(fontsize=7)
ax1.grid(True, alpha=0.3)
ax1.invert_yaxis()

# --- 子图2：安全系数沿深度分布 ---
ax2 = axes[0, 1]
ax2.plot(FS_vs_z, z_range * 100, 'b-', linewidth=2.5)
ax2.axhline(y=z_grid * 100, color='gray', linestyle='--', linewidth=1.5)
ax2.axvline(x=FS_required, color='red', linestyle='--', linewidth=2,
            label=f'Required FS = {FS_required}')
ax2.axvline(x=1.0, color='orange', linestyle=':', linewidth=1,
            label='FS = 1.0 (limit)')

# 标注当前深度
ax2.plot(FS, z_grid*100, 'ro', markersize=12)
ax2.annotate(f'FS = {FS:.2f}\n@ z={z_grid*100:.0f}cm',
             xy=(FS, z_grid*100), xytext=(FS + 0.3, z_grid*100 + 2),
             arrowprops=dict(arrowstyle='->', color='red'),
             fontsize=10, color='red', fontweight='bold')

# 填充安全区域
ax2.fill_betweenx(z_range*100, FS_required, ax2.get_xlim()[1],
                  alpha=0.1, color='green', label=f'Safe zone (FS ≥ {FS_required})')
ax2.fill_betweenx(z_range*100, 1.0, FS_required,
                  alpha=0.1, color='yellow', label='Marginal zone')
ax2.fill_betweenx(z_range*100, 0, 1.0,
                  alpha=0.1, color='red', label='Failure zone')

ax2.set_xlabel('Safety Factor FS', fontsize=10)
ax2.set_ylabel('Depth [cm]', fontsize=10)
ax2.set_title('Safety Factor vs Depth', fontsize=11)
ax2.legend(fontsize=7)
ax2.grid(True, alpha=0.3)
ax2.invert_yaxis()
ax2.set_xlim(0, max(5, np.max(FS_vs_z) * 1.1))

# --- 子图3：莫尔应力圆与破坏包线 ---
ax3 = axes[1, 0]
# 莫尔圆绘制
sigma_center = (sigma_1 + sigma_3) / 2.0 * 1000  # kPa
sigma_radius = (sigma_1 - sigma_3) / 2.0 * 1000   # kPa

theta_circle = np.linspace(0, 2*np.pi, 200)
circle_sigma = sigma_center + sigma_radius * np.cos(theta_circle)
circle_tau = sigma_radius * np.sin(theta_circle)

ax3.plot(circle_sigma, np.abs(circle_tau), 'b-', linewidth=2, label='Mohr Circle')

# 破坏包线：τ = c + σ·tan(φ)
sigma_range_mpa = np.linspace(0, max(sigma_1, sigma_z) * 1500, 100)
failure_tau = c_cohesion + sigma_range_mpa * np.tan(phi_rad)
ax3.plot(sigma_range_mpa, failure_tau, 'r-', linewidth=2.5, label='Failure envelope')

# 标注最大剪应力点
ax3.plot(sigma_center, sigma_radius, 'ro', markersize=10,
         label=f'τ_max={tau_max_kPa:.0f} kPa')

# 标注法向应力在破坏包线上的对应强度
ax3.plot([sigma_n_kPa, sigma_n_kPa], [0, tau_allow],
         'g--', linewidth=1.5, alpha=0.7)
ax3.plot(sigma_n_kPa, tau_allow, 'gs', markersize=10,
         label=f'τ_allow={tau_allow:.0f} kPa')

ax3.set_xlabel('Normal Stress σ [kPa]', fontsize=10)
ax3.set_ylabel('Shear Stress τ [kPa]', fontsize=10)
ax3.set_title(f'Mohr Circle & Failure Envelope (FS={FS:.2f})', fontsize=11)
ax3.legend(fontsize=7)
ax3.grid(True, alpha=0.3)
ax3.set_xlim(0, max(sigma_1*1000*1.5, sigma_n_kPa*1.5))
ax3.set_ylim(0, sigma_radius * 1.5)
ax3.set_aspect('equal')

# --- 子图4：敏感性与改进建议 ---
ax4 = axes[1, 1]

# 双y轴图：FS vs φ 和 FS vs ν
color_phi = 'tab:blue'
color_nu = 'tab:red'

ax4.plot(phi_range, FS_vs_phi, color=color_phi, linewidth=2, label='FS vs φ')
ax4.axhline(y=FS_required, color='black', linestyle='--', linewidth=1.5,
            label=f'Required FS={FS_required}')
ax4.axhline(y=FS, color='gray', linestyle=':', linewidth=1)
ax4.axvline(x=phi_friction, color=color_phi, linestyle='--', alpha=0.5)
ax4.set_xlabel('Friction Angle φ [°]', fontsize=10, color=color_phi)
ax4.set_ylabel('Safety Factor FS', fontsize=10, color=color_phi)
ax4.tick_params(axis='y', labelcolor=color_phi)

ax4_twin = ax4.twiny()
ax4_twin.plot(nu_range, FS_vs_nu, color=color_nu, linewidth=2, linestyle='-.',
              label='FS vs ν')
ax4_twin.axvline(x=nu_asphalt, color=color_nu, linestyle='--', alpha=0.5)
ax4_twin.set_xlabel("Poisson's Ratio ν", fontsize=10, color=color_nu)
ax4_twin.tick_params(axis='x', labelcolor=color_nu)

# 标注参考线
ax4.axhline(y=1.0, color='orange', linestyle=':', linewidth=1, alpha=0.7)
ax4.set_title('Sensitivity Analysis', fontsize=11)
ax4.grid(True, alpha=0.2)

# 合并图例
lines1, labels1 = ax4.get_legend_handles_labels()
lines2, labels2 = ax4_twin.get_legend_handles_labels()
ax4.legend(lines1 + lines2, labels1 + labels2, fontsize=7, loc='lower right')

plt.tight_layout()
plt.savefig('task4_shear.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"\n图片已保存: task4_shear.png")

# ============================================================
# 9. 输出汇总
# ============================================================
shear_pass = FS >= FS_required

print(f"\n{'='*60}")
print(f"任务四 汇总结果")
print(f"{'='*60}")
print(f"  归一化深度 a = z/δ = {a:.3f}")
print(f"  竖向应力 σ_z = {sigma_z*1000:.1f} kPa")
print(f"  径向应力 σ_r = {sigma_r*1000:.1f} kPa")
print(f"  最大剪应力 τ_max = {tau_max_kPa:.1f} kPa")
print(f"  法向应力 σ_n = {sigma_n_kPa:.1f} kPa")
print(f"  剪切强度 τ_allow = {tau_allow:.1f} kPa")
print(f"  安全系数 FS = {FS:.3f}")
print(f"  要求安全系数 = {FS_required}")
print(f"  判断: {'✓ 通过' if shear_pass else '✗ 不通过'}")

# 保存结果
np.savez('task4_results.npz',
         sigma_z_kPa=sigma_z*1000,
         sigma_r_kPa=sigma_r*1000,
         tau_max_kPa=tau_max_kPa,
         tau_allow=tau_allow,
         FS=FS,
         FS_required=FS_required,
         shear_pass=shear_pass)

print(f"\n关键数据已保存至: task4_results.npz")
print(f"任务四脚本执行完毕。")
