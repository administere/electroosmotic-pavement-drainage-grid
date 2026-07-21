#!/usr/bin/env python3
"""
任务二：电渗透排水速率评估
=============================================
方法：基于 Helmholtz-Smoluchowski 方程计算电渗透流速，
      考虑多孔介质迂曲度和孔隙率修正，建立含水率衰减模型，
      评估24小时排水能力。

理论：
  v_eo = (ε_r * ε_0 * ζ / η) * E          (Helmholtz-Smoluchowski)
  q = v_eo * n / τ                          (多孔介质修正)
  Q(t) = ∫ q(t) dt                          (累积排水量)

作者：计算物理工程师
日期：2026-07-22
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 1. 设计参数定义
# ============================================================
epsilon_r = 12.0            # 水的相对介电常数
epsilon_0 = 8.854e-12       # 真空介电常数 [F/m]
zeta = 0.03                 # Zeta电位绝对值 [V]（由-30mV转化）
eta = 1.0e-3                # 水的动力粘度 [Pa·s] (20°C)
n_porosity = 0.15           # 孔隙率 [-]
tau_tortuosity = 2.5        # 迂曲度 [-]
V_applied = 48.0            # 工作电压 [V]
d_electrode = 0.4           # 电极间距 [m]
Q_target = 3.0              # 目标排水量 [L/m²]
T_drain = 24 * 3600         # 排水时限 [s] = 24小时
sigma_initial = 0.05        # 初始电导率 [S/m]
sigma_min_ratio = 0.20      # 有效排水停止时电导率占初始比例

# 尝试从任务一加载更精确的E场数据
try:
    data = np.load('task1_results.npz')
    E_avg_domain = float(data['E_avg_domain'])
    E_avg_top = float(data['E_avg_top'])
    V_d = float(data['V_d_ratio'])
    print(f"从 task1_results.npz 加载数据:")
    print(f"  E_avg_domain = {E_avg_domain:.2f} V/m")
    print(f"  E_avg_top = {E_avg_top:.2f} V/m")
    print(f"  V/d = {V_d:.2f} V/m")
    E_field = E_avg_domain  # 使用域内平均电场
except FileNotFoundError:
    print("未找到 task1_results.npz，使用 V/d 理论近似")
    V_d = V_applied / d_electrode
    E_field = V_d
    print(f"  E ≈ V/d = {E_field:.2f} V/m")

# ============================================================
# 2. 电渗透流速计算
# ============================================================
# Helmholtz-Smoluchowski 方程
# v_eo = (ε_r * ε_0 * ζ / η) * E
# 物理意义：双电层中净电荷在切向电场作用下迁移，带动流体运动
eo_mobility = epsilon_r * epsilon_0 * zeta / eta  # 电渗透迁移率 [m²/(V·s)]
v_eo = eo_mobility * E_field                        # 电渗透流速 [m/s]

print(f"\n{'='*50}")
print(f"电渗透排水速率计算")
print(f"{'='*50}")
print(f"  电渗透迁移率 μ_eo = {eo_mobility:.3e} m²/(V·s)")
print(f"  电渗透流速 v_eo  = {v_eo:.3e} m/s")
print(f"                     = {v_eo*1000:.3f} mm/s")
print(f"                     = {v_eo*3600*1000:.3f} mm/h")

# 多孔介质修正
# q = v_eo * n / τ
# n/τ 为有效流动截面修正因子
q_specific = v_eo * n_porosity / tau_tortuosity  # 比流量 [m³/(s·m²)] = [m/s]
q_specific_Lh = q_specific * 3600 * 1000          # [L/(h·m²)]
q_specific_Ld = q_specific * 86400 * 1000         # [L/(day·m²)] 理论恒定速率

print(f"\n  多孔介质修正因子 n/τ = {n_porosity/tau_tortuosity:.3f}")
print(f"  比流量 q = {q_specific:.3e} m/s = {q_specific_Lh:.4f} L/(h·m²)")
print(f"  恒定速率下24h排水量 = {q_specific_Ld:.2f} L/m²")
print(f"  目标排水量 = {Q_target} L/m²")
print(f"  初步满足率 = {q_specific_Ld/Q_target*100:.1f}%")

# ============================================================
# 3. 含水率衰减排水模型
# ============================================================
# 假设：
#   - 初始含水饱和，电导率 σ_0 = 0.05 S/m
#   - 排水过程中，含水率 θ 线性下降
#   - 电导率 ∝ 含水率：σ(θ) = σ_0 * (θ/θ_0)
#   - 当 σ 降至 σ_min = 0.2*σ_0 时，对应残余含水率 θ_r = 0.2*θ_0
#   - 有效排水量 = θ_0 - θ_r (可排水量)
#   - 排水速率 ∝ E，而 E ≈ V/d 保持不变（电压固定），但有效渗透率随含水率降低
#
# 更精确的衰减模型：
#   可排水总量 Q_max = (θ_0 - θ_r) * h_effective
#   其中 h_effective 为有效排水层厚度
#
# 这里采用简化模型：排水速率随剩余可排水量线性衰减
#   dQ/dt = q_0 * (1 - Q/Q_max)
#   其中 q_0 为初始排水速率，Q_max 为最大可排水量

# 初始体积含水率（假设饱和）≈ 孔隙率 = 0.15
theta_0 = n_porosity  # 初始体积含水率
theta_r = 0.2 * theta_0  # 残余含水率（对应σ降至20%）
drainable_porosity = theta_0 - theta_r  # 可排水孔隙率

# 有效排水深度：保守取电极影响深度
h_effective = 0.15  # [m] 电渗透有效影响深度（电极场穿透深度）

# 最大可排水量
Q_max = drainable_porosity * h_effective * 1000  # [L/m²]
print(f"\n  初始体积含水率 θ_0 = {theta_0:.3f}")
print(f"  残余体积含水率 θ_r = {theta_r:.3f}")
print(f"  可排水孔隙率 = {drainable_porosity:.3f}")
print(f"  有效排水深度 h_eff = {h_effective:.2f} m")
print(f"  理论最大可排水量 Q_max = {Q_max:.2f} L/m²")

# 排水速率衰减模型：指数衰减
# q(t) = q_0 * exp(-q_0 * t / Q_max)
# 此模型保证：∫₀^∞ q(t)dt = Q_max
# 推导：dQ/dt = q_0 * (1 - Q/Q_max) → Q(t) = Q_max * (1 - exp(-q_0*t/Q_max))
q_0 = q_specific * 1000  # 初始排水速率 [L/(s·m²)]

def drainage_rate(t, Q):
    """排水速率衰减模型 ODE"""
    if Q >= Q_max:
        return 0.0
    return q_0 * max(0, 1 - Q / Q_max)

# 数值积分求解
t_span = (0, T_drain)
t_eval = np.linspace(0, T_drain, 1000)

# 使用解析解
Q_analytic = Q_max * (1 - np.exp(-q_0 * t_eval / Q_max))
q_analytic = q_0 * np.exp(-q_0 * t_eval / Q_max)
Q_24h = Q_max * (1 - np.exp(-q_0 * T_drain / Q_max))

# 也使用 ODE 求解器验证
sol = solve_ivp(drainage_rate, t_span, [0], t_eval=t_eval, method='RK45')
Q_ode = sol.y[0]
Q_24h_ode = Q_ode[-1]

print(f"\n  初始排水速率 q_0 = {q_0*3600:.4f} L/(h·m²)")
print(f"  24h 累积排水量（解析）= {Q_24h:.3f} L/m²")
print(f"  24h 累积排水量（ODE）= {Q_24h_ode:.3f} L/m²")
print(f"  目标排水量 = {Q_target} L/m²")
print(f"  满足率 = {Q_24h/Q_target*100:.1f}%")

# ============================================================
# 4. 参数敏感性分析
# ============================================================
# 4.1 不同E场强度下的排水量
E_range = np.linspace(50, 300, 50)
Q_24h_E = np.zeros_like(E_range)
for i, E in enumerate(E_range):
    v_eo_i = eo_mobility * E
    q_i = v_eo_i * n_porosity / tau_tortuosity * 1000  # L/(s·m²)
    Q_24h_E[i] = Q_max * (1 - np.exp(-q_i * T_drain / Q_max))

# 4.2 不同电极间距下的排水量
d_range = np.linspace(0.2, 0.8, 50)
Q_24h_d = np.zeros_like(d_range)
for i, d_i in enumerate(d_range):
    E_i = V_applied / d_i
    v_eo_i = eo_mobility * E_i
    q_i = v_eo_i * n_porosity / tau_tortuosity * 1000
    Q_24h_d[i] = Q_max * (1 - np.exp(-q_i * T_drain / Q_max))

# ============================================================
# 5. 可视化
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 11))
fig.suptitle('任务二：电渗透排水速率评估', fontsize=14, fontweight='bold')

# --- 子图1：累积排水量随时间变化 ---
ax1 = axes[0, 0]
ax1.plot(t_eval / 3600, Q_analytic, 'b-', linewidth=2, label='Cumulative drainage Q(t)')
ax1.plot(t_eval / 3600, q_0 * t_eval, 'b--', linewidth=1, alpha=0.5,
         label=f'Constant rate (no attenuation)')
ax1.axhline(y=Q_target, color='r', linestyle='--', linewidth=1.5,
            label=f'Target: {Q_target} L/m²')
ax1.axhline(y=Q_max, color='gray', linestyle=':', linewidth=1,
            label=f'Max drainable: {Q_max:.2f} L/m²')
ax1.axvline(x=24, color='g', linestyle='-.', linewidth=1, alpha=0.5, label='24h limit')
ax1.fill_between([0, 24], 0, Q_target, alpha=0.1, color='green', label='Required zone')

# 标注24h值
ax1.plot(24, Q_24h, 'ro', markersize=8)
ax1.annotate(f'{Q_24h:.2f} L/m²\n({Q_24h/Q_target*100:.0f}%)',
             xy=(24, Q_24h), xytext=(20, Q_24h + 0.3),
             arrowprops=dict(arrowstyle='->', color='red'),
             fontsize=9, color='red', fontweight='bold')

ax1.set_xlabel('Time [hours]', fontsize=10)
ax1.set_ylabel('Cumulative Drainage [L/m²]', fontsize=10)
ax1.set_title('Cumulative Drainage vs Time', fontsize=11)
ax1.legend(loc='lower right', fontsize=7)
ax1.grid(True, alpha=0.3)
ax1.set_xlim(0, 30)

# --- 子图2：排水速率随时间衰减 ---
ax2 = axes[0, 1]
ax2.plot(t_eval / 3600, q_analytic * 3600 * 1000, 'b-', linewidth=2)
# 标注初始速率和24h速率
ax2.plot(0, q_0 * 3600 * 1000, 'go', markersize=8, label=f'Initial: {q_0*3600*1000:.4f} L/h/m²')
q_24h_rate = q_0 * np.exp(-q_0 * T_drain / Q_max) * 3600 * 1000
ax2.plot(24, q_24h_rate, 'ro', markersize=8, label=f'24h: {q_24h_rate:.4f} L/h/m²')
ax2.set_xlabel('Time [hours]', fontsize=10)
ax2.set_ylabel('Drainage Rate [L/(h·m²)]', fontsize=10)
ax2.set_title('Drainage Rate Decay over Time', fontsize=11)
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.3)

# --- 子图3：排水量对电场强度的敏感性 ---
ax3 = axes[1, 0]
ax3.plot(E_range, Q_24h_E, 'b-', linewidth=2)
ax3.axvline(x=E_field, color='orange', linestyle='--', linewidth=1.5,
            label=f'Design E ≈ {E_field:.0f} V/m')
ax3.axhline(y=Q_target, color='r', linestyle='--', linewidth=1.5, label=f'Target {Q_target} L/m²')
ax3.set_xlabel('Electric Field E [V/m]', fontsize=10)
ax3.set_ylabel('24h Drainage [L/m²]', fontsize=10)
ax3.set_title('Sensitivity: 24h Drainage vs E-field', fontsize=11)
ax3.legend(fontsize=8)
ax3.grid(True, alpha=0.3)

# --- 子图4：排水量对电极间距的敏感性 ---
ax4 = axes[1, 1]
ax4.plot(d_range, Q_24h_d, 'b-', linewidth=2)
ax4.axvline(x=d_electrode, color='orange', linestyle='--', linewidth=1.5,
            label=f'Design d = {d_electrode:.1f} m')
ax4.axhline(y=Q_target, color='r', linestyle='--', linewidth=1.5, label=f'Target {Q_target} L/m²')
ax4.set_xlabel('Electrode Spacing d [m]', fontsize=10)
ax4.set_ylabel('24h Drainage [L/m²]', fontsize=10)
ax4.set_title('Sensitivity: 24h Drainage vs Spacing', fontsize=11)
ax4.legend(fontsize=8)
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('task2_drainage.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"\n图片已保存: task2_drainage.png")

# ============================================================
# 6. 判断与改进建议
# ============================================================
print(f"\n{'='*60}")
print(f"任务二 汇总结果")
print(f"{'='*60}")
print(f"  电渗透迁移率:        {eo_mobility:.3e} m²/(V·s)")
print(f"  电渗透流速 v_eo:     {v_eo:.3e} m/s")
print(f"  多孔介质比流量 q:    {q_specific:.3e} m/s = {q_specific_Ld:.2f} L/(day·m²)")
print(f"  24h 累积排水量:      {Q_24h:.3f} L/m²")
print(f"  目标排水量:          {Q_target} L/m²")
print(f"  满足率:              {Q_24h/Q_target*100:.1f}%")
drainage_pass = Q_24h >= Q_target
print(f"  判断: {'✓ 通过' if drainage_pass else '✗ 不通过'} "
      f"- 24小时内{'可以' if drainage_pass else '无法'}排完{Q_target} L/m²")

if not drainage_pass:
    # 计算需要的最小E场
    # Q_target = Q_max * (1 - exp(-q_0*T/Q_max))
    # 其中 q_0 = (ε_r*ε_0*ζ/η)*E * n/τ * 1000
    # 解出 E_min
    from scipy.optimize import fsolve

    def Q_24h_func(E):
        v = eo_mobility * E
        q = v * n_porosity / tau_tortuosity * 1000
        return Q_max * (1 - np.exp(-q * T_drain / Q_max)) - Q_target

    # 寻找使 Q_24h = Q_target 的 E
    E_guess = E_field
    for _ in range(10):
        Q_guess = Q_24h_func(E_guess)
        if abs(Q_guess) < 0.01:
            break
        E_guess *= 1.5

    try:
        E_min = fsolve(Q_24h_func, E_guess * 2)[0]
        V_min = E_min * d_electrode
        print(f"\n  改进建议:")
        print(f"    所需最小电场: E_min = {E_min:.1f} V/m")
        print(f"    对应工作电压: V_min = {V_min:.1f} V (当前 {V_applied} V)")
        print(f"    或缩小电极间距至: d_max = {V_applied/E_min:.3f} m (当前 {d_electrode} m)")
    except:
        print(f"  无法通过调整单一参数满足排水需求，需综合优化")

# 保存结果
np.savez('task2_results.npz',
         eo_mobility=eo_mobility,
         v_eo=v_eo,
         q_specific=q_specific,
         Q_24h=Q_24h,
         Q_target=Q_target,
         drainage_pass=drainage_pass,
         E_field=E_field)

print(f"\n关键数据已保存至: task2_results.npz")
print(f"任务二脚本执行完毕。")
