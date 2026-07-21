#!/usr/bin/env python3
"""
新方案验证：排水能力（EO+重力复合）
=====================================
参数变更：V=24V, d=0.30m, ζ=50mV, 3D芯体重力排水为主
"""
import numpy as np
import matplotlib.pyplot as plt

# ===== 新设计参数 =====
epsilon_r, epsilon_0 = 12.0, 8.854e-12
zeta = 0.050              # ← 0.03→0.05 (硅烷改性)
eta = 1.0e-3
n_porosity, tau_tort = 0.15, 2.5
V_applied, d_electrode = 24.0, 0.30  # ← 48→24, 0.4→0.3
Q_target, T_drain = 3.0, 24*3600

# 加载 Task1 精算电场
try:
    d1 = np.load('verify_task1.npz')
    E_field = float(d1['E_avg_domain'])
    E_top = float(d1['E_avg_top'])
    print(f"FDM精算: E_domain={E_field:.1f}, E_top={E_top:.1f} V/m")
except:
    E_field = V_applied/d_electrode
    print(f"理论近似: E≈V/d={E_field:.1f} V/m")

# ===== EO 计算 =====
mu_eo = epsilon_r*epsilon_0*zeta/eta
v_eo = mu_eo * E_field
q_eo = v_eo * n_porosity / tau_tort  # m/s
q_eo_Lh = q_eo * 3600 * 1000         # L/(h·m²)

# 间歇运行: 1h ON / 3h OFF, 每日6h有效
duty = 0.25
hours_on_per_day = 6
Q_eo_24h = q_eo_Lh * hours_on_per_day  # L/m²

print(f"\n{'='*50}")
print(f"新方案 Task2 排水计算")
print(f"{'='*50}")
print(f"  μ_eo = {mu_eo:.3e} m²/(V·s)")
print(f"  v_eo = {v_eo:.3e} m/s = {v_eo*3600*1000:.2f} mm/h")
print(f"  q_eo = {q_eo:.3e} m/s = {q_eo_Lh:.4f} L/(h·m²)")
print(f"  EO 24h (间歇{hours_on_per_day}h): {Q_eo_24h:.3f} L/m²")

# ===== 重力排水（3D芯体） =====
theta_core = 2e-3        # m²/s 导水率
i_slope = 0.02           # 2%横坡
L_drain = 10.0           # 集水沟间距 m
t_core = 0.008           # 芯体厚度 m
crush_factor = 0.5       # 碾压厚度折减
theta_eff = theta_core * 0.25  # 碾压+长期蠕变折减至25%

q_g_per_m = theta_eff * i_slope  # m³/(s·m_width)
q_g_per_m2 = q_g_per_m / L_drain * 1000 * 3600  # L/(h·m²)
Q_g_24h = q_g_per_m2 * 24  # L/m²

print(f"\n  3D芯体重力排水:")
print(f"    有效导水率 θ_eff = {theta_eff:.2e} m²/s (最不利折减)")
print(f"    单位宽导水 q = {q_g_per_m*3600*1000:.1f} L/(h·m)")
print(f"    面均导水 q = {q_g_per_m2:.2f} L/(h·m²)")
print(f"    重力24h: {Q_g_24h:.1f} L/m²")

# ===== EO衰减模型（间歇） =====
# 每天6个ON周期，每个1h，间隔3h
# EO排水主要在ON期间，OFF期间重力继续排水
t_hours = np.linspace(0, 24, 1000)
Q_cumulative = np.zeros_like(t_hours)
Q_total = 0.0

# 简化：前6h等效ON时间内的指数衰减
Q_max_eo = 18.0  # 最大电渗透可释放量 L/m² (含水孔隙)
q0_eo = q_eo_Lh  # L/(h·m²)
# 有效EO时间
t_on = np.minimum(t_hours * duty, hours_on_per_day)
Q_eo = Q_max_eo * (1 - np.exp(-q0_eo * t_on / Q_max_eo))
# 重力连续排水（只要有自由水）
q0_g = q_g_per_m2  # L/(h·m²)
Q_g = q0_g * t_hours
# 重力只能排走EO释放的水
Q_total = np.minimum(Q_eo + Q_g, Q_max_eo + q0_g * t_hours)
# 实际上 Q = Q_eo + min(Q_g, 取决于水量)

# 简化处理：总排水 = EO释放 + 重力排走（受限于总水量）
Q_g_effective = np.minimum(q0_g * t_hours, Q_max_eo)
Q_cumulative = Q_eo + Q_g_effective

Q_24h_total = Q_eo[-1] + min(q0_g * 24, Q_max_eo)
# 实际上重力远大于EO，总排水受限于可排水量
Q_24h_realistic = min(Q_max_eo + q0_g * 24, Q_max_eo + 100)  # 上限18+ = 很快排完
Q_24h_cons = Q_eo_24h  # 保守只算EO

print(f"\n  复合排水 24h:")
print(f"    EO贡献: {Q_eo_24h:.2f} L/m²")
print(f"    重力贡献: >> {Q_g_24h:.1f} L/m² (受限于可排水量)")
print(f"    保守估计: {Q_24h_cons:.2f} L/m² (仅算EO)")

# 仅EO能否满足？
eo_only_pass = Q_eo_24h >= Q_target
# 复合能否满足？
combined_pass = (Q_eo_24h + min(Q_g_24h, 100)) >= Q_target

# 计算仅靠EO需要的参数
if not eo_only_pass:
    E_needed = Q_target / (Q_eo_24h / E_field) if Q_eo_24h > 0 else 999
    zeta_needed = zeta * Q_target / Q_eo_24h if Q_eo_24h > 0 else 999
    print(f"\n  若仅靠EO达到{Q_target} L/m²:")
    print(f"    需E={E_needed:.0f} V/m 或 ζ={zeta_needed*1000:.0f} mV")

print(f"\n  目标{Q_target} L/m²: {'✓ 通过(复合)' if combined_pass else '✗ 不通过'}")

# ===== 画图 =====
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle(f'新方案排水验证 | ζ={zeta*1000:.0f}mV V={V_applied}V d={d_electrode}m 间歇1:3', fontsize=13, fontweight='bold')

# 累积排水
ax=axes[0]
ax.plot(t_hours, Q_eo, 'b-', lw=2, label=f'EO ({hours_on_per_day}h ON)')
ax.plot(t_hours, q0_g*t_hours, 'g--', lw=1.5, alpha=0.7, label=f'Gravity (θ_eff={theta_eff:.1e})')
ax.plot(t_hours, Q_eo + np.minimum(q0_g*t_hours, Q_max_eo), 'r-', lw=2.5, label='EO+Gravity')
ax.axhline(y=Q_target, color='red', ls='--', lw=1.5, label=f'Target {Q_target} L/m²')
ax.axvline(x=6, color='gray', ls=':', alpha=0.5)
ax.annotate(f'{Q_eo_24h:.2f} (EO only)\n+重力 = >>{Q_target}', xy=(6, Q_eo_24h),
            xytext=(8, Q_eo_24h+1), arrowprops=dict(arrowstyle='->'), fontsize=8, color='blue')
ax.set_xlabel('Time [h]'); ax.set_ylabel('Cumulative [L/m²]')
ax.set_title('Drainage: EO + Gravity'); ax.legend(fontsize=7); ax.grid(alpha=0.3)

# EO 参数敏感性
ax=axes[1]
zeta_range = np.linspace(20, 100, 50)
Q_zeta = np.array([(epsilon_r*epsilon_0*zv*1e-3/eta)*E_field*n_porosity/tau_tort*3600*1000*6 for zv in zeta_range])
ax.plot(zeta_range, Q_zeta, 'b-', lw=2)
ax.axhline(y=Q_target, color='red', ls='--', lw=1.5, label=f'Target {Q_target} L/m²')
ax.axvline(x=zeta*1000, color='orange', ls='--', lw=1.5, label=f'Design ζ={zeta*1000:.0f}mV')
ax.fill_between(zeta_range, Q_target, 10, alpha=0.1, color='green', label='Pass zone')
ax.set_xlabel('Zeta Potential [mV]'); ax.set_ylabel('EO 24h Drainage [L/m²]')
ax.set_title('EO-only drainage vs ζ'); ax.legend(fontsize=7); ax.grid(alpha=0.3)

# 机制对比
ax=axes[2]
mechanisms = ['EO only\n(old design)', 'EO only\n(new design)', 'Gravity\n(3D core)', 'EO+Gravity\n(combined)']
values = [1.05, Q_eo_24h, min(Q_g_24h, 20), Q_eo_24h + min(Q_g_24h, 20)]
colors = ['#ff6b6b', '#ffa502', '#4ecdc4', '#44bd32']
bars = ax.bar(range(4), values, color=colors, edgecolor='black')
ax.axhline(y=Q_target, color='red', ls='--', lw=1.5, label=f'Target {Q_target} L/m²')
for b,v in zip(bars, values):
    ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.2, f'{v:.1f}', ha='center', fontweight='bold', fontsize=10)
ax.set_xticks(range(4)); ax.set_xticklabels(mechanisms, fontsize=8)
ax.set_ylabel('24h Drainage [L/m²]'); ax.set_title('Mechanism Comparison')
ax.legend(fontsize=7); ax.grid(alpha=0.3, axis='y')

plt.tight_layout(); plt.savefig('verify_task2.png', dpi=150, bbox_inches='tight'); plt.close()
print("  图片: verify_task2.png")

np.savez('verify_task2.npz', Q_eo_24h=Q_eo_24h, Q_g_24h=Q_g_24h, mu_eo=mu_eo,
         v_eo=v_eo, q_eo=q_eo, eo_only_pass=eo_only_pass, combined_pass=combined_pass)
print("  数据: verify_task2.npz\n")
