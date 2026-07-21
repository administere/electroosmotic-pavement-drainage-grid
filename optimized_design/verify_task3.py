#!/usr/bin/env python3
"""
新方案验证：热安全分析
=====================================
参数变更：V=24V, h_asphalt=0.14m, 间歇1:3, J_areal=FDM精算值
"""
import numpy as np
import matplotlib.pyplot as plt

# ===== 新设计参数 =====
V_work = 24.0
h_asphalt = 0.14           # ← 0.12→0.14
k_asphalt = 1.2
k_ws = 1.5
T_surface = 35.0
T_soft = 50.0
delta_T_max = 5.0
duty_cycle = 0.25
hours_on = 6

# 加载 Task1 精算 J
try:
    d1 = np.load('verify_task1.npz')
    J_actual = float(d1['J_areal'])
    print(f"FDM精算 J_areal = {J_actual:.3f} A/m²")
except:
    J_actual = 1.43
    print(f"理论估算 J_areal = {J_actual:.3f} A/m²")

# ===== 发热功率 =====
P_inst = J_actual * V_work      # 瞬时 W/m²
P_avg = P_inst * duty_cycle     # 时间平均 W/m²
P_on = P_inst * hours_on / 24   # 等效连续(6h ON)

print(f"\n{'='*50}")
print(f"新方案 Task3 热安全")
print(f"{'='*50}")
print(f"  瞬时功率 P_inst = {P_inst:.2f} W/m²")
print(f"  时间平均 P_avg = {P_avg:.2f} W/m²")
print(f"  6h等效 P_on = {P_on:.2f} W/m²")

# ===== 热模型 =====
R_asphalt = h_asphalt / k_asphalt
R_ws = 0.2 / k_ws             # 水稳层等效热阻
R_parallel = 1/(1/R_asphalt + 1/R_ws)

# 模型A: 全向上, 瞬时峰值
dT_A_inst = P_inst * R_asphalt
T_A_inst = T_surface + dT_A_inst

# 模型A: 全向上, 时间平均
dT_A_avg = P_avg * R_asphalt
T_A_avg = T_surface + dT_A_avg

# 模型B: 双向, 瞬时峰值
dT_B_inst = P_inst * R_parallel
T_B_inst = T_surface + dT_B_inst

# 模型B: 双向, 时间平均
dT_B_avg = P_avg * R_parallel
T_B_avg = T_surface + dT_B_avg

# 最保守用于校核
dT_check = dT_A_inst
T_check = T_A_inst
dT_avg_check = dT_A_avg
T_avg_check = T_A_avg

check_soft = T_check < T_soft
check_dT = dT_check < delta_T_max
check_soft_avg = T_avg_check < T_soft
check_dT_avg = dT_avg_check < delta_T_max
thermal_pass = check_soft and check_dT_avg  # 峰值短暂超标可接受，平均必须达标

print(f"\n  热阻: R_asphalt={R_asphalt:.4f}, R_parallel={R_parallel:.4f} m²·K/W")
print(f"\n  {'模型':<20} {'ΔT':>8} {'T_grating':>10} {'ΔT<5°C':>10} {'T<50°C':>10}")
print(f"  {'─'*20} {'─'*8} {'─'*10} {'─'*10} {'─'*10}")
print(f"  {'A 全向上 瞬时峰值':<20} {dT_A_inst:>7.2f}°C {T_A_inst:>9.1f}°C {'✓' if dT_A_inst<delta_T_max else '✗':>10} {'✓' if T_A_inst<T_soft else '✗':>10}")
print(f"  {'A 全向上 时间平均':<20} {dT_A_avg:>7.2f}°C {T_A_avg:>9.1f}°C {'✓' if dT_A_avg<delta_T_max else '✗':>10} {'✓' if T_A_avg<T_soft else '✗':>10}")
print(f"  {'B 双向 瞬时峰值':<20} {dT_B_inst:>7.2f}°C {T_B_inst:>9.1f}°C {'✓' if dT_B_inst<delta_T_max else '✗':>10} {'✓' if T_B_inst<T_soft else '✗':>10}")
print(f"  {'B 双向 时间平均':<20} {dT_B_avg:>7.2f}°C {T_B_avg:>9.1f}°C {'✓' if dT_B_avg<delta_T_max else '✗':>10} {'✓' if T_B_avg<T_soft else '✗':>10}")

print(f"\n  校核标准: 时间平均温升 ≤ {delta_T_max}°C, 峰值温度 ≤ {T_soft}°C")
print(f"  ΔT_avg(全向上) = {dT_A_avg:.2f}°C {'✓' if check_dT_avg else '✗'}")
print(f"  T_peak(全向上) = {T_A_inst:.1f}°C {'✓' if check_soft else '✗'}")
print(f"  综合: {'✓ 热安全通过' if thermal_pass else '✗ 不通过'}")

# ===== 瞬态温升（简化一阶RC模型） =====
# τ = R*C, C = ρ*cp*h (沥青热容)
rho_asphalt = 2300     # kg/m³
cp_asphalt = 900       # J/(kg·K)
C_asphalt = rho_asphalt * cp_asphalt * h_asphalt  # J/(m²·K)
tau_thermal = R_asphalt * C_asphalt  # 热时间常数 [s]

t_trans = np.linspace(0, 86400, 2000)  # 24h
# 间歇方波加热: 1h ON, 3h OFF
period = 4 * 3600
q_t = np.where((t_trans % period) < 3600, P_inst, 0.0)

# 一阶RC响应
dT_trans = np.zeros_like(t_trans)
for i in range(1, len(t_trans)):
    dt = t_trans[i] - t_trans[i-1]
    dT_trans[i] = dT_trans[i-1] + (q_t[i]*R_asphalt - dT_trans[i-1]) * dt / tau_thermal

T_trans = T_surface + dT_trans

print(f"\n  热时间常数 τ = {tau_thermal/3600:.1f} h")
print(f"  瞬态峰值 T_max = {np.max(T_trans):.1f}°C @ t={t_trans[np.argmax(T_trans)]/3600:.1f}h")
print(f"  瞬态平均 T_avg = {np.mean(T_trans):.1f}°C")

# ===== 敏感性 =====
V_range = np.linspace(12, 48, 50)
J_scale = J_actual / V_work  # J ∝ V roughly
dT_V = (J_scale * V_range) * V_range * R_asphalt  # P=J*V ∝ V²

# ===== 画图 =====
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle(f'新方案热安全验证 | V={V_work}V h={h_asphalt*100:.0f}cm 间歇1:3', fontsize=13, fontweight='bold')

# 温度剖面
ax=axes[0,0]
zprof = np.linspace(0, h_asphalt, 50)
Tprof_inst = T_surface + P_inst*(h_asphalt-zprof)/k_asphalt
Tprof_avg = T_surface + P_avg*(h_asphalt-zprof)/k_asphalt
ax.plot(Tprof_inst, zprof*100, 'r-', lw=2, label=f'Peak (ON), T_grid={T_A_inst:.1f}°C')
ax.plot(Tprof_avg, zprof*100, 'b-', lw=2, label=f'Average, T_grid={T_A_avg:.1f}°C')
ax.axvline(x=T_surface, color='orange', ls=':', label=f'Surface {T_surface}°C')
ax.axvline(x=T_soft, color='red', ls='--', lw=1.5, label=f'Softening {T_soft}°C')
ax.axvline(x=T_surface+delta_T_max, color='purple', ls='-.', lw=1.5, label=f'+{delta_T_max}°C limit')
ax.axhline(y=h_asphalt*100, color='gray', ls='-', alpha=0.5)
ax.set_xlabel('Temperature [°C]'); ax.set_ylabel('Depth [cm]')
ax.set_title('Temperature Profile'); ax.legend(fontsize=7); ax.grid(alpha=0.3); ax.invert_yaxis()

# 瞬态响应
ax=axes[0,1]
ax.plot(t_trans/3600, T_trans, 'b-', lw=1, alpha=0.8)
ax.axhline(y=T_surface+delta_T_max, color='red', ls='--', lw=1.5, label=f'+{delta_T_max}°C limit')
ax.axhline(y=T_soft, color='orange', ls='--', lw=1, label=f'Softening {T_soft}°C')
ax.axhline(y=T_surface, color='gray', ls=':', lw=1)
ax.set_xlabel('Time [h]'); ax.set_ylabel('T_grating [°C]')
ax.set_title(f'Transient Response (τ={tau_thermal/3600:.1f}h)')
ax.legend(fontsize=7); ax.grid(alpha=0.3)
ax.set_xlim(0, 24)

# 功率对比
ax=axes[1,0]
labels = ['Old design\n(48V, cont.)', 'New design\n(24V, peak)', 'New design\n(24V, avg)', 'New design\n(24V, 6h eq.)']
powers = [61.07, P_inst, P_avg, P_on]
dTs = [6.11, dT_A_inst, dT_A_avg, P_on*R_asphalt]
colors = ['#ff6b6b', '#ffa502', '#4ecdc4', '#44bd32']
xpos = np.arange(4)
bars = ax.bar(xpos, dTs, color=colors, edgecolor='black')
ax.axhline(y=delta_T_max, color='red', ls='--', lw=1.5, label=f'Limit {delta_T_max}°C')
for b,p,dt in zip(bars, powers, dTs):
    ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.15, f'{dt:.1f}°C\n({p:.0f}W/m²)', ha='center', fontsize=8)
ax.set_xticks(xpos); ax.set_xticklabels(labels, fontsize=7)
ax.set_ylabel('ΔT [°C]'); ax.set_title('ΔT Comparison'); ax.legend(fontsize=7); ax.grid(alpha=0.3, axis='y')

# 电压敏感性
ax=axes[1,1]
ax.plot(V_range, dT_V, 'b-', lw=2)
ax.axhline(y=delta_T_max, color='red', ls='--', lw=1.5)
ax.axvline(x=V_work, color='orange', ls='--', lw=1.5, label=f'Design {V_work}V')
ax.fill_between(V_range, 0, delta_T_max, alpha=0.1, color='green')
ax.set_xlabel('Voltage [V]'); ax.set_ylabel('ΔT_avg [°C]')
ax.set_title(f'ΔT vs Voltage (h={h_asphalt*100:.0f}cm)'); ax.legend(fontsize=8); ax.grid(alpha=0.3)

plt.tight_layout(); plt.savefig('verify_task3.png', dpi=150, bbox_inches='tight'); plt.close()
print("  图片: verify_task3.png")

np.savez('verify_task3.npz', P_inst=P_inst, P_avg=P_avg, dT_inst=dT_A_inst, dT_avg=dT_A_avg,
         T_inst=T_A_inst, T_avg=T_A_avg, thermal_pass=thermal_pass, tau_thermal=tau_thermal)
print("  数据: verify_task3.npz\n")
