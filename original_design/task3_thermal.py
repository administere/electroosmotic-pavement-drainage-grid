#!/usr/bin/env python3
"""
任务三：热安全分析（格栅发热与沥青层温升）
=============================================
方法：
  1. 计算格栅焦耳发热功率（电流流经水稳层介质产热）
  2. 一维稳态热传导模型（傅里叶定律）
  3. 评估格栅温度是否超过沥青软化点及温升限值
  4. 若超限，给出改进建议

传热模型：
  - 发热机理：电流从阳极经水稳层介质流向阴极，介质电阻产热
  - 单位面积发热功率 P = J_areal × V（面电流密度 × 工作电压）
  - 热量主要通过沥青层向上传导至路面（T_surface = 35°C）
  - 部分热量向下传入水稳层
  - 稳态一维：q = k * ΔT / h

作者：计算物理工程师
日期：2026-07-22
"""

import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 1. 设计参数定义
# ============================================================
V_work = 48.0             # 工作电压 [V]
J_target = 0.3            # 目标面电流密度 [A/m²]
R_sheet = 10.0            # 碳纤维格栅面电阻 [Ω/sq]
h_asphalt = 0.12          # 沥青层厚度 [m]
k_asphalt = 1.2           # 沥青导热系数 [W/(m·K)]
k_waterstable = 1.5       # 水稳层导热系数 [W/(m·K)]
T_surface = 35.0          # 路表温度 [°C]（夏季最不利工况）
T_softening = 50.0        # 沥青软化点控制温度 [°C]
delta_T_max = 5.0         # 容许最大附加温升 [°C]

# 尝试从任务一加载实际电流密度
try:
    data = np.load('task1_results.npz')
    J_actual = float(data['J_areal'])
    print(f"从 task1_results.npz 加载: J_areal = {J_actual:.3f} A/m²")
except FileNotFoundError:
    J_actual = J_target
    print(f"未找到 task1_results.npz，使用目标值 J = {J_target} A/m²")

# ============================================================
# 2. 发热功率计算
# ============================================================
# 发热机理分析：
#   叉指电极结构中，电流从阳极碳纤维束 → 水稳层介质 → 阴极碳纤维束
#   主要焦耳热产生在水稳层介质中（介质电阻远大于碳纤维电阻）
#
# 方法一（主要）：P = J_areal × V
#   物理含义：单位面积格栅驱动电流 J [A/m²]，电子在电压 V 下降落，
#   释放的能量全部转化为介质焦耳热。
#   这是最直接的面平均发热功率。
P_heat_per_area = J_actual * V_work  # [W/m²]

# 方法二（校核）：P = I² × R_eff
#   从FDM结果：I_per_meter / period_width = J_areal
#   等效面电阻 R_eff = V / J_areal = 48/1.272 = 37.7 Ω·m²
#   P = V²/R_eff = J_areal² × R_eff = J_areal × V
#   与法一一致。
R_eff = V_work / J_actual  # 等效面电阻 [Ω·m²]
P_heat_check = V_work**2 / R_eff  # 应与法一相同

# 方法三（参考）：格栅碳纤维束自身的欧姆发热
#   碳纤维束纵向导电，面电阻 R_sheet=10 Ω/sq 表示电流在面内流动的电阻
#   但碳纤维仅是汇流电极，电流密度远小于介质中的电流密度
#   碳纤维发热 ≈ (J_carbon)² × R_sheet，J_carbon << J_areal
#   碳纤维发热可忽略不计

# 方法四（参考）：对应目标电流密度的发热
P_heat_target = J_target * V_work  # 若J降至目标值时的发热 [W/m²]

print(f"\n{'='*50}")
print(f"格栅发热功率计算")
print(f"{'='*50}")
print(f"  实际面电流密度 J_areal = {J_actual:.3f} A/m²")
print(f"  等效介质面电阻 R_eff = V/J = {R_eff:.1f} Ω·m²")
print(f"  方法一 P = J·V = {J_actual:.3f} × {V_work} = {P_heat_per_area:.2f} W/m²")
print(f"  方法二 P = V²/R_eff = {V_work}²/{R_eff:.1f} = {P_heat_check:.2f} W/m²")
print(f"  对照-目标电流 P_target = J_target × V = {P_heat_target:.2f} W/m²")
print(f"  (碳纤维自身发热可忽略，因其为高导电汇流电极)")
print(f"  采用 P_heat = {P_heat_per_area:.2f} W/m²")

P_heat = P_heat_per_area

# ============================================================
# 3. 一维稳态热传导模型
# ============================================================
# 模型：格栅为面热源，位于沥青层底部
#       热量向上传导至路面(T_surface)，向下至水稳层深处
#
# 注意：焦耳热实际分布在介质中（有一定深度分布），保守假设全部在界面
#
# 稳态热路（并联）：
#        T_surface ──[R_asphalt]── T_grating ──[R_waterstable]── T_deep
#                    ← q_up              → q_down
#                     q_up + q_down = P_heat

# 热阻计算
R_asphalt = h_asphalt / k_asphalt  # [m²·K/W]

# 水稳层视为半无限体等效热阻
# 特征导热长度取电极场影响深度 ≈ 0.2m（保守）或取稳态热扩散长度
L_char = 0.2  # 水稳层特征导热深度 [m]
R_waterstable = L_char / k_waterstable  # [m²·K/W]

print(f"\n{'='*50}")
print(f"热传导模型")
print(f"{'='*50}")
print(f"  沥青层热阻 R_asphalt = h/k = {h_asphalt}/{k_asphalt} = {R_asphalt:.4f} m²·K/W")
print(f"  水稳层热阻 R_waterstable = L/k = {L_char}/{k_waterstable} = {R_waterstable:.4f} m²·K/W")

# 模型A：全向上（最保守边界）
delta_T_A = P_heat * R_asphalt
T_grating_A = T_surface + delta_T_A

print(f"\n  模型A（全向上，最保守）:")
print(f"    ΔT = P × R_asphalt = {P_heat:.1f} × {R_asphalt:.4f} = {delta_T_A:.2f} °C")
print(f"    T_grating = {T_surface} + {delta_T_A:.2f} = {T_grating_A:.2f} °C")

# 模型B：双向热传导（更真实）
# 假设向下远端温度 T_deep ≈ T_surface（水稳层深处接近路表温度）
# 并联热阻
R_parallel = 1.0 / (1.0/R_asphalt + 1.0/R_waterstable)
delta_T_B = P_heat * R_parallel
T_grating_B = T_surface + delta_T_B
q_up_B = (T_grating_B - T_surface) / R_asphalt
q_down_B = (T_grating_B - T_surface) / R_waterstable

print(f"\n  模型B（双向导热，更真实）:")
print(f"    并联热阻 R_parallel = 1/(1/{R_asphalt:.4f}+1/{R_waterstable:.4f}) = {R_parallel:.4f} m²·K/W")
print(f"    ΔT = P × R_parallel = {P_heat:.1f} × {R_parallel:.4f} = {delta_T_B:.2f} °C")
print(f"    T_grating = {T_surface} + {delta_T_B:.2f} = {T_grating_B:.2f} °C")
print(f"    向上热流 q_up = {q_up_B:.2f} W/m² ({q_up_B/P_heat*100:.1f}%)")
print(f"    向下热流 q_down = {q_down_B:.2f} W/m² ({q_down_B/P_heat*100:.1f}%)")

# 模型C：若电流密度降至目标值 J_target
delta_T_C = P_heat_target * R_asphalt
T_grating_C = T_surface + delta_T_C

print(f"\n  模型C（若J降至目标值 {J_target} A/m²，全向上）:")
print(f"    P_target = {P_heat_target:.2f} W/m²")
print(f"    ΔT = {delta_T_C:.2f} °C")
print(f"    T_grating = {T_grating_C:.2f} °C")

# 采用模型A（最保守）作为评定基准
delta_T = delta_T_A
T_grating = T_grating_A

print(f"\n  采用模型A（最保守）作为评定基准")

# ============================================================
# 4. 安全判定
# ============================================================
check_softening = T_grating < T_softening
check_delta_T = delta_T < delta_T_max
thermal_pass = check_softening and check_delta_T

print(f"\n{'='*50}")
print(f"热安全判定")
print(f"{'='*50}")
print(f"  格栅温度 T_grating = {T_grating:.2f} °C")
print(f"  沥青软化点 T_soft = {T_softening} °C (裕度 {T_softening - T_grating:.1f} °C)")
print(f"  附加温升 ΔT = {delta_T:.2f} °C")
print(f"  容许最大温升 = {delta_T_max} °C (裕度 {delta_T_max - delta_T:.1f} °C)")
print(f"  软化点检查: {'✓ 通过' if check_softening else '✗ 不通过'}")
print(f"  温升限值检查: {'✓ 通过' if check_delta_T else '✗ 不通过'}")
print(f"  综合判断: {'✓ 热安全通过' if thermal_pass else '✗ 热安全不通过'}")

# ============================================================
# 5. 改进建议
# ============================================================
print(f"\n{'='*50}")
print(f"改进建议")
print(f"{'='*50}")

if not thermal_pass:
    if not check_delta_T:
        # ΔT = J*V * h/k < ΔT_max
        # J_max = ΔT_max * k / (V * h)
        J_max_dT = delta_T_max * k_asphalt / (V_work * h_asphalt)
        V_max_dT = delta_T_max * k_asphalt / (J_actual * h_asphalt)
        print(f"  为满足ΔT ≤ {delta_T_max}°C:")
        print(f"    降低面电流密度至: J ≤ {J_max_dT:.3f} A/m²")
        print(f"    或降低工作电压至: V ≤ {V_max_dT:.1f} V")

    if not check_softening:
        delta_T_soft = T_softening - T_surface
        J_max_soft = delta_T_soft * k_asphalt / (V_work * h_asphalt)
        print(f"  为满足T_grating ≤ {T_softening}°C:")
        print(f"    降低面电流密度至: J ≤ {J_max_soft:.3f} A/m²")

    # 若同时降低J至目标值
    print(f"\n  若将J从{J_actual:.3f}降至{J_target:.3f} A/m²:")
    delta_T_new = P_heat_target * R_asphalt
    print(f"    ΔT_new = {delta_T_new:.2f} °C (当前 {delta_T:.2f} °C)")

    # 增加沥青厚度
    h_needed = delta_T_max * k_asphalt / P_heat
    print(f"  若保持当前J={J_actual:.3f} A/m²，需增加沥青厚度至: h ≥ {h_needed*100:.1f} cm")

# ============================================================
# 6. 可视化
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 11))
fig.suptitle('任务三：热安全分析', fontsize=14, fontweight='bold')

# --- 子图1：沥青层温度剖面 ---
ax1 = axes[0, 0]
z_profile = np.linspace(0, h_asphalt, 50)
T_profile_A = T_surface + P_heat * (h_asphalt - z_profile) / k_asphalt
T_profile_B = T_surface + q_up_B * (h_asphalt - z_profile) / k_asphalt
T_profile_C = T_surface + P_heat_target * (h_asphalt - z_profile) / k_asphalt

ax1.plot(T_profile_A, z_profile * 100, 'r-', linewidth=2.5, label=f'Model A (all up), J={J_actual:.2f}')
ax1.plot(T_profile_B, z_profile * 100, 'b--', linewidth=2, label=f'Model B (bidirectional)')
ax1.plot(T_profile_C, z_profile * 100, 'g:', linewidth=2, label=f'Model C (J={J_target:.1f} target)')
ax1.axvline(x=T_surface, color='orange', linestyle=':', linewidth=1, label=f'Surface {T_surface}°C')
ax1.axvline(x=T_softening, color='red', linestyle='--', linewidth=1.5,
            label=f'Softening {T_softening}°C')
ax1.axvline(x=T_surface + delta_T_max, color='purple', linestyle='-.', linewidth=1.5,
            label=f'Surface+{delta_T_max}°C limit')
ax1.axhline(y=h_asphalt * 100, color='gray', linestyle='-', linewidth=1, alpha=0.5)

ax1.set_xlabel('Temperature [°C]', fontsize=10)
ax1.set_ylabel('Depth from surface [cm]', fontsize=10)
ax1.set_title('Temperature Profile through Asphalt Layer', fontsize=11)
ax1.legend(fontsize=7)
ax1.grid(True, alpha=0.3)
ax1.invert_yaxis()

# --- 子图2：发热功率与温升关系 ---
ax2 = axes[0, 1]
J_range = np.linspace(0.05, 2.0, 100)
P_range = J_range * V_work
delta_T_range = P_range * R_asphalt

ax2.plot(J_range, delta_T_range, 'b-', linewidth=2)
ax2.axvline(x=J_actual, color='red', linestyle='--', linewidth=2,
            label=f'Actual J={J_actual:.3f} A/m²')
ax2.axvline(x=J_target, color='green', linestyle='--', linewidth=2,
            label=f'Target J={J_target} A/m²')
ax2.axhline(y=delta_T_max, color='red', linestyle=':', linewidth=2,
            label=f'ΔT limit = {delta_T_max}°C')
ax2.axhline(y=T_softening - T_surface, color='orange', linestyle=':', linewidth=2,
            label=f'T_soft margin = {T_softening-T_surface}°C')
ax2.fill_between(J_range, 0, delta_T_max, alpha=0.1, color='green')
ax2.set_xlabel('Current Density J [A/m²]', fontsize=10)
ax2.set_ylabel('Temperature Rise ΔT [°C]', fontsize=10)
ax2.set_title('ΔT vs Current Density (Model A)', fontsize=11)
ax2.legend(fontsize=7)
ax2.grid(True, alpha=0.3)

# --- 子图3：热路模型示意 ---
ax3 = axes[1, 0]
# 热路图
ax3.set_xlim(0, 10)
ax3.set_ylim(0, 10)
ax3.axis('off')
ax3.set_title('Thermal Circuit Model', fontsize=11)

# 画热路
ax3.plot([3, 3], [8, 6], 'k-', linewidth=3)  # T_surface
ax3.plot([3, 3], [6, 3], 'r-', linewidth=4)  # R_asphalt
ax3.plot([3, 3], [3, 2], 'k-', linewidth=3)  # T_grating
# 分叉
ax3.plot([3, 1], [3, 1], 'b-', linewidth=2)  # R_waterstable
ax3.plot([1, 1], [1, 0.5], 'k-', linewidth=2)  # T_deep

ax3.text(3.5, 7, f'T_surface = {T_surface}°C', fontsize=10, verticalalignment='center')
ax3.text(3.5, 4.5, f'R_asphalt = {R_asphalt:.4f}\nK·m²/W', fontsize=9, color='red')
ax3.text(3.5, 2.5, f'T_grating = {T_grating:.1f}°C', fontsize=10, fontweight='bold', color='red')
ax3.text(3.5, 1.5, f'q_up = {P_heat:.1f} W/m²', fontsize=9)
ax3.text(0.2, 1, f'R_waterstable\n= {R_waterstable:.4f}', fontsize=9, color='blue')
ax3.text(0.2, 0.2, f'T_deep ≈ {T_surface}°C', fontsize=9)

ax3.text(4, 9, 'HEAT FLOW MODEL', fontsize=11, fontweight='bold')
infos = [
    f'P_heat = J × V = {P_heat:.1f} W/m²',
    f'Model A (conservative):',
    f'  ΔT = P × R_asphalt = {delta_T_A:.2f}°C',
    f'  T_grating = {T_grating_A:.1f}°C',
    f'Model B (bidirectional):',
    f'  ΔT = P × R_parallel = {delta_T_B:.2f}°C',
    f'  T_grating = {T_grating_B:.1f}°C',
]
for i, info in enumerate(infos):
    ax3.text(5.5, 9 - i*0.5, info, fontsize=8, family='monospace')

# --- 子图4：敏感性分析 ---
ax4 = axes[1, 1]
# 对沥青厚度和导热系数的敏感性
h_range = np.linspace(0.05, 0.30, 100) * 100  # cm
k_range_vals = np.linspace(0.5, 2.5, 5)

for k_val in k_range_vals:
    delta_T_h = P_heat_target * (h_range/100) / k_val
    ax4.plot(h_range, delta_T_h, linewidth=1.5, alpha=0.7,
             label=f'k={k_val:.1f} W/(m·K)')

ax4.axhline(y=delta_T_max, color='red', linestyle='--', linewidth=2, label=f'ΔT limit={delta_T_max}°C')
ax4.axvline(x=h_asphalt*100, color='gray', linestyle=':', linewidth=1.5, alpha=0.5,
            label=f'Design h={h_asphalt*100:.0f}cm')
ax4.set_xlabel('Asphalt Thickness [cm]', fontsize=10)
ax4.set_ylabel('Temperature Rise ΔT [°C]', fontsize=10)
ax4.set_title(f'Sensitivity: ΔT vs h for J={J_target} A/m²', fontsize=11)
ax4.legend(fontsize=7)
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('task3_thermal.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"\n图片已保存: task3_thermal.png")

# ============================================================
# 7. 输出汇总
# ============================================================
print(f"\n{'='*60}")
print(f"任务三 汇总结果")
print(f"{'='*60}")
print(f"  实际面电流密度 J_areal = {J_actual:.3f} A/m²")
print(f"  发热功率 P_heat = {P_heat:.2f} W/m²")
print(f"  沥青层热阻 R_asphalt = {R_asphalt:.4f} m²·K/W")
print(f"  附加温升 (模型A) ΔT = {delta_T_A:.2f} °C")
print(f"  格栅温度 (模型A) T_grating = {T_grating_A:.2f} °C")
print(f"  附加温升 (模型B) ΔT = {delta_T_B:.2f} °C")
print(f"  格栅温度 (模型B) T_grating = {T_grating_B:.2f} °C")
print(f"  沥青软化点 T_soft = {T_softening} °C")
print(f"  容许温升 ΔT_max = {delta_T_max} °C")
print(f"  综合判断: {'✓ 热安全通过' if thermal_pass else '✗ 热安全不通过'}")

# 保存结果
np.savez('task3_results.npz',
         P_heat=P_heat,
         delta_T_A=delta_T_A,
         delta_T_B=delta_T_B,
         T_grating_A=T_grating_A,
         T_grating_B=T_grating_B,
         T_surface=T_surface,
         thermal_pass=thermal_pass,
         check_softening=check_softening,
         check_delta_T=check_delta_T,
         J_actual=J_actual)

print(f"\n关键数据已保存至: task3_results.npz")
print(f"任务三脚本执行完毕。")
