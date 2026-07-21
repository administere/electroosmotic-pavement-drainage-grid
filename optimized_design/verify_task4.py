#!/usr/bin/env python3
"""
新方案验证：层间抗剪安全系数
=====================================
参数变更：z=0.14m, c=85kPa, φ=30°
"""
import numpy as np
import matplotlib.pyplot as plt

# ===== 新设计参数 =====
p_tire = 0.7              # MPa
delta = 0.15              # m
z_grid = 0.14             # ← 0.12→0.14
c_cohesion = 85.0         # ← 40→85 kPa (3D芯体机械咬合)
phi_friction = 30.0       # ← 25→30° (芯体凸点)
phi_rad = np.radians(phi_friction)
nu_asphalt = 0.35
FS_required = 1.5

a = z_grid / delta  # 0.14/0.15 = 0.933

# ===== 弹性半空间应力 =====
sigma_z = p_tire * (1.0 - a**3 / (1.0+a**2)**1.5)
sigma_r = p_tire * ((1+2*nu_asphalt)/2.0
                     - (1+nu_asphalt)*a/np.sqrt(1+a**2)
                     + a**3/(2*(1+a**2)**1.5))

sigmas = sorted([sigma_z, sigma_r, sigma_r], reverse=True)
tau_max = (sigmas[0] - sigmas[2]) / 2.0  # MPa
tau_max_kPa = tau_max * 1000

sigma_n_kPa = sigma_z * 1000
tau_allow = c_cohesion + sigma_n_kPa * np.tan(phi_rad)
FS = tau_allow / tau_max_kPa

print(f"{'='*50}")
print(f"新方案 Task4 抗剪验算")
print(f"{'='*50}")
print(f"  a = z/δ = {z_grid}/{delta} = {a:.3f}")
print(f"  σ_z = {sigma_z*1000:.1f} kPa ({sigma_z/p_tire*100:.1f}% of p)")
print(f"  σ_r = σ_θ = {sigma_r*1000:.1f} kPa")
print(f"  τ_max = {tau_max_kPa:.1f} kPa")
print(f"  σ_n = {sigma_n_kPa:.1f} kPa")
print(f"  τ_allow = {c_cohesion} + {sigma_n_kPa:.1f}·tan({phi_friction}°)")
print(f"          = {c_cohesion} + {sigma_n_kPa*np.tan(phi_rad):.1f}")
print(f"          = {tau_allow:.1f} kPa")
print(f"  FS = {tau_allow:.1f} / {tau_max_kPa:.1f} = {FS:.3f}")
print(f"  FS_req = {FS_required}")
print(f"  判断: {'✓ 通过' if FS>=FS_required else '✗ 不通过'}")
if FS >= FS_required:
    print(f"  裕度: +{(FS/FS_required-1)*100:.1f}%")

# ===== 参数敏感性（等高线） =====
c_range = np.linspace(30, 120, 50)
phi_range = np.linspace(20, 40, 50)
C, PHI = np.meshgrid(c_range, phi_range)
FS_grid = np.zeros_like(C)
for i in range(len(phi_range)):
    for j in range(len(c_range)):
        tallow = C[i,j] + sigma_n_kPa * np.tan(np.radians(PHI[i,j]))
        FS_grid[i,j] = tallow / tau_max_kPa

# ===== 与原方案对比 =====
# 原方案参数
z_old = 0.12
a_old = z_old / delta
sz_old = p_tire * (1 - a_old**3/(1+a_old**2)**1.5)
sr_old = p_tire * ((1+2*nu_asphalt)/2 - (1+nu_asphalt)*a_old/np.sqrt(1+a_old**2)
                    + a_old**3/(2*(1+a_old**2)**1.5))
tmax_old = abs(sz_old - sr_old)/2 * 1000
tallow_old = 40 + sz_old*1000*np.tan(np.radians(25))
FS_old = tallow_old / tmax_old

print(f"\n  原方案对比: FS_old = {FS_old:.3f} (z={z_old}m, c=40kPa, φ=25°)")
print(f"  提升幅度: +{(FS/FS_old-1)*100:.0f}%")

# ===== 画图 =====
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle(f'新方案抗剪验证 | z={z_grid*100:.0f}cm c={c_cohesion}kPa φ={phi_friction}° FS={FS:.2f}', fontsize=13, fontweight='bold')

# 莫尔圆
ax=axes[0,0]
sc = (sigmas[0]+sigmas[2])/2*1000; sr = (sigmas[0]-sigmas[2])/2*1000
th = np.linspace(0, 2*np.pi, 200)
ax.plot(sc+sr*np.cos(th), np.abs(sr*np.sin(th)), 'b-', lw=2, label='Mohr Circle')
sigma_line = np.linspace(0, max(sigma_n_kPa*1.8, tau_allow*1.5), 100)
ax.plot(sigma_line, c_cohesion + sigma_line*np.tan(phi_rad), 'r-', lw=2.5, label='Failure envelope')
ax.plot([sigma_n_kPa, sigma_n_kPa], [0, tau_allow], 'g--', lw=1, alpha=0.7)
ax.plot(sigma_n_kPa, tau_allow, 'gs', ms=10, label=f'τ_allow={tau_allow:.0f} kPa')
ax.plot(sc, sr, 'ro', ms=10, label=f'τ_max={tau_max_kPa:.0f} kPa')
ax.set_xlabel('σ [kPa]'); ax.set_ylabel('τ [kPa]')
ax.set_title(f'Mohr Circle (FS={FS:.2f})'); ax.legend(fontsize=7); ax.grid(alpha=0.3)
ax.set_xlim(0, max(sc+sr*1.5, sigma_n_kPa*1.5)); ax.set_ylim(0, sr*1.8)
ax.set_aspect('equal')

# FS vs depth
ax=axes[0,1]
z_range = np.linspace(0.02, 0.40, 200)
FS_z = np.zeros_like(z_range)
for i, zi in enumerate(z_range):
    ai = zi/delta
    sz = p_tire*(1-ai**3/(1+ai**2)**1.5)
    sr_ = p_tire*((1+2*nu_asphalt)/2-(1+nu_asphalt)*ai/np.sqrt(1+ai**2)+ai**3/(2*(1+ai**2)**1.5))
    tmax = abs(sz-sr_)/2*1000
    tallow = c_cohesion + sz*1000*np.tan(phi_rad)
    FS_z[i] = tallow/tmax if tmax>0 else 99
ax.plot(FS_z, z_range*100, 'b-', lw=2.5)
ax.axhline(y=z_grid*100, color='gray', ls='--', lw=1.5)
ax.axvline(x=FS_required, color='red', ls='--', lw=2, label=f'FS_req={FS_required}')
ax.axvline(x=1.0, color='orange', ls=':', lw=1)
ax.plot(FS, z_grid*100, 'ro', ms=12)
ax.annotate(f'FS={FS:.2f}', xy=(FS, z_grid*100), xytext=(FS+0.2, z_grid*100+2),
            arrowprops=dict(arrowstyle='->',color='red'), fontsize=10, fontweight='bold',color='red')
ax.fill_betweenx(z_range*100, FS_required, 5, alpha=0.1, color='green', label=f'Safe (FS≥{FS_required})')
ax.fill_betweenx(z_range*100, 1, FS_required, alpha=0.08, color='yellow', label='Marginal')
ax.fill_betweenx(z_range*100, 0, 1, alpha=0.08, color='red', label='Fail')
ax.set_xlabel('Safety Factor FS'); ax.set_ylabel('Depth [cm]')
ax.set_title('FS vs Depth'); ax.legend(fontsize=7); ax.grid(alpha=0.3); ax.invert_yaxis()
ax.set_xlim(0, 5)

# FS 等高线 (c vs φ)
ax=axes[1,0]
cf=ax.contourf(C, PHI, FS_grid, levels=np.linspace(0.8, 3.0, 23), cmap='RdYlGn', extend='both')
cs=ax.contour(C, PHI, FS_grid, levels=[1.0, 1.3, 1.5, 2.0, 2.5], colors='k', linewidths=[1,1,2,1,1])
ax.clabel(cs, inline=True, fontsize=8, fmt='FS=%.1f')
ax.plot(c_cohesion, phi_friction, 'ro', ms=12, markeredgecolor='white', markeredgewidth=2)
ax.annotate(f'NEW\nc={c_cohesion},φ={phi_friction}°', xy=(c_cohesion, phi_friction),
            xytext=(c_cohesion+15, phi_friction-2), arrowprops=dict(arrowstyle='->',color='red'),
            fontsize=9, fontweight='bold', color='red')
ax.plot(40, 25, 'bo', ms=10, markeredgecolor='white', markeredgewidth=2)
ax.annotate(f'OLD\nc=40,φ=25°', xy=(40, 25), xytext=(25, 22),
            arrowprops=dict(arrowstyle='->',color='blue'), fontsize=8, color='blue')
ax.set_xlabel('Cohesion c [kPa]'); ax.set_ylabel('Friction Angle φ [°]')
ax.set_title('FS Contour (c vs φ)'); plt.colorbar(cf, ax=ax, shrink=0.8, label='FS')

# 新旧对比柱状图
ax=axes[1,1]
categories = ['τ_max\n[kPa]', 'σ_n\n[kPa]', 'τ_allow\n[kPa]', 'FS\n[−]']
old_vals = [tmax_old, sz_old*1000, tallow_old, FS_old]
new_vals = [tau_max_kPa, sigma_n_kPa, tau_allow, FS]
xpos = np.arange(4); width = 0.35
bars1 = ax.bar(xpos-width/2, old_vals, width, color='#ff6b6b', edgecolor='black', label='Old (z=12,c=40,φ=25)')
bars2 = ax.bar(xpos+width/2, new_vals, width, color='#44bd32', edgecolor='black', label='New (z=14,c=85,φ=30)')
ax.axhline(y=FS_required, color='red', ls='--', lw=1.5, alpha=0.7)
ax.text(3.4, FS_required+0.05, f'FS_req={FS_required}', fontsize=8, color='red')
for b1,b2,ov,nv in zip(bars1, bars2, old_vals, new_vals):
    ax.text(b1.get_x()+b1.get_width()/2, b1.get_height()+2, f'{ov:.0f}' if ov>2 else f'{ov:.2f}',
            ha='center', fontsize=8, color='darkred')
    ax.text(b2.get_x()+b2.get_width()/2, b2.get_height()+2, f'{nv:.0f}' if nv>2 else f'{nv:.2f}',
            ha='center', fontsize=8, fontweight='bold', color='darkgreen')
ax.set_xticks(xpos); ax.set_xticklabels(categories, fontsize=9)
ax.set_title('Old vs New Comparison'); ax.legend(fontsize=8); ax.grid(alpha=0.3, axis='y')

plt.tight_layout(); plt.savefig('verify_task4.png', dpi=150, bbox_inches='tight'); plt.close()
print("  图片: verify_task4.png")

np.savez('verify_task4.npz', FS=FS, tau_max_kPa=tau_max_kPa, tau_allow=tau_allow,
         sigma_n_kPa=sigma_n_kPa, c_cohesion=c_cohesion, phi=phi_friction,
         FS_old=FS_old, shear_pass=FS>=FS_required)
print("  数据: verify_task4.npz\n")
