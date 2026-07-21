#!/usr/bin/env python3
"""
New Design Verification: 2D Electric Field FDM
Parameters: d=0.30m, V=24V, w=0.015m, domain 0.60m×0.20m
Method: Sparse direct solver (scipy.sparse.linalg.spsolve) — robust, no convergence issues
"""
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt

# ===== New Design Parameters =====
V_anode = 24.0
d = 0.30
Lx = 2 * d              # 0.60 m
Lz = 0.20
sigma_med = 0.05
w_electrode = 0.015

# ===== FDM Grid =====
Nx, Nz = 161, 41         # coarser for faster sparse solve
dx, dz = Lx/(Nx-1), Lz/(Nz-1)
N = Nx * Nz              # total unknowns

x = np.linspace(0, Lx, Nx)
z = np.linspace(0, Lz, Nz)
X, Z = np.meshgrid(x, z, indexing='ij')

print(f"Grid: {Nx}×{Nz}={N}, dx={dx:.4f}m, dz={dz:.4f}m")

# ===== Map (i,j) -> k = i + j*Nx =====
def idx(i, j):
    return i + j * Nx

# ===== Identify Dirichlet nodes =====
is_dir = np.zeros((Nx, Nz), dtype=bool)
V_dir = np.zeros((Nx, Nz))

anode_left_idx = int(w_electrode / dx)
cathode_left_idx = int((d - w_electrode) / dx)
cathode_right_idx = int((d + w_electrode) / dx)
anode_right_idx = int((Lx - w_electrode) / dx)

for i in range(Nx):
    xi = i * dx
    if xi <= w_electrode:
        is_dir[i, 0] = True; V_dir[i, 0] = V_anode
    elif abs(xi - d) <= w_electrode:
        is_dir[i, 0] = True; V_dir[i, 0] = 0.0
    elif xi >= Lx - w_electrode:
        is_dir[i, 0] = True; V_dir[i, 0] = V_anode

n_dir = is_dir.sum()
print(f"Dirichlet nodes: {n_dir}, Free nodes: {N - n_dir}")

# ===== Assemble Sparse System A·x = b =====
# For each free node (i,j): standard 5-point Laplacian
# (V_left+V_right)/dx² + (V_up+V_down)/dz² - 2(1/dx²+1/dz²)*V_ij = 0
# Neumann boundaries: ghost point = interior point => effective stencil modified

# Map free nodes to equation indices
free_map = -np.ones((Nx, Nz), dtype=int)
free_nodes = []
for i in range(Nx):
    for j in range(Nz):
        if not is_dir[i, j]:
            free_map[i, j] = len(free_nodes)
            free_nodes.append((i, j))

N_free = len(free_nodes)
print(f"Assembling {N_free}×{N_free} sparse system...")

# Build in COO format
row_vals, col_vals, data_vals = [], [], []
rhs = np.zeros(N_free)

inv_dx2 = 1.0 / dx**2
inv_dz2 = 1.0 / dz**2

for (i, j) in free_nodes:
    k = free_map[i, j]
    diag = 0.0

    # Left neighbor
    if i == 0:
        # Neumann ∂V/∂x=0 => V_ghost = V[1,j], so V_left = V[1,j]
        # (V_right - V)/dx - (V - V_ghost)/dx = (V[1]-V)/dx² - 0 = ...
        # Actually: ∂²V/∂x² ≈ (V[1]-V[0])/dx²  (one-sided, using ∂V/∂x=0 at boundary)
        if is_dir[1, j]:
            rhs[k] -= inv_dx2 * V_dir[1, j]
        else:
            col_vals.append(k); row_vals.append(k); data_vals.append(-inv_dx2)
            # will add V[1] contribution
            # Actually this is getting complex. Let me handle Neumann properly.
            # At i=0 with Neumann: V[-1] = V[1] => (V[-1]+V[1]-2V[0])/dx² = (2V[1]-2V[0])/dx²
            # So: 2*inv_dx2 * V[1,j] - 2*inv_dx2 * V[0,j] = 0
            pass
    elif is_dir[i-1, j]:
        rhs[k] -= inv_dx2 * V_dir[i-1, j]
    else:
        col_vals.append(k)
        row_vals.append(k)
        data_vals.append(inv_dx2)
        # We'll add V_left via its own equation. Actually this is wrong.
        # Let me use a simpler approach - just build the Laplacian directly.
        pass

print("Switching to simpler assembly approach...")

# Cleaner approach: build A matrix row by row
A_rows, A_cols, A_data = [], [], []
b = np.zeros(N_free)

for (i, j) in free_nodes:
    k = free_map[i, j]
    A_rows.append(k); A_cols.append(k); A_data.append(0.0)  # placeholder for diagonal
    diag_pos = len(A_data) - 1  # index of diagonal entry
    diag_val = 0.0

    # Process 4 neighbors
    neighbors = []
    # Left
    if i == 0:
        # Neumann: V[-1,j] = V[1,j] (ghost = interior by symmetry)
        ni, nj = 1, j
    else:
        ni, nj = i-1, j
    neighbors.append((ni, nj))

    # Right
    if i == Nx - 1:
        ni, nj = Nx-2, j
    else:
        ni, nj = i+1, j
    neighbors.append((ni, nj))

    # Up (z-1)
    if j == 0:
        if is_dir[i, 0]:
            ni, nj = i, 0  # Dirichlet, will add to RHS
        else:
            ni, nj = i, 1  # Neumann ghost = below
    else:
        ni, nj = i, j-1
    neighbors.append((ni, nj))

    # Down (z+1)
    if j == Nz - 1:
        ni, nj = i, Nz-2  # Neumann
    else:
        ni, nj = i, j+1
    neighbors.append((ni, nj))

    for idx_n, (ni, nj) in enumerate(neighbors):
        if idx_n < 2:  # x-direction
            coeff = inv_dx2
        else:           # z-direction
            coeff = inv_dz2

        if is_dir[ni, nj]:
            b[k] -= coeff * V_dir[ni, nj]
        else:
            nk = free_map[ni, nj]
            A_rows.append(k); A_cols.append(nk); A_data.append(coeff)
        diag_val -= coeff  # ALL neighbors contribute to diagonal

    A_data[diag_pos] = diag_val

A = sparse.coo_matrix((A_data, (A_rows, A_cols)), shape=(N_free, N_free)).tocsr()
print(f"A: {A.shape}, nnz={A.nnz}")

# ===== Solve =====
print("Solving...")
V_free = spsolve(A, b)
print(f"Solved. V range: [{V_free.min():.2f}, {V_free.max():.2f}] V")

# ===== Reconstruct full V =====
V = np.zeros((Nx, Nz))
for k, (i, j) in enumerate(free_nodes):
    V[i, j] = V_free[k]
V[is_dir] = V_dir[is_dir]

# ===== E-field & J =====
Ex = np.zeros((Nx, Nz))
Ez = np.zeros((Nx, Nz))
for i in range(Nx):
    for j in range(Nz):
        if i == 0: Ex[i,j] = -(V[1,j]-V[0,j])/dx
        elif i == Nx-1: Ex[i,j] = -(V[Nx-1,j]-V[Nx-2,j])/dx
        else: Ex[i,j] = -(V[i+1,j]-V[i-1,j])/(2*dx)
        if j == 0: Ez[i,j] = -(V[i,1]-V[i,0])/dz
        elif j == Nz-1: Ez[i,j] = -(V[i,Nz-1]-V[i,Nz-2])/dz
        else: Ez[i,j] = -(V[i,j+1]-V[i,j-1])/(2*dz)

E_mag = np.sqrt(Ex**2 + Ez**2)
Jx, Jz = sigma_med*Ex, sigma_med*Ez
J_mag = sigma_med*E_mag

# ===== Integrated current =====
I_left  = abs(np.sum(Jz[:anode_left_idx+1, 0]) * dx)
I_right = abs(np.sum(Jz[anode_right_idx:, 0]) * dx)
I_total = I_left + I_right
J_areal = I_total / Lx

# ===== Statistics =====
gap = ~is_dir
E_avg_domain = np.mean(E_mag[gap])
gap_top = np.array([not is_dir[i,0] for i in range(Nx)])
E_gap_vals = E_mag[gap_top, 0]
E_avg_top = np.mean(E_gap_vals) if len(E_gap_vals) > 0 else V_anode/d
E_gap_cv = np.std(E_gap_vals)/np.mean(E_gap_vals) if np.mean(E_gap_vals)>0 else 0
J_avg_domain = sigma_med * E_avg_domain

P_inst = J_areal * V_anode
P_avg = P_inst / 4  # 1:3 duty

# ===== Print Results =====
print(f"\n{'='*55}")
print(f"NEW DESIGN Task 1 — Electric Field Verification")
print(f"{'='*55}")
print(f"  V/d (uniform approx)         = {V_anode/d:8.1f} V/m")
print(f"  Avg |E| (domain)             = {E_avg_domain:8.1f} V/m")
print(f"  Avg |E| (top gap)            = {E_avg_top:8.1f} V/m")
print(f"  E-field CV (top gap)         = {E_gap_cv:8.3f}")
print(f"  Avg |J| (domain)             = {J_avg_domain:8.3f} A/m²")
print(f"  Areal current density        = {J_areal:8.3f} A/m²")
print(f"  Total current per meter      = {I_total:8.3f} A/m")
print(f"  Instantaneous power          = {P_inst:8.2f} W/m²")
print(f"  Time-avg power (1:3 duty)    = {P_avg:8.2f} W/m²")
print(f"  Check: J_areal={J_areal:.3f} A/m², time-avg={J_areal/4:.3f} A/m²")

# ===== Plot (English labels for font compat) =====
fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
fig.suptitle(f'Task 1: E-field Verification | V={V_anode}V d={d}m J={J_areal:.3f}A/m²',
             fontsize=13, fontweight='bold')

ax = axes[0]
cf = ax.contourf(X, Z, V, levels=20, cmap='RdYlBu_r', extend='both')
ax.contour(X, Z, V, levels=20, colors='k', linewidths=0.2)
for xi, h, c in [(0, w_electrode, 'red'), (d-w_electrode, 2*w_electrode, 'blue'),
                  (Lx-w_electrode, w_electrode, 'red')]:
    ax.add_patch(plt.Rectangle((xi, -0.004), h, 0.008, facecolor=c,
                                edgecolor='darkred' if c == 'red' else 'darkblue'))
ax.set_xlabel('x [m]'); ax.set_ylabel('z [m]')
ax.set_title('Potential V(x,z) [V]'); ax.invert_yaxis(); ax.set_aspect('equal')
plt.colorbar(cf, ax=ax, shrink=0.8, label='V [V]')

ax = axes[1]
cf = ax.contourf(X, Z, E_mag, levels=25, cmap='plasma')
skip = 8
sx = slice(None, None, skip); sy = slice(None, None, 2)
ax.quiver(X[sx, sy], Z[sx, sy], Ex[sx, sy], Ez[sx, sy],
          E_mag[sx, sy], cmap='Greys', alpha=0.6, scale=400, width=0.005)
for xi, h, c in [(0, w_electrode, 'red'), (d-w_electrode, 2*w_electrode, 'blue'),
                  (Lx-w_electrode, w_electrode, 'red')]:
    ax.add_patch(plt.Rectangle((xi, -0.004), h, 0.008, facecolor=c,
                                edgecolor='darkred' if c == 'red' else 'darkblue'))
ax.set_xlabel('x [m]'); ax.set_ylabel('z [m]')
ax.set_title('|E| [V/m] & vectors'); ax.invert_yaxis(); ax.set_aspect('equal')
plt.colorbar(cf, ax=ax, shrink=0.8, label='|E| [V/m]')

ax = axes[2]
ax.plot(x, E_mag[:, 0], 'b-', lw=1.5, label='|E| at z=0')
ax.plot(x, J_mag[:, 0], 'r-', lw=1.5, label='|J| at z=0')
for xi, h, c in [(0, w_electrode, 'red'), (d-w_electrode, 2*w_electrode, 'blue'),
                  (Lx-w_electrode, w_electrode, 'red')]:
    ax.axvspan(xi, xi+h, alpha=0.12, color=c)
ax.axhline(y=J_areal, color='g', ls=':', lw=1.5, label=f'J_areal={J_areal:.2f} A/m²')
ax.set_xlabel('x [m]'); ax.set_ylabel('Magnitude')
ax.set_title('Top surface (z=0) distribution'); ax.legend(fontsize=7); ax.grid(alpha=0.3)

plt.tight_layout(); plt.savefig('verify_task1.png', dpi=150, bbox_inches='tight'); plt.close()
print("  Figure saved: verify_task1.png")

np.savez('verify_task1.npz', E_avg_domain=E_avg_domain, E_avg_top=E_avg_top,
         J_areal=J_areal, I_total=I_total, V_d=V_anode/d,
         P_inst=P_inst, P_avg=P_avg, J_avg_domain=J_avg_domain)
print("  Data saved: verify_task1.npz")
