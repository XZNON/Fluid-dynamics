"""Circulation controls lift. Pressure stays symmetric => zero drag (d'Alembert)."""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

OUT = Path(__file__).parent / 'images'
OUT.mkdir(exist_ok=True)

n, U, R = 400, 1.0, 1.0
x, y = np.meshgrid(np.linspace(-2.5, 2.5, n), np.linspace(-2, 2, n))
r2 = np.maximum(x**2 + y**2, 1e-12)

fig, axes = plt.subplots(1, 4, figsize=(16, 3.6))

for ax, Gam in zip(axes[:3], [0.0, 4*np.pi*U*R*0.5, 4*np.pi*U*R*1.0]):
    psi = U*y*(1 - R**2/r2) + (Gam/(4*np.pi))*np.log(r2/R**2)
    psi = np.where(r2 < R**2, np.nan, psi)
    ax.contour(x, y, psi, levels=np.linspace(-2.5, 2.5, 71),
               colors='#185FA5', linewidths=0.7, linestyles='solid')
    ax.add_patch(plt.Circle((0, 0), R, color='#3d3d3a', zorder=5))
    # stagnation points: sin(theta) = -Gamma / (4 pi U R)
    s = -Gam/(4*np.pi*U*R)
    if abs(s) <= 1:
        th = np.arcsin(s)
        for t in (np.pi - th, th):
            ax.plot(R*np.cos(t), R*np.sin(t), 'o', color='#E24B4A', ms=7, zorder=6)
    ax.set_title(f'$\\Gamma$ = {Gam:.2f}   (lift = $\\rho U \\Gamma$ = {Gam:.2f})', fontsize=10)
    ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])

# surface pressure coefficient, no circulation
th = np.linspace(0, 2*np.pi, 400)
cp = 1 - 4*np.sin(th)**2
ax = axes[3]
ax.plot(np.degrees(th), cp, color='#185FA5', lw=1.6)
ax.axhline(0, color='#b4b2a9', lw=0.8)
ax.fill_between(np.degrees(th), cp, 0, where=cp > 0, color='#F0997B', alpha=.45)
ax.fill_between(np.degrees(th), cp, 0, where=cp < 0, color='#85B7EB', alpha=.45)
ax.set_title('Surface $C_p$: front = back\n$\\Rightarrow$ drag is exactly zero', fontsize=10)
ax.set_xlabel('angle around cylinder (deg)'); ax.set_ylabel('$C_p$')
ax.set_xticks([0, 90, 180, 270, 360])

fig.tight_layout()
fig.savefig(OUT / 'cylinder_circulation.png', dpi=140)