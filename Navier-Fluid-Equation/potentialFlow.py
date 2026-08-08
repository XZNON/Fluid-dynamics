"""Potential flow: streamlines from analytic stream functions. No solver."""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

OUT = Path(__file__).parent / 'images'
OUT.mkdir(exist_ok=True)

# ---- grid -------------------------------------------------------------
n = 400
x, y = np.meshgrid(np.linspace(-3, 3, n), np.linspace(-2, 2, n))
r2 = x**2 + y**2

# ---- elementary stream functions (psi) --------------------------------
U = 1.0          # freestream speed
R = 1.0          # cylinder radius
Lam = 2.0        # source strength
Gam = 3.0        # vortex circulation

uniform = U * y
source = lambda X, Y, L: (L / (2*np.pi)) * np.arctan2(y - Y, x - X)
doublet = -(U * R**2) * y / np.where(r2 == 0, 1e-12, r2)
vortex = (Gam / (2*np.pi)) * 0.5 * np.log(np.where(r2 == 0, 1e-12, r2))

# ---- four canonical superpositions ------------------------------------
cases = [
    ("Uniform + source\n(half-body)",      uniform + source(0, 0, Lam),        None),
    ("Source + sink in uniform\n(Rankine oval)",
        uniform + source(-1, 0, Lam) + source(1, 0, -Lam),                     None),
    ("Uniform + doublet\n(cylinder, no lift)", uniform + doublet,               R),
    ("Uniform + doublet + vortex\n(cylinder with lift)",
        uniform + doublet + vortex,                                            R),
]

fig, axes = plt.subplots(2, 2, figsize=(13, 8))
for ax, (title, psi, rad) in zip(axes.ravel(), cases):
    if rad:                                   # hide the interior of the body
        psi = np.where(r2 < rad**2, np.nan, psi)
    ax.contour(x, y, psi, levels=np.linspace(-3, 3, 61),
               colors='#185FA5', linewidths=0.8, linestyles='solid')
    if rad:
        ax.add_patch(plt.Circle((0, 0), rad, color='#3d3d3a', zorder=5))
    ax.set_title(title, fontsize=10)
    ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color('#b4b2a9')

fig.suptitle('Potential flow = adding stream functions together', fontsize=13)
fig.tight_layout()
fig.savefig(OUT / 'potential_flow_basics.png', dpi=140)