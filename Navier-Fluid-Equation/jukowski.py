"""Joukowski airfoil: same cylinder solution, conformally mapped. Still analytic."""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

OUT = Path(__file__).parent / 'images'
OUT.mkdir(exist_ok=True)

U = 1.0
mu = -0.09 + 0.09j                 # circle centre: shifts left => thickness, up => camber
R0 = abs(1 - mu)                   # circle must pass through zeta = 1 (the trailing edge)
beta = np.arcsin(mu.imag / R0)

# polar grid outside the circle in the zeta plane
r = np.geomspace(R0, 14*R0, 500)[:, None]
t = np.linspace(0, 2*np.pi, 601)[None, :]
zeta = mu + r*np.exp(1j*t)

fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
for ax, alpha_deg in zip(axes, [0, 6, 12]):
    a = np.radians(alpha_deg)
    Gam = 4*np.pi*U*R0*np.sin(a + beta)          # Kutta condition
    d = zeta - mu
    w = U*(d*np.exp(-1j*a) + R0**2*np.exp(1j*a)/d) + (1j*Gam/(2*np.pi))*np.log(d)
    psi = w.imag

    z = zeta + 1/zeta                            # Joukowski map
    ax.contour(z.real, z.imag, psi, levels=np.linspace(-3, 3, 81),
               colors='#185FA5', linewidths=0.6, linestyles='solid')
    foil = mu + R0*np.exp(1j*np.linspace(0, 2*np.pi, 400))
    foil = foil + 1/foil
    ax.fill(foil.real, foil.imag, color='#3d3d3a', zorder=5)
    ax.set_xlim(-3, 3); ax.set_ylim(-1.6, 1.6)
    ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f'$\\alpha$ = {alpha_deg}$^\\circ$    $\\Gamma$ = {Gam:.2f}    '
                 f'$C_L$ = {2*Gam/(U*4):.2f}', fontsize=10)

fig.suptitle('Joukowski airfoil — lift grows linearly with angle of attack, forever', fontsize=12)
fig.tight_layout()
fig.savefig(OUT / 'joukowski_airfoil.png', dpi=140)