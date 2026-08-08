"""Potential flow past arbitrary polygons, via the source panel method."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
from pathlib import Path as FilePath
from panels import solve_panels, field, polygon, panel_velocity

OUT = FilePath(__file__).parent / 'images'
OUT.mkdir(exist_ok=True)

def refine(verts, per_side=40):
    """Split each polygon edge into many small panels."""
    p2 = np.roll(verts, -1, axis=0)
    s = np.linspace(0, 1, per_side, endpoint=False)[:, None]
    return np.concatenate([v1 + s*(v2 - v1) for v1, v2 in zip(verts, p2)])

def surface_cp(sol):
    P, t, n = sol['cp'], sol['t'], sol['n']
    Pout = P + 1e-7*n
    V = np.broadcast_to(sol['Uinf'], Pout.shape).copy()
    for j in range(len(sol['L'])):
        V += panel_velocity(Pout, sol['p1'][j], sol['t'][j],
                            sol['nn'][j], sol['L'][j], sol['sigma'][j])
    Vt = V[:, 0]*t[:, 0] + V[:, 1]*t[:, 1]
    return 1 - Vt**2

shapes = {
    'Triangle':      polygon(3,  1.0, np.pi/2),
    'Square':        polygon(4,  1.0, np.pi/4),
    'Diamond':       polygon(4,  1.0, 0.0),
    'Hexagon':       polygon(6,  1.0, 0.0),
}

x, y = np.meshgrid(np.linspace(-2.6, 2.6, 320), np.linspace(-1.9, 1.9, 260))
fig, axes = plt.subplots(2, 2, figsize=(13, 8))

for ax, (name, verts) in zip(axes.ravel(), shapes.items()):
    fine = refine(verts, 60)
    sol = solve_panels(fine, U=1.0)
    u, v = field(sol, x, y)

    inside = Path(verts).contains_points(np.stack([x.ravel(), y.ravel()], 1)).reshape(x.shape)
    u = np.where(inside, np.nan, u); v = np.where(inside, np.nan, v)
    speed = np.hypot(u, v)

    ax.streamplot(x, y, u, v, density=1.6, linewidth=0.7,
                  color=speed, cmap='viridis', arrowsize=0.7)
    ax.fill(verts[:, 0], verts[:, 1], color='#3d3d3a', zorder=5)

    cp = surface_cp(sol)
    cd = -np.sum(cp * sol['n'][:, 0] * sol['L']) / 2.0
    ax.set_title(f'{name}   —   $C_D$ = {cd:+.4f}   |   max surface speed = {np.sqrt(np.abs(1-cp.min())):.1f}$U$',
                 fontsize=10)
    ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])

fig.suptitle('Source panel method: any polygon you like — and drag is still zero', fontsize=13)
fig.tight_layout()
fig.savefig(OUT / 'polygon_potential_flow.png', dpi=140)
print('done')