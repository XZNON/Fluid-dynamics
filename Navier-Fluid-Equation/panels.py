"""Source panel method: potential flow past ANY closed polygon."""
import numpy as np

def solve_panels(verts, U=1.0, alpha=0.0):
    """verts: (N,2) counter-clockwise, unclosed. Returns panel geometry + strengths."""
    p1 = verts
    p2 = np.roll(verts, -1, axis=0)
    d = p2 - p1
    L = np.hypot(d[:, 0], d[:, 1])
    t = d / L[:, None]                     # tangent
    nn = np.stack([-t[:, 1], t[:, 0]], 1)  # left normal (right-handed with t)
    n = -nn                                # outward normal for CCW ordering
    cp = 0.5 * (p1 + p2)                   # control points
    Uinf = U * np.array([np.cos(alpha), np.sin(alpha)])

    N = len(verts)
    A = np.empty((N, N))
    for j in range(N):
        v = panel_velocity(cp, p1[j], t[j], nn[j], L[j], 1.0)
        A[:, j] = v[:, 0]*n[:, 0] + v[:, 1]*n[:, 1]
    np.fill_diagonal(A, 0.5)               # a panel induces sigma/2 on itself
    b = -(Uinf[0]*n[:, 0] + Uinf[1]*n[:, 1])
    sigma = np.linalg.solve(A, b)
    return dict(p1=p1, t=t, nn=nn, L=L, sigma=sigma, Uinf=Uinf, cp=cp, n=n)


def panel_velocity(P, p1, t, nn, L, sigma):
    """Velocity induced at points P by one constant-strength source panel."""
    rel = P - p1
    xl = rel[..., 0]*t[0] + rel[..., 1]*t[1]     # local coords
    yl = rel[..., 0]*nn[0] + rel[..., 1]*nn[1]
    r1sq = np.maximum(xl**2 + yl**2, 1e-14)
    r2sq = np.maximum((xl - L)**2 + yl**2, 1e-14)
    u = (sigma/(4*np.pi)) * np.log(r1sq/r2sq)
    v = (sigma/(2*np.pi)) * (np.arctan2(yl, xl - L) - np.arctan2(yl, xl))
    return np.stack([u*t[0] + v*nn[0], u*t[1] + v*nn[1]], axis=-1)


def field(sol, X, Y):
    """Total velocity on a grid."""
    P = np.stack([X, Y], -1)
    V = np.broadcast_to(sol['Uinf'], P.shape).copy()
    for j in range(len(sol['L'])):
        V += panel_velocity(P, sol['p1'][j], sol['t'][j],
                            sol['nn'][j], sol['L'][j], sol['sigma'][j])
    return V[..., 0], V[..., 1]


def polygon(nsides, radius=1.0, rotate=0.0):
    a = np.linspace(0, 2*np.pi, nsides, endpoint=False) + rotate
    return np.stack([radius*np.cos(a), radius*np.sin(a)], 1)