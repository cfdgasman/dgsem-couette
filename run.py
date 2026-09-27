"""DGSEM on curvilinear elements: circular Couette flow convergence study, spin-up animation, figures."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from scipy.sparse import identity
from scipy.sparse.linalg import splu

from dgsem import Couette, Laplacian, Mesh, inner_torque, l2_error, solve_steady, wall_data

FLOW = Couette()


def triangulation(mesh):
    n1 = mesh.N + 1
    tris = []
    for e in range(mesh.K):
        base = e * n1 * n1
        for i in range(mesh.N):
            for j in range(mesh.N):
                a, b = base + i * n1 + j, base + (i + 1) * n1 + j
                c, d = base + (i + 1) * n1 + j + 1, base + i * n1 + j + 1
                tris += [(a, b, c), (a, c, d)]
    return mtri.Triangulation(mesh.x.ravel(), mesh.y.ravel(), np.array(tris))


def u_theta(mesh, u):
    r = np.hypot(mesh.x, mesh.y)
    return (-u[..., 0] * mesh.y + u[..., 1] * mesh.x) / r


def draw_elements(ax, mesh, **kw):
    for e in range(mesh.K):
        for xs, ys in ((mesh.x[e, 0], mesh.y[e, 0]), (mesh.x[e, -1], mesh.y[e, -1]),
                       (mesh.x[e, :, 0], mesh.y[e, :, 0]), (mesh.x[e, :, -1], mesh.y[e, :, -1])):
            ax.plot(xs, ys, **kw)


def p_study():
    rows = []
    for N in range(1, 13):
        row = [N]
        for curved in (True, False):
            m = Mesh(N, 2, 8, curved=curved)
            u, op, g = solve_steady(m, FLOW)
            row += [l2_error(m, FLOW, u), abs(inner_torque(m, FLOW, u, op, g) / FLOW.torque() - 1)]
        rows.append(row)
    print("p-refinement on a 2 x 8 mesh")
    print("| N | curved L2 | curved torque err | straight L2 | straight torque err |\n|---|---|---|---|---|")
    for N, a, b, c, d in rows:
        print(f"| {N} | {a:.2e} | {b:.2e} | {c:.2e} | {d:.2e} |")
    return np.array(rows)


def h_study():
    ks = [1, 2, 4, 8]
    out = {}
    print("\nh-refinement (Kr x Kt = k x 4k), observed L2 orders between the two finest meshes")
    for curved in (True, False):
        for N in (1, 2, 3, 4):
            e = []
            for k in ks:
                m = Mesh(N, k, 4 * k, curved=curved)
                u, _, _ = solve_steady(m, FLOW)
                e.append(l2_error(m, FLOW, u))
            out[(curved, N)] = e
            print(f"{'curved  ' if curved else 'straight'} N={N}: order {np.log2(e[-2] / e[-1]):.2f}")
    return ks, out


def spin_up(N=6, Kr=3, Kt=12, dt=2e-3, t_end=0.6, every=5):
    """Impulsively started inner cylinder: du/dt = nu lap(u), backward Euler, nu = 1."""
    m = Mesh(N, Kr, Kt)
    op = Laplacian(m)
    A = op.matrix()
    g_in, g_out = wall_data(m, FLOW)
    c = op.affine(g_in, g_out)
    lu = splu((identity(m.ndof) - dt * A).tocsc())
    u = np.zeros((m.ndof, 2))
    frames = [(0.0, u.reshape(m.shape + (2,)).copy())]
    for k in range(1, int(round(t_end / dt)) + 1):
        u = lu.solve(u + dt * c)
        if k % every == 0:
            frames.append((k * dt, u.reshape(m.shape + (2,)).copy()))
    return m, frames


def main():
    rows = p_study()
    ks, hres = h_study()

    # ---- figure: mesh and solution, curved vs straight
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.8))
    for ax, curved in zip(axes, (True, False)):
        m = Mesh(6, 2, 8, curved=curved)
        u, op, g = solve_steady(m, FLOW)
        tri = triangulation(m)
        tc = ax.tripcolor(tri, u_theta(m, u).ravel(), shading="gouraud", cmap="viridis", vmin=0, vmax=1)
        draw_elements(ax, m, color="w", lw=0.8)
        th = np.linspace(0, 2 * np.pi, 400)
        for r in (FLOW.r1, FLOW.r2):
            ax.plot(r * np.cos(th), r * np.sin(th), "r--", lw=0.8)
        ax.set(aspect="equal", xticks=[], yticks=[],
               title=f"{'Curved (isoparametric)' if curved else 'Straight-sided'} elements, N = 6\n"
                     f"L2 error {l2_error(m, FLOW, u):.1e}")
    fig.colorbar(tc, ax=axes, label="u_θ", shrink=0.8)
    fig.savefig("docs/mesh_solution.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    # ---- figure: convergence
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))
    N = rows[:, 0]
    a1.semilogy(N, rows[:, 1], "o-", label="curved: L2 velocity")
    a1.semilogy(N, rows[:, 2], "s-", mfc="none", label="curved: torque")
    a1.semilogy(N, rows[:, 3], "o--", color="C3", label="straight: L2 velocity")
    a1.semilogy(N, rows[:, 4], "s--", color="C3", mfc="none", label="straight: torque")
    a1.set(xlabel="polynomial order N", ylabel="relative error", title="p-refinement (2 × 8 elements)")
    a1.grid(alpha=0.3, which="both")
    a1.legend(fontsize=8)
    h = 1 / np.array(ks)
    for Np in (1, 2, 3, 4):
        e = hres[(True, Np)]
        a2.loglog(h, e, "o-", label=f"curved N={Np}")
        a2.loglog(h[-2:], e[-1] * (h[-2:] / h[-1]) ** (Np + 1), "k:", lw=0.8)
        a2.text(h[-1] * 0.9, e[-1], f"{Np + 1}", fontsize=8, ha="right", va="center")
    a2.loglog(h, hres[(False, 4)], "s--", color="C3", label="straight N=4 (geometry-limited, O(h²))")
    a2.set_xticks(h, ["1", "1/2", "1/4", "1/8"])
    a2.minorticks_off()
    a2.set(xlabel="h (relative element size)", ylabel="relative L2 error", title="h-refinement: slope N+1 vs 2")
    a2.grid(alpha=0.3, which="both")
    a2.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig("docs/convergence.png", dpi=120)
    plt.close(fig)

    # ---- spin-up: profiles and animation
    m, frames = spin_up()
    tri = triangulation(m)
    r_line = np.linspace(FLOW.r1, FLOW.r2, 200)
    fig, ax = plt.subplots(figsize=(5.8, 4.2))
    sel = m.shape[2] // 2  # take the profile along one element row at mid-eta nodes
    rr = np.hypot(m.x[::m.Kt, :, sel], m.y[::m.Kt, :, sel]).ravel()
    colors = plt.cm.plasma(np.linspace(0, 0.9, 7))
    for c, target in zip(colors, (0.01, 0.03, 0.06, 0.1, 0.2, 0.35, 0.6)):
        t, u = min(frames, key=lambda f: abs(f[0] - target))
        ax.plot(rr, u_theta(m, u)[::m.Kt, :, sel].ravel(), "-", color=c, label=f"t = {t:.3f}")
    ax.plot(r_line, FLOW.u_theta(r_line), "k--", lw=1.5, label="steady Couette (exact)")
    ax.set(xlabel="r", ylabel="u_θ", title="Spin-up of the inner cylinder (ν = 1)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig("docs/spin_up.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.2, 4.2))
    fig.subplots_adjust(0, 0, 1, 0.9)
    tc = ax.tripcolor(tri, u_theta(m, frames[0][1]).ravel(), shading="gouraud", cmap="viridis", vmin=0, vmax=1)
    draw_elements(ax, m, color="w", lw=0.4, alpha=0.6)
    ax.set(aspect="equal", xticks=[], yticks=[])
    ax.axis("off")
    title = ax.set_title("")

    def update(k):
        t, u = frames[k]
        tc.set_array(u_theta(m, u).ravel())
        title.set_text(f"DGSEM N=6, spin-up: u_θ at t = {t:.3f}")
        return [tc]

    FuncAnimation(fig, update, frames=len(frames)).save("docs/spin_up.gif", writer=PillowWriter(fps=12), dpi=70)
    plt.close(fig)


if __name__ == "__main__":
    main()
