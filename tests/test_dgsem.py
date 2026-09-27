import numpy as np
import pytest

from dgsem import Couette, Laplacian, Mesh, inner_torque, l2_error, solve_steady
from dgsem.basis import diff_matrix, lgl

FLOW = Couette()


def test_lgl_quadrature_exact_to_degree_2n_minus_1():
    x, w = lgl(6)
    for k in range(12):
        assert w @ x**k == pytest.approx((1 - (-1) ** (k + 1)) / (k + 1), abs=1e-13)


def test_differentiation_exact_for_polynomials():
    x, _ = lgl(7)
    assert np.allclose(diff_matrix(x) @ x**7, 7 * x**6, atol=1e-12)


def test_metric_identities_hold_discretely():
    m = Mesh(6, 2, 6, curved=True)
    div = np.einsum("ik,ekj->eij", m.D, m.Ja1[0]) + np.einsum("jk,eik->eij", m.D, m.Ja2[0])
    assert np.abs(div).max() < 1e-12


def test_linear_field_is_reproduced_exactly():
    """A linear function is harmonic, so it must be recovered to round-off (free-stream-type check)."""
    m = Mesh(4, 2, 8, curved=True)
    op = Laplacian(m)
    u = (2 * m.x - 3 * m.y + 1)[..., None]
    g_in = u[: m.Kt, 0]
    g_out = u[-m.Kt :, -1]
    assert np.abs(op.apply(u, g_in, g_out)).max() < 1e-9


def test_spectral_convergence_on_curved_mesh():
    errs = []
    for N in (4, 8):
        m = Mesh(N, 2, 8, curved=True)
        u, _, _ = solve_steady(m, FLOW)
        errs.append(l2_error(m, FLOW, u))
    assert errs[1] < 1e-9 and errs[0] / errs[1] > 1e4


def test_straight_sided_elements_are_geometry_limited():
    m = Mesh(8, 2, 8, curved=False)
    u, _, _ = solve_steady(m, FLOW)
    assert l2_error(m, FLOW, u) > 0.1


def test_torque_converges_to_exact():
    m = Mesh(10, 2, 8, curved=True)
    u, op, g = solve_steady(m, FLOW)
    assert inner_torque(m, FLOW, u, op, g) == pytest.approx(FLOW.torque(), rel=1e-7)
