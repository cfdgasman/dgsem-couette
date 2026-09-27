# DGSEM on Curvilinear Elements: Circular Couette Flow

[![CI](https://github.com/cfdgasman/dgsem-couette/actions/workflows/ci.yml/badge.svg)](https://github.com/cfdgasman/dgsem-couette/actions/workflows/ci.yml)

A nodal **discontinuous Galerkin spectral element method (DGSEM)** on curvilinear quadrilaterals, applied to **circular (Taylor–)Couette flow** between two rotating cylinders. It shows **exponential convergence** on curved, isoparametric elements. It also shows how **straight-sided elements limit accuracy through geometry error**, no matter how high the polynomial order.

<p align="center">
<img src="docs/spin_up.gif" width="330" alt="Spin-up animation">
<img src="docs/spin_up.png" width="420" alt="Spin-up velocity profiles">
</p>

## Problem

Fluid between r₁ = 1 (rotating, Ω₁ = 1) and r₂ = 2 (at rest). For purely azimuthal, axisymmetric flow the pressure gradient exactly balances the centripetal acceleration. The Cartesian velocity components (u<sub>x</sub>, u<sub>y</sub>) then satisfy

$$ \partial_t \mathbf u = \nu \nabla^2 \mathbf u, \qquad \mathbf u = \boldsymbol\Omega\times\mathbf x \ \text{on the walls}, $$

with the exact steady solution

$$ u_\theta(r) = A r + \frac{B}{r}, \qquad A = \frac{\Omega_2 r_2^2-\Omega_1 r_1^2}{r_2^2-r_1^2},\quad B = \frac{(\Omega_1-\Omega_2)r_1^2r_2^2}{r_2^2-r_1^2}, $$

and torque per unit length |T| = 4πμ|B| on each cylinder.

## Method

| | |
|---|---|
| Elements | K<sub>r</sub> × K<sub>θ</sub> quadrilaterals, tensor-product Lagrange basis of degree N on **Legendre–Gauss–Lobatto** nodes (collocation, SBP property) |
| Geometry | **Curved:** nodes on the exact polar map (isoparametric). **Straight:** bilinear map between corners (polygonal walls). Metric terms from the interpolated mapping, so the **discrete metric identities hold exactly** (checked in the tests). |
| Viscous terms | Mixed form q = ∇u, ∇²u = ∇·q in strong DGSEM form; **BR1** central fluxes plus an interior-penalty term (η = (N+1)²/h<sub>n</sub>) |
| Boundary conditions | Rotating-wall velocity **Ω × x imposed at the wall nodes**, so a polygonal wall really does carry a geometry error |
| Linear algebra | Operator assembled as a sparse matrix by batched probing; sparse LU for the steady and implicit (backward Euler) spin-up solves |
| Torque | Wall traction from the DG velocity gradient, integrated with LGL quadrature |

## Results

### Curved vs straight-sided elements

<p align="center"><img src="docs/mesh_solution.png" width="760" alt="Curved and straight-sided meshes"></p>

### Convergence

<p align="center"><img src="docs/convergence.png" width="900" alt="p- and h-convergence"></p>

**p-refinement** on a fixed 2 × 8 mesh:

| N | curved L2 | curved torque error | straight L2 | straight torque error |
|---|---|---|---|---|
| 2 | 4.09e-03 | 2.95e-02 | 1.91e-01 | 4.37e-02 |
| 4 | 1.65e-05 | 6.76e-04 | 1.81e-01 | 3.19e-02 |
| 6 | 1.06e-07 | 1.19e-05 | 1.82e-01 | 3.33e-02 |
| 8 | 8.40e-10 | 1.83e-07 | 1.82e-01 | 3.38e-02 |
| 10 | 7.04e-12 | 2.58e-09 | 1.82e-01 | 3.40e-02 |
| 12 | **9.77e-14** | **3.43e-11** | 1.82e-01 | 3.40e-02 |

**h-refinement**, observed L2 order between the two finest meshes:

| N | curved | straight |
|---|---|---|
| 1 | 2.35 | 2.35 (identical: both maps are bilinear) |
| 2 | 3.42 | 1.98 |
| 3 | 4.36 | 1.96 |
| 4 | 5.11 | 1.97 |

- **Curved elements** converge **exponentially** in N, reaching 10⁻¹³ with just 16 elements, and at the optimal rate **O(h<sup>N+1</sup>)** under h-refinement. The torque, which depends on the gradient, converges at about O(h<sup>N</sup>).
- **Straight-sided elements** stall at 18 % velocity error and 3.4 % torque error for *any* N. Under h-refinement they converge only at **O(h²)**, the rate at which the polygon approaches the circle. High-order methods need high-order geometry.

### Spin-up

The inner cylinder is started impulsively (backward Euler, Δt = 2×10⁻³, N = 6, 3 × 12 elements). Momentum diffuses outward, and the profile relaxes onto the exact Couette solution over the viscous time scale (r₂ − r₁)²/ν.

## Usage

```bash
pip install -r requirements.txt
python run.py      # tables, figures and GIF in docs/ (~40 s)
pytest             # quadrature, SBP derivative, metric identities, exactness, convergence, torque
```

## References

D. A. Kopriva, *Implementing Spectral Methods for Partial Differential Equations*, Springer, 2009.
F. Bassi, S. Rebay, *A high-order accurate discontinuous finite element method for the numerical solution of the compressible Navier–Stokes equations*, J. Comput. Phys. 131 (1997) 267–279.

## License

MIT
