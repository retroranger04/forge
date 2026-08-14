"""Single-sample FE pipeline: 2D square plate with a central circular hole.

gmsh geometry/mesh -> dolfinx P2 linear elasticity -> von Mises on a regular
grid. Far-field uniform tension of magnitude sigma_inf at angle theta from the
x-axis, applied as a Neumann traction on the outer boundary; hole traction-free.

Geometry/BC/material setup adapted from benchmarks/scratch/b2_fenicsx_plate.py
(FORGE Session Three, B2).
"""
from __future__ import annotations

import math
import signal
from typing import Literal

import gmsh
import numpy as np
import ufl
from dolfinx import fem, geometry
from dolfinx.fem.petsc import apply_lifting, assemble_matrix, assemble_vector, set_bc
from dolfinx.io.gmsh import model_to_mesh
from mpi4py import MPI
from petsc4py import PETSc

DOMAIN_TAG, OUTER_TAG = 1, 1


class SampleTimeoutError(RuntimeError):
    """Raised when a single FE solve exceeds its wall-clock budget."""


def _build_mesh(r: float, plate_side: float, comm: MPI.Comm):
    """Square-with-hole geometry, refined to <= r/10 on the hole boundary."""
    half = plate_side / 2.0
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        if comm.rank == 0:
            gmsh.model.add("plate_with_hole")
            square = gmsh.model.occ.addRectangle(-half, -half, 0.0, plate_side, plate_side)
            hole = gmsh.model.occ.addDisk(0.0, 0.0, 0.0, r, r)
            cut, _ = gmsh.model.occ.cut([(2, square)], [(2, hole)])
            gmsh.model.occ.synchronize()

            surface = cut[0][1]
            gmsh.model.addPhysicalGroup(2, [surface], DOMAIN_TAG)

            # Outer edges touch |x| = half or |y| = half; the hole curve does not.
            tol = 1e-6
            outer, hole_curves = [], []
            for dim, tag in gmsh.model.getBoundary([(2, surface)], oriented=False):
                xmin, ymin, _, xmax, ymax, _ = gmsh.model.getBoundingBox(dim, tag)
                on_outer = (
                    abs(abs(xmin) - half) < tol or abs(abs(xmax) - half) < tol
                    or abs(abs(ymin) - half) < tol or abs(abs(ymax) - half) < tol
                )
                (outer if on_outer else hole_curves).append(tag)
            gmsh.model.addPhysicalGroup(1, outer, OUTER_TAG)

            fine = gmsh.model.mesh.field.add("Distance")
            gmsh.model.mesh.field.setNumbers(fine, "CurvesList", hole_curves)
            thr = gmsh.model.mesh.field.add("Threshold")
            gmsh.model.mesh.field.setNumber(thr, "InField", fine)
            gmsh.model.mesh.field.setNumber(thr, "SizeMin", r / 10.0)
            gmsh.model.mesh.field.setNumber(thr, "SizeMax", plate_side / 32.0)
            gmsh.model.mesh.field.setNumber(thr, "DistMin", r)
            gmsh.model.mesh.field.setNumber(thr, "DistMax", 4.0 * r)
            gmsh.model.mesh.field.setAsBackgroundMesh(thr)
            for opt in ("MeshSizeExtendFromBoundary", "MeshSizeFromPoints", "MeshSizeFromCurvature"):
                gmsh.option.setNumber(f"Mesh.{opt}", 0)
            gmsh.option.setNumber("Mesh.Algorithm", 6)  # Frontal-Delaunay, deterministic
            gmsh.model.mesh.generate(2)

        mesh_data = model_to_mesh(gmsh.model, comm, 0, gdim=2)
    finally:
        gmsh.finalize()
    return mesh_data.mesh, mesh_data.facet_tags


def _solve(r, sigma_inf, theta_deg, physics, resolution, plate_side, E, nu,
           E1=None, E2=None, nu12=None, G12=None, pre_stress_p=0.0):
    comm = MPI.COMM_WORLD
    half = plate_side / 2.0
    msh, facet_tags = _build_mesh(r, plate_side, comm)

    if physics == "plane_stress":
        lmbda = E * nu / (1.0 - nu**2)
    else:
        lmbda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    mu = E / (2.0 * (1.0 + nu))

    V = fem.functionspace(msh, ("Lagrange", 2, (2,)))
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)

    def eps(w):
        return ufl.sym(ufl.grad(w))

    if E1 is None:
        def sig(w):
            return lmbda * ufl.tr(eps(w)) * ufl.Identity(2) + 2.0 * mu * eps(w)
    else:
        # Orthotropic plane-stress stiffness Q, standard CLT symmetric form.
        # Fiber direction along x-axis.
        # Reciprocity nu21/E2 = nu12/E1 gives Q12 == Q21, so the bilinear form
        # stays symmetric: a non-symmetric Q has no strain energy density.
        nu21 = (E2 / E1) * nu12
        den = 1.0 - nu12 * nu21
        if den <= 0.0:
            raise ValueError(f"non positive-definite orthotropic law: 1 - nu12*nu21 = {den}")
        # Runtime Constants rather than literals folded into the form, for the
        # same reason as nhat/sigma_inf below: baking the stiffness in gives
        # every anisotropy ratio its own form signature and a fresh FFCx
        # compilation, which would blow the per-sample timeout once per ratio.
        def _c(value):
            return fem.Constant(msh, PETSc.ScalarType(value))

        q11, q12, q22, q66 = _c(E1 / den), _c(nu21 * E1 / den), _c(E2 / den), _c(G12)

        def sig(w):
            e = eps(w)
            sxy = 2.0 * q66 * e[0, 1]
            return ufl.as_matrix([[q11 * e[0, 0] + q12 * e[1, 1], sxy],
                                  [sxy, q12 * e[0, 0] + q22 * e[1, 1]]])

    # Rigid-body modes: full pin at (-half,-half), u_y pin at (+half,+half).
    # Exactly 3 constrained DOFs for 3 RBMs; the traction load is
    # self-equilibrated, so the reactions vanish and the stress is unaffected.
    atol = 1e-8

    def at_lower_left(x):
        return np.isclose(x[0], -half, atol=atol) & np.isclose(x[1], -half, atol=atol)

    def at_upper_right(x):
        return np.isclose(x[0], half, atol=atol) & np.isclose(x[1], half, atol=atol)

    bc_full = fem.dirichletbc(
        np.zeros(2, dtype=PETSc.ScalarType),
        fem.locate_dofs_geometrical(V, at_lower_left),
        V,
    )
    Vy, _ = V.sub(1).collapse()
    bc_uy = fem.dirichletbc(
        PETSc.ScalarType(0.0),
        fem.locate_dofs_geometrical((V.sub(1), Vy), at_upper_right)[0],
        V.sub(1),
    )
    bcs = [bc_full, bc_uy]

    # Far-field uniaxial stress sigma_inf * (n_hat (x) n_hat) gives boundary
    # traction sigma_inf * (n_hat . N) n_hat. This vanishes on the edges
    # parallel to the load direction (exactly traction-free at theta = 0, 90).
    # nhat and sigma_inf are runtime Constants, not literals folded into the
    # form: baking them in gives every theta its own form signature, which
    # sends FFCx into a fresh C compilation per sample.
    th = math.radians(theta_deg)
    nhat = fem.Constant(msh, np.array([math.cos(th), math.sin(th)], dtype=PETSc.ScalarType))
    N = ufl.FacetNormal(msh)
    ds = ufl.Measure("ds", domain=msh, subdomain_data=facet_tags)
    trac = fem.Constant(msh, PETSc.ScalarType(sigma_inf)) * ufl.dot(nhat, N) * nhat

    a_form = fem.form(ufl.inner(sig(u), eps(v)) * ufl.dx)
    L_form = fem.form(ufl.inner(trac, v) * ds(OUTER_TAG))

    A = assemble_matrix(a_form, bcs=bcs)
    A.assemble()
    b = assemble_vector(L_form)
    apply_lifting(b, [a_form], bcs=[bcs])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
    set_bc(b, bcs)

    uh = fem.Function(V)
    ksp = PETSc.KSP().create(comm)
    ksp.setOperators(A)
    ksp.setType("preonly")
    ksp.getPC().setType("lu")
    ksp.solve(b, uh.x.petsc_vec)
    uh.x.scatter_forward()

    # sigma(P2 displacement) is discontinuous-linear, so DG1 is exact.
    W = fem.functionspace(msh, ("DG", 1))
    s = sig(uh)
    # Uniform biaxial pre-stress p added post-solve. Superposition-valid for
    # linear elasticity.
    # Skipped entirely at p == 0 so the zero case stays bit-identical to the
    # form before pre-stress support existed. A runtime Constant rather than a folded literal, for
    # the same reason as nhat/sigma_inf above: one form signature covers every
    # pre-stress magnitude instead of one FFCx compilation per magnitude.
    if pre_stress_p:
        s = s + fem.Constant(msh, PETSc.ScalarType(pre_stress_p)) * ufl.Identity(2)
    # Full 3D von Mises. Plane strain carries sigma_zz = nu*(sxx+syy); plane
    # stress has sigma_zz = 0, where this reduces exactly to the 2D form.
    szz = nu * (s[0, 0] + s[1, 1]) if physics == "plane_strain" else 0.0
    vm_ufl = ufl.sqrt(
        0.5 * ((s[0, 0] - s[1, 1]) ** 2 + (s[1, 1] - szz) ** 2 + (szz - s[0, 0]) ** 2)
        + 3.0 * s[0, 1] ** 2
    )
    vm = fem.Function(W)
    vm.interpolate(fem.Expression(vm_ufl, W.element.interpolation_points))

    # Peak stress read off the FE field in a thin band at the hole boundary,
    # not off the coarse output grid (which under-resolves small holes).
    dof_xy = W.tabulate_dof_coordinates()
    rho = np.hypot(dof_xy[:, 0], dof_xy[:, 1])
    band = (rho >= r) & (rho <= 1.05 * r)
    peak = float(vm.x.array[band].max()) if np.any(band) else float("nan")

    axis = np.linspace(-half, half, resolution)
    gx, gy = np.meshgrid(axis, axis)
    sdf = np.hypot(gx, gy) - r
    mask = sdf > 0.0
    pts = np.column_stack([gx.ravel(), gy.ravel(), np.zeros(gx.size)])

    tree = geometry.bb_tree(msh, msh.topology.dim)
    colliding = geometry.compute_colliding_cells(
        msh, geometry.compute_collisions_points(tree, pts), pts
    )
    cells = np.full(pts.shape[0], -1, dtype=np.int32)
    for i in range(pts.shape[0]):
        links = colliding.links(i)
        if len(links) > 0:
            cells[i] = links[0]

    # Grid points just outside the hole can miss every cell because gmsh
    # inscribes a polygon in the circle; snap those to the nearest cell.
    stray = np.flatnonzero((cells < 0) & mask.ravel())
    if stray.size:
        ncells = msh.topology.index_map(msh.topology.dim).size_local
        mid = geometry.create_midpoint_tree(msh, msh.topology.dim, np.arange(ncells, dtype=np.int32))
        cells[stray] = geometry.compute_closest_entity(tree, mid, msh, pts[stray])

    von_mises = np.zeros(pts.shape[0])
    good = np.flatnonzero(cells >= 0)
    if good.size:
        von_mises[good] = vm.eval(pts[good], cells[good]).ravel()
    von_mises = von_mises.reshape(gx.shape)
    von_mises[~mask] = 0.0  # hole interior

    A.destroy()
    b.destroy()
    ksp.destroy()

    return {
        "von_mises": von_mises.astype(np.float32),
        "sdf": sdf.astype(np.float32),
        "mask": mask.astype(np.uint8),
        "params": np.array([r, sigma_inf, theta_deg], dtype=np.float32),
        "physics": physics,
        "peak_hole_von_mises": peak,
    }


def generate_sample(
    r: float,
    sigma_inf: float,
    theta_deg: float,
    physics: Literal["plane_stress", "plane_strain"],
    resolution: int = 64,
    plate_side: float = 1.0,
    E: float = 1.0,
    nu: float = 0.3,
    E1: float | None = None,
    E2: float | None = None,
    nu12: float | None = None,
    G12: float | None = None,
    pre_stress_p: float = 0.0,
    timeout_sec: float = 30.0,
) -> dict:
    """Solve plate-with-hole. Raises SampleTimeoutError if solve exceeds timeout.

    r is the hole radius in physical plate coordinates, not the r/L ratio.
    The plate has side `plate_side` and half-width L = plate_side / 2, so a
    caller working in r/L must pass r = (r/L) * plate_side / 2. The spec's
    ratio ranges live in scripts/generate_dataset.py, which does that
    conversion via L_HALF before calling here.

    Passing all four of E1, E2, nu12, G12 selects the orthotropic plane-stress
    law (fibers along x) instead of the isotropic (E, nu) one; passing none
    keeps the isotropic path. A partial set is an error.

    pre_stress_p is a uniform biaxial pre-stress added to sigma_xx and sigma_yy
    after the solve, in the same units as sigma_inf; sigma_xy is untouched. It
    models an isotropic in-plane residual stress state. The default 0.0 leaves
    the field exactly as it was before pre-stress support was added. Under plane_stress, sigma_zz is
    0 regardless; under plane_strain, sigma_zz follows the pre-stressed in-plane
    components.

    Returns dict with keys: von_mises, sdf, mask, params, physics.
    (Also peak_hole_von_mises, the hole-boundary peak read off the FE field,
    used by the physics validation harness.)
    """
    if physics not in ("plane_stress", "plane_strain"):
        raise ValueError(f"unknown physics: {physics!r}")

    # Negative pre-stress is not part of the design; NaN fails this too.
    if not pre_stress_p >= 0.0:
        raise ValueError(f"pre_stress_p must be non-negative, got {pre_stress_p!r}")

    ortho = (E1, E2, nu12, G12)
    if any(c is not None for c in ortho):
        if not all(c is not None for c in ortho):
            raise ValueError(f"orthotropic needs all of E1, E2, nu12, G12; got {ortho}")
        # A negative or non-finite modulus does not fail the solve, it silently
        # produces a non-physical stiffness (G12 < 0 gives q66 < 0), so the
        # moduli are screened here rather than read back out of the field.
        if not all(math.isfinite(c) for c in ortho):
            raise ValueError(f"orthotropic parameters must be finite; got {ortho}")
        if E1 <= 0.0 or E2 <= 0.0 or G12 <= 0.0:
            raise ValueError(f"E1, E2 and G12 must be positive; got {ortho}")
        # The reduced stiffness below is the plane-stress form only; plane strain
        # would need a different derivation, so refuse rather than solve a law
        # that does not match the requested physics.
        if physics != "plane_stress":
            raise ValueError(f"orthotropic law is plane_stress only, got {physics!r}")

    def _fire(signum, frame):
        raise SampleTimeoutError(
            f"solve exceeded {timeout_sec}s: r={r} sigma_inf={sigma_inf} "
            f"theta_deg={theta_deg} physics={physics}"
        )

    # SIGALRM only interrupts at a Python bytecode boundary; a hang inside a
    # gmsh/PETSc C call is caught by the batch-level watchdog instead.
    prev = signal.signal(signal.SIGALRM, _fire)
    signal.setitimer(signal.ITIMER_REAL, timeout_sec)
    try:
        return _solve(r, sigma_inf, theta_deg, physics, resolution, plate_side, E, nu,
                      E1, E2, nu12, G12, pre_stress_p)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, prev)
