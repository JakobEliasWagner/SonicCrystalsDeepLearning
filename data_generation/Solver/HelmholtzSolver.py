import numpy as np
import time
import matplotlib
import dolfinx
import dolfinx.geometry
import ufl
from mpi4py import MPI
from dolfinx.io import XDMFFile
from dolfinx.cpp.mesh import CellType
from petsc4py import PETSc
import matplotlib.pyplot as plt
import sys
import os

# Relative imports
CURR_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(CURR_DIR, "../utilities"))
from gmsh_helpers import read_from_msh


def save(mesh_, u_, name):
    with XDMFFile(MPI.COMM_WORLD, name, "w") as file:
        file.write_mesh(mesh_)
        file.write_function(u_)


class IncidentWave:
    def __init__(self, k0):
        self.k = k0

    def eval(self, x):
        return np.exp(1.0j * self.k * x[0])


class AdiabaticLayer:
    def __init__(self, deg_absorber, k0, lmda):
        self.deg_absorber = deg_absorber
        self.k0 = k0
        d_absorb = 5 * lmda
        rt = 1.0e-6  # round-trip reflection
        self.sigma_0 = -(self.deg_absorber + 1) * np.log(rt) / (2.0 * d_absorb)

    def eval(self, x):
        # In absorbing layer: k = k0 + 1j * sigma
        # => k^2 = (k0 + 1j*sigma)^2 = k0^2 + 2j*sigma - sigma^2
        # Therefore, the 2j*sigma - sigma^2 piece must be included in the layer.

        # Find domains with absorbtion-tag
        in_absorber_x = (np.abs(x[0]) >= 0.168)

        # Function sigma = sigma_0 * x^d, where x is the depth into adiabatic layer
        sigma_x = self.sigma_0 * (np.abs(x[0]) >= 0.168) ** self.deg_absorber

        x_layers = in_absorber_x * (2j * sigma_x * self.k0 - sigma_x ** 2)

        return x_layers


class HelmholtzSolver:
    def __init__(self, c=343., rho_0=1.2041, degree=3, deg_absorber=2):
        self.check_PETSc()

        # physical constants at 20°C and SL
        self.c = c
        self.rho_0 = rho_0
        self.f = None
        self.omega = None
        self.lmda = None
        self.k0 = None

        # solver constants
        self.degree = degree  # polynomial degree
        self.deg_absorber = deg_absorber  # degree of absorption nominal

    def check_PETSc(self):
        if not dolfinx.has_petsc_complex:
            raise Exception("This solver only works with PETSc-complex. Try switching to complex builds of DOLFINX.")

    def solve(self, f, filename):
        # redifine constans for specific frequency
        self.f = f
        self.omega = 2 * np.pi * f
        self.lmda = self.c / f
        self.k0 = self.omega / self.c

        # Load mesh
        mesh, cell_tags, facet_tags = read_from_msh(
            filename,
            cell_data=True, facet_data=True, gdim=2)

        # Define function space
        V = dolfinx.FunctionSpace(mesh, ("Lagrange", self.degree))

        # Interpolate wavenumber k onto V
        k = dolfinx.Constant(V, self.k0)

        # Interpolate absorbing layer piece of wavenumber k_absorb onto V
        k_absorb = dolfinx.Function(V)
        adiabatic_layer = AdiabaticLayer(self.deg_absorber, self.k0, self.lmda)
        k_absorb.interpolate(adiabatic_layer.eval)

        # Interpolate incident wave field onto V
        ui = dolfinx.Function(V)
        ui_expr = IncidentWave(self.k0)
        ui.interpolate(ui_expr.eval)

        # Define variational problem
        u = ufl.TrialFunction(V)
        v = ufl.TestFunction(V)

        a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx \
            - k ** 2 * ufl.inner(u, v) * ufl.dx \
            - k_absorb * ufl.inner(u, v) * ufl.dx

        L = -1j * self.omega * self.rho_0 * ufl.inner(ui, v) * ufl.dx

        # Assemble matrix and vector and set up direct solver
        A = dolfinx.fem.assemble_matrix(a)
        A.assemble()
        b = dolfinx.fem.assemble_vector(L)
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)

        solver = PETSc.KSP().create(mesh.mpi_comm())
        opts = PETSc.Options()
        opts["ksp_type"] = "preonly"
        opts["pc_type"] = "lu"
        opts["pc_factor_mat_solver_type"] = "mumps"
        solver.setFromOptions()
        solver.setOperators(A)

        # Solve linear system
        u = dolfinx.Function(V)
        start = time.time()
        solver.solve(b, u.vector)
        end = time.time()
        time_elapsed = end - start
        print('Solve time: ', time_elapsed)
        u.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT,
                             mode=PETSc.ScatterMode.FORWARD)
        save(mesh, u, "sol.xdmf")


if __name__ == "__main__":
    sol = HelmholtzSolver()
    sol.solve(10000, "../mesh_generation/C-Shape_meshes/C_00065_0005_0004_10x10_25000.msh")
