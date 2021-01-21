import dolfinx
from mpi4py import MPI
import ufl
import numpy as np
import sys
import os

# Relative imports
CURR_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(CURR_DIR, "../utilities"))
from gmsh_helpers import read_from_msh


class HelmholtzSolver:
    def __init__(self, deg):
        # Physical constants for air at 20°C and SL
        self.c = 343.
        self.rho_f = 1.2041

        # Simulation properties
        self.deg = deg

    def run(self, f, file_name, amplitude):
        # Derived physical quantities
        omega = 2. * np.pi * f
        k = omega / self.c

        # Load mesh
        mesh, cell_tags, facet_tags = read_from_msh(file_name, cell_data=True, facet_data=True, gdim=2)
        n = ufl.FacetNormal(mesh)

        # Source amplitude
        if dolfinx.has_petsc_complex:
            amp = (1 + 1j) * amplitude
        else:
            amp = amplitude

        # Define Test and trial function space
        V = dolfinx.FunctionSpace(mesh, ("Lagrange", self.deg))

        # Test and trial function
        u = ufl.TrialFunction(V)
        v = ufl.TestFunction(V)
        f = dolfinx.Function(V)

        # Define variational problem
        f.interpolate(lambda x: amp * k ** 2 * np.cos(k * x[0]))
        a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx - k ** 2 * ufl.inner(u, v) * ufl.dx
        L = ufl.inner(f, v) * ufl.dx

        # solve problem
        uh = dolfinx.fem.Function(V)
        uh.name = "u"
        problem = dolfinx.fem.LinearProblem(a, L, u=uh)
        problem.solve()

        self.save(mesh, uh, "plane_wave.xdmf")

    def save(self, mesh_, uh_, name):
        # Save solution in XDMF format (to be viewed in Paraview, for example)
        with dolfinx.io.XDMFFile(MPI.COMM_WORLD, name, "w",
                                 encoding=dolfinx.io.XDMFFile.Encoding.HDF5) as file:
            file.write_mesh(mesh_)
            file.write_function(uh_)


if __name__ == "__main__":
    solver = HelmholtzSolver(1)
    solver.run(8000, "../mesh_generation/Disk_meshes/D_00065_10x10_8000.msh", 1)
