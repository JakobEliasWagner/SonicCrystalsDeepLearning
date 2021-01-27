import dolfinx
import dolfinx.plotting
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

    def run(self, freq, file_name, amplitude):
        # Derived physical quantities
        omega = 2. * np.pi * freq
        k = omega / self.c

        # Load mesh
        mesh, cell_tags, facet_tags = read_from_msh(file_name, cell_data=True, facet_data=True, gdim=2)
        n = ufl.FacetNormal(mesh)

        # idk experiments
        ds_excitation = ufl.Measure("ds", domain=mesh, subdomain_data=facet_tags, subdomain_id=1)

        # Source amplitude
        if dolfinx.has_petsc_complex:
            amp = (1 + 1j) * amplitude
        else:
            amp = amplitude

        # Define Test and trial function space
        V = dolfinx.FunctionSpace(mesh, ("Lagrange", self.deg))

        # Define Function for Perfectly Matched Layer
        p = 2  # degree of sigma function in PML transformation (Usually power function)
        sigma = dolfinx.Function(V)
        sigma.interpolate(lambda x: np.maximum(0, x[0] - 0.33) ** p)

        # Test and trial function
        u = ufl.TrialFunction(V)
        chi = ufl.TestFunction(V)
        v_s = dolfinx.Function(V)
        v_s_const = dolfinx.Constant(V, amp)

        # Define variational problem
        v_s.interpolate(lambda x: amp * np.cos(k * x[0]) * np.cos(k * x[1]))
        a = ufl.inner(ufl.grad(u), ufl.grad(chi)) * (1 / (1 - 1j * sigma)) * ufl.dx - k ** 2 * ufl.inner(u,
                                                                                                         chi) * (
                    1 / (1 - 1j * sigma)) * ufl.dx
        L = -1j * omega * self.rho_f * ufl.inner(v_s_const, chi) * ds_excitation

        # solve problem
        uh = dolfinx.fem.Function(V)
        uh.name = "u"
        problem = dolfinx.fem.LinearProblem(a, L, u=uh)
        problem.solve()

        self.save(mesh, uh, f"../Solution/{freq}_sim.xdmf")

    def save(self, mesh_, uh_, name):
        # Save solution in XDMF format (to be viewed in Paraview, for example)
        with dolfinx.io.XDMFFile(MPI.COMM_WORLD, name, "w",
                                 encoding=dolfinx.io.XDMFFile.Encoding.HDF5) as file:
            file.write_mesh(mesh_)
            file.write_function(uh_)


if __name__ == "__main__":
    solver = HelmholtzSolver(1)
    for x in range(500, 20001, 500):
        solver.run(x, f"../mesh_generation/C-Shape_meshes/C_00065_0005_0004_10x10_{x}.msh", 1)
