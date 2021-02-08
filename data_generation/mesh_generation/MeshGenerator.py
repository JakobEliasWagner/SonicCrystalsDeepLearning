import gmsh
import sys
import numpy as np
from Shapes import Disk, C, Matryoshka


class MeshGenerator:
    def __init__(self, obstacle, a, rows, cols, left_space, bottom_space, right_space, top_space, left_pml_size=5,
                 right_pml_size=5):
        self.obstacle = obstacle
        self.a = a
        self.rows = rows
        self.cols = cols
        self.left_space = left_space
        self.bottom_space = bottom_space
        self.right_space = right_space
        self.top_space = top_space
        self.left_pml_size = left_pml_size
        self.right_pml_size = right_pml_size
        self.L = (left_space + right_space + cols - 1) * a
        self.H = (top_space + bottom_space + rows - 1) * a

    def run(self, f, epwl):
        """
        :param f: Frequency to ensure mesh width
        :param epwl: Minimum elements per wave length
        :save: File with generated mesh in .msh format
        """
        gmsh.initialize(sys.argv)
        gmsh.model.add("sonic crystals")

        # Define Fluid Domain
        rect = gmsh.model.occ.addRectangle(0, 0, 0, self.L, self.H)

        # Define Obstacles
        obstacle_sfs = []
        for row in range(self.rows):
            for col in range(self.cols):
                shape = self.obstacle.touch(x=(self.left_space + col) * self.a,
                                            y=(self.bottom_space + row) * self.a,
                                            z=0)
                obstacle_sfs.extend(shape)

        # boolean subtraction of obstacles from fluid space
        fluid = gmsh.model.occ.cut([(2, rect)], [(2, surface) for surface in obstacle_sfs])

        # Add PML areas
        lmbda = 343. / f  # wave length at 20°C and at SL
        l_pml = r_pml = None
        pmls = []
        if self.left_pml_size > 0:
            left_pml_length = self.left_pml_size * lmbda  # calculated as multiples of wave length
            left_pml = gmsh.model.occ.addRectangle(0, 0, 0, -left_pml_length, self.H)  # left pml area
            pmls.append(left_pml)

        if self.right_pml_size > 0:
            right_pml_length = self.right_pml_size * lmbda  # calculated as multiples of wave length
            right_pml = gmsh.model.occ.addRectangle(self.L, 0, 0, right_pml_length, self.H)  # right pml area
            pmls.append(right_pml)

        if len(pmls) > 0:
            gmsh.model.occ.fragment([fluid[0][0]], [(2, x) for x in pmls])  # connect surfaces

        # physical Groups
        gmsh.model.occ.synchronize()
        surfaces = gmsh.model.getEntities(dim=2)
        gmsh.model.occ.translate(surfaces, -self.L / 2, -self.H / 2, 0)
        gmsh.model.occ.synchronize()
        fluid_marker, pml_marker = 11, 13

        pml_surfaces = []
        for surface in surfaces:
            com = gmsh.model.occ.getCenterOfMass(surface[0], surface[1])
            if -self.L / 2 < com[0] < self.L / 2:
                gmsh.model.addPhysicalGroup(surface[0], [surface[1]], fluid_marker)
                gmsh.model.setPhysicalName(surfaces[0][0], fluid_marker, "Fluid")
            else:
                pml_surfaces.append(surface)
        gmsh.model.addPhysicalGroup(surfaces[0][0], [x[1] for x in pml_surfaces], pml_marker)
        gmsh.model.setPhysicalName(surfaces[0][0], pml_marker, "Perfectly Matched Layer")

        left_marker, bottom_marker, right_marker, top_marker, obstacle_marker = 1, 3, 5, 7, 9

        curves = gmsh.model.occ.getEntities(dim=1)
        obstacles = []

        for curve in curves:
            com = gmsh.model.occ.getCenterOfMass(curve[0], curve[1])

            # left wall
            if np.allclose(com, [-self.L / 2, 0, 0]):
                gmsh.model.addPhysicalGroup(curve[0], [curve[1]], left_marker)
                gmsh.model.setPhysicalName(curve[0], left_marker, "Left Wall")

            # bottom wall
            elif np.allclose(com, [0, -self.H / 2, 0]):
                gmsh.model.addPhysicalGroup(curve[0], [curve[1]], bottom_marker)
                gmsh.model.setPhysicalName(curve[0], bottom_marker, "Bottom Wall")

            # right wall
            elif np.allclose(com, [self.L / 2, 0, 0]):
                gmsh.model.addPhysicalGroup(curve[0], [curve[1]], right_marker)
                gmsh.model.setPhysicalName(curve[0], right_marker, "Right Wall")

            # top wall
            elif np.allclose(com, [0, self.H / 2, 0]):
                gmsh.model.addPhysicalGroup(curve[0], [curve[1]], top_marker)
                gmsh.model.setPhysicalName(curve[0], top_marker, "Top Wall")

            # obstacles
            else:
                obstacles.append(curve[1])

        gmsh.model.addPhysicalGroup(1, obstacles, obstacle_marker)
        gmsh.model.setPhysicalName(1, obstacle_marker, "Obstacle")

        # Resolution
        gmsh.model.occ.synchronize()
        # Calculate Resolution in dependence on the frequency
        lc = lmbda / epwl

        # Define Distance field on circle curve. this field returns the distance to (100 equidistant points on)
        # circle curve
        gmsh.model.mesh.field.add("Distance", 1)
        gmsh.model.mesh.field.setNumbers(1, "CurvesList", obstacles)
        gmsh.model.mesh.field.setNumber(1, "NumPointsPerCurve", 100)

        # Define second field which uses the return value of filed 1 in order to define a simple change in element size
        # depending on the computed distances
        gmsh.model.mesh.field.add("Threshold", 2)
        gmsh.model.mesh.field.setNumber(2, "InField", 1)
        gmsh.model.mesh.field.setNumber(2, "SizeMin", lc)
        gmsh.model.mesh.field.setNumber(2, "SizeMax", lc)
        gmsh.model.mesh.field.setNumber(2, "DistMin", 0.1)
        gmsh.model.mesh.field.setNumber(2, "DistMax", 0.5)

        # apply field
        gmsh.model.mesh.field.add("Min", 3)
        gmsh.model.mesh.field.setNumbers(3, "FieldsList", [2])

        gmsh.model.mesh.field.setAsBackgroundMesh(3)

        #gmsh.model.occ.translate(surfaces, -self.L / 2, 0, 0)
        gmsh.model.occ.synchronize()

        gmsh.model.mesh.generate(2)

        path = f"{self.obstacle.save_path}{self.obstacle.name}_{self.rows}x{self.cols}_{f}.msh"
        gmsh.write(path)

        gmsh.finalize()


if __name__ == "__main__":
    obs = C(0.0065, 0.005, 0.004)

    mesh_gen = MeshGenerator(obstacle=obs,
                             a=0.022,
                             rows=10,
                             cols=10,
                             left_space=3,
                             bottom_space=.5,
                             right_space=3,
                             top_space=.5)
    x = 10000
    mesh_gen.run(f=x, epwl=10)
