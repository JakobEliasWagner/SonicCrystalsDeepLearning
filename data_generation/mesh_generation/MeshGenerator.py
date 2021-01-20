import gmsh
import sys
import numpy as np
from Shapes import Disk, C, Matryoshka


class MeshGenerator:
    def __init__(self, obstacle, a, rows, cols, left_space, bottom_space, right_space, top_space):
        self.obstacle = obstacle
        self.a = a
        self.rows = rows
        self.cols = cols
        self.left_space = left_space
        self.bottom_space = bottom_space
        self.right_space = right_space
        self.top_space = top_space
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

        # boolean substraction of obstacles from fluid space
        fluid = gmsh.model.occ.cut([(2, rect)], [(2, surface) for surface in obstacle_sfs])

        gmsh.model.occ.synchronize()

        # physical Groups
        surfaces = gmsh.model.getEntities(dim=2)
        fluid_marker = 11
        gmsh.model.addPhysicalGroup(surfaces[0][0], [surfaces[0][1]], fluid_marker)
        gmsh.model.setPhysicalName(surfaces[0][0], fluid_marker, "Fluid")

        left_marker, bottom_marker, right_marker, top_marker, obstacle_marker = 1, 3, 5, 7, 9

        curves = gmsh.model.occ.getEntities(dim=1)
        obstacles = []

        for curve in curves:
            com = gmsh.model.occ.getCenterOfMass(curve[0], curve[1])

            # left wall
            if np.allclose(com, [0, self.H / 2, 0]):
                gmsh.model.addPhysicalGroup(curve[0], [curve[1]], left_marker)
                gmsh.model.setPhysicalName(curve[0], left_marker, "Left Wall")

            # bottom wall
            elif np.allclose(com, [self.L / 2, 0, 0]):
                gmsh.model.addPhysicalGroup(curve[0], [curve[1]], bottom_marker)
                gmsh.model.setPhysicalName(curve[0], bottom_marker, "Bottom Wall")

            # right wall
            elif np.allclose(com, [self.L, self.H / 2, 0]):
                gmsh.model.addPhysicalGroup(curve[0], [curve[1]], right_marker)
                gmsh.model.setPhysicalName(curve[0], right_marker, "Right Wall")

            # top wall
            elif np.allclose(com, [self.L / 2, self.H, 0]):
                gmsh.model.addPhysicalGroup(curve[0], [curve[1]], top_marker)
                gmsh.model.setPhysicalName(curve[0], top_marker, "Top Wall")

            # obstacles
            else:
                obstacles.append(curve[1])

        gmsh.model.addPhysicalGroup(1, obstacles, obstacle_marker)
        gmsh.model.setPhysicalName(1, obstacle_marker, "Obstacle")

        # Resolution
        gmsh.model.geo.synchronize()
        # Calculate Resolution for frequency
        lmbda = 343. / f  # wave length at 20°C and at SL
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

        gmsh.model.occ.synchronize()

        gmsh.model.mesh.generate(2)

        path = f"{self.obstacle.save_path}{self.obstacle.name}_{self.rows}x{self.cols}_{f}.msh"
        gmsh.write(path)

        gmsh.finalize()


if __name__ == "__main__":
    obs = C(0.1, 0.08, 0.05)
    obs1 = Disk(0.1)
    obs2 = Matryoshka(0.1, 0.08, 0.05, 0.7, 8)

    mesh_gen = MeshGenerator(obstacle=obs2,
                             a=0.3,
                             rows=2,
                             cols=2,
                             left_space=3,
                             bottom_space=.5,
                             right_space=3,
                             top_space=.5)
    mesh_gen.run(f=5000, epwl=10)
