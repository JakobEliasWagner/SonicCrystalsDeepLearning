import gmsh
from abc import ABC, abstractmethod


class Shape(ABC):
    """
    Shapes to place in fluid domain
    """

    @abstractmethod
    def touch(self, x, y, z):
        """
        :param x:
        :param y:
        :param z:
        :return: Shape tag
        """
        pass


class Disk(Shape):
    """
    Round Disk
    """

    def __init__(self, r):
        """
        :param r: Radius
        """
        self.r = r

        # for saving
        self.save_path = "Disk_meshes/"
        self.name = f"D_{str(r).replace('.', '')}"

    def touch(self, x, y, z):
        # shapes
        disk = gmsh.model.occ.addDisk(xc=x, yc=y, zc=z, rx=self.r, ry=self.r)

        return [disk]


class C(Shape):
    """
    C Shape
    """

    def __init__(self, r1, r2, b):
        """
        :param r1: Outer Radius
        :param r2: Inner Radius
        :param b: Gap Width
        """
        self.r1 = r1
        self.r2 = r2
        self.b = b

        # for saving
        self.save_path = "C-Shape_meshes/"
        self.name = f"C_{str(r1).replace('.', '')}_{str(r2).replace('.', '')}_{str(b).replace('.', '')}"

    def touch(self, x, y, z):
        # shapes
        circle1 = gmsh.model.occ.addDisk(xc=x, yc=y, zc=z, rx=self.r1, ry=self.r1)
        circle2 = gmsh.model.occ.addDisk(xc=x, yc=y, zc=z, rx=self.r2, ry=self.r2)
        rectangle1 = gmsh.model.occ.addRectangle(x=x, y=y - self.b / 2, z=z, dx=-self.r1, dy=self.b)

        # boolean operations
        cut1 = gmsh.model.occ.cut([(2, circle1)], [(2, circle2), (2, rectangle1)])[0][0][1]

        return [cut1]


class Matryoshka(Shape):
    def __init__(self, r1, r2, b, scale_factor, layers):
        """
        :param r1: Outer radius of outer most layer
        :param r2: Inner radius of outer most layer
        :param b: Gap width of outer most layer
        :param scale_factor: Scale factor of    scale_factor = next_layer_size / layer_size
        :param layers: number of nested layers
        """
        self.r1 = r1
        self.r2 = r2
        self.b = b
        self.scale_factor = scale_factor
        self.layers = layers

        # for saving
        self.save_path = "Matryoshka_meshes/"
        self.name = f"Mtr_{str(r1).replace('.', '')}_{str(r2).replace('.', '')}_{str(b).replace('.', '')}_" +\
                        f"{str(scale_factor).replace('.', '')}_{str(layers).replace('.', '')}"

    def touch(self, x, y, z):
        outer_layer = C(self.r1, self.r2, self.b)

        cs = outer_layer.touch(x=x, y=y, z=z)

        for layer in range(self.layers - 1):
            a = [(2, cs[-1])]
            cp = gmsh.model.occ.copy([(2, cs[-1])])
            gmsh.model.occ.dilate(cp, x, y, z, self.scale_factor, self.scale_factor, self.scale_factor)
            cs.append(cp[0][1])

        # fuse into one surface
        fuse1 = gmsh.model.occ.fuse([(2, cs[0])], [(2, x) for x in cs[1:]])
        return [x[1] for x in fuse1[0]]