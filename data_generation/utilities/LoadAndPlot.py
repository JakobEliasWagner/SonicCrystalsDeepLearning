import h5py
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import numpy as np

# get data
f = h5py.File('sol.h5', 'r')

a = f['Function']['real_f_6']['0']  # Werte
b = f['Mesh']['mesh']['topology']
c = f['Mesh']['mesh']['geometry']  # Koordinaten pro eintrag xy
x = [i[0] for i in c[:]]
y = [i[1] for i in c[:]]
z = [i[0] for i in a]


# plotting
resolution = 300
xi = np.linspace(min(x), max(x), resolution)
yi = np.linspace(min(y), max(y), resolution)

# Perform linear interpolation of data
triang = tri.Triangulation(x, y)
interpolator = tri.LinearTriInterpolator(triang, z)
Xi, Yi = np.meshgrid(xi, yi)
ai = interpolator(Xi, Yi)


fig = plt.figure()
ax = fig.add_subplot(111)

ax.contourf(xi, yi, ai, levels=64)
ax.axis('equal')
plt.show()
