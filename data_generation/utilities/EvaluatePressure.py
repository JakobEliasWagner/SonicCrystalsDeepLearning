import h5py
from scipy.integrate import simps
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-1, 1, 100)
y = np.linspace(-1, 1, 100)
z = x**2 + y ** 2

inte = simps([simps(zz_x,x) for zz_x in zz],y)
print(inte)

exit()
# get data
f = h5py.File('sol.h5', 'r')

a = f['Function']['real_f_6']['0']  # Werte
b = f['Mesh']['mesh']['topology']
c = f['Mesh']['mesh']['geometry']  # Koordinaten pro eintrag xy
x = [i[0] for i in c[:]]
y = [i[1] for i in c[:]]
z = [i[0] for i in a]





