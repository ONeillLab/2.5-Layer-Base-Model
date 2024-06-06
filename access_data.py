import numpy as np
import math
import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from numba import jit, objmode, threading_layer, config
import psutil
from netCDF4 import Dataset

#### read data from netcdf file ####

rootgroup = Dataset("data.nc", "r")
print(rootgroup)
print(rootgroup.dimensions)
print(rootgroup.ncattrs)
print(rootgroup.variables)
zeta2mat = rootgroup.variables["zeta2mat"]

#### animate data ####

frames = zeta2mat
frameslen = int(
    rootgroup.variables["u2mat"].shape[0] / 8
)  # accounts for a bug; fix later

fig = plt.figure()
ax = fig.add_subplot(111)


cv0 = frames[0]
vminlist = []
vmaxlist = []
for j in frames:
    if math.isnan(j[0, 0]) == False:
        vminlist.append(np.min(j))
        vmaxlist.append(np.max(j))
vmin = np.min(vminlist)
vmax = np.max(vmaxlist)
im = ax.imshow(cv0, cmap="bwr", vmin=vmin, vmax=vmax)
cb = fig.colorbar(im)


def animate(i):
    arr = frames[7 * i + (i - 1)]
    vmax = np.max(arr)
    vmin = np.min(arr)
    im.set_data(arr)


ani = animation.FuncAnimation(fig, animate, interval=100, frames=frameslen)
plt.show()


rootgroup.close()
