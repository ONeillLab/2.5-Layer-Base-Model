import numpy as np
import math
import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from numba import jit, objmode, threading_layer, config
import psutil
from netCDF4 import Dataset

#### read data from netcdf file ####

def display_data(data_name):
    rootgroup = Dataset("data.nc", "r")
    print(rootgroup)
    print(rootgroup.dimensions)
    print(rootgroup.ncattrs)
    print(rootgroup.variables)
    rootgroup.close()

#### extract data ####

def last_timestep(data_name):
    rootgroup = Dataset("data.nc", "r")
    u1 = np.asarray(rootgroup.variables["u1mat"][-1])
    u2 = np.asarray(rootgroup.variables["u2mat"][-1])
    v1 = np.asarray(rootgroup.variables["v1mat"][-1])
    v2 = np.asarray(rootgroup.variables["v2mat"][-1])
    h1 = np.asarray(rootgroup.variables["h1mat"][-1])
    h2 = np.asarray(rootgroup.variables["h2mat"][-1])
    locs = np.asarray(rootgroup.variables["locsmat"][-1])
    time = rootgroup.__dict__["time"]
    rootgroup.close()
    return u1, u2, v1, v2, h1, h2, locs, time

#### animate data ####

def animate_data(data_name, element):   # data_name is name of the file e.g., "data.nc"
    rootgroup = Dataset("data.nc", "r") # element is either "u1mat","u2mat","v1mat","v2mat","h1mat","h2mat","Wmatmat"
    elementmat = rootgroup.variables[element]
    frames = elementmat
    frameslen = int(elementmat.shape[0])

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
        arr = frames[i] #frames[7 * i + (i - 1)]
        vmax = np.max(arr)
        vmin = np.min(arr)
        im.set_data(arr)
    
    ani = animation.FuncAnimation(fig, animate, interval=100, frames=frameslen)
    plt.show()
    rootgroup.close()

#### examples ####

#display_data("data.nc")
#animate_data("data.nc", "h1mat")

