import numpy as np
import math
import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from numba import jit, objmode, threading_layer, config
import psutil
from netCDF4 import Dataset
from name_list import *


def animate_data(data_name, element):
    """
    Animates data from the file data_name

    data_name is name of the file e.g., "data.nc"

    element is either "u1mat","u2mat","v1mat","v2mat","h1mat","h2mat", "zeta1", "zeta2"
    """   
    if element == "zeta1":
        rootgroup = Dataset(data_name, 'r')
        u1mat = np.array(rootgroup.variables["u1mat"])
        v1mat = np.array(rootgroup.variables["v1mat"])
        
        zeta1 = (1 / dx) * (v1mat[:] - v1mat[:,:,l] + u1mat[:,l,:] - u1mat[:])
        frames = zeta1
        frameslen = int(frames.shape[0])

    elif element == "zeta2":
        rootgroup = Dataset(data_name, 'r')
        u2mat = np.array(rootgroup.variables["u2mat"])
        v2mat = np.array(rootgroup.variables["v2mat"])
        
        zeta2 = (1 / dx) * (v2mat[:] - v2mat[:,:,l] + u2mat[:,l,:] - u2mat[:])
        frames = zeta2
        frameslen = int(frames.shape[0])

    else:
        rootgroup = Dataset(data_name, "r") 
        elementmat = rootgroup.variables[element]
        frames = np.array(elementmat)
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
        arr = np.array(frames[i])
        vmax = np.max(arr)
        vmin = np.min(arr)
        im.set_data(arr)
    
    ani = animation.FuncAnimation(fig, animate, interval=ani_interval, frames=frameslen)
    plt.show()
    rootgroup.close()


def animate_multiple(files, element):
    """
    Takes multple files (i.e files = ['data1.nc', 'data2.nc', ...]) and produces a single smooth animation 
    """
    rootgroup = Dataset(data_name1, "r") 
    rootgroup2 = Dataset(data_name2, "r") 
    elementmat1 = np.asarray(rootgroup.variables[element])
    elementmat2 = np.asarray(rootgroup2.variables[element])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    frames = np.append(elementmat1, elementmat2, axis=0)
    frameslen = int(len(frames))
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

    ani = animation.FuncAnimation(fig, animate, interval=ani_interval, frames=frameslen)
    plt.show()
    rootgroup.close()
    rootgroup2.close()



#### examples ####
#animate_two("run1.nc", "run1continued.nc", "u1mat")

#display_data("data2.nc")
#animate_data("data2.nc", "zeta2")
#animate_data("data2.nc", "u1mat")






#animate_zeta("data.nc", 2)
#last_timestep("data.nc")
#u1, u2, v1, v2, h1, h2, locs, lasttime = last_timestep("data2.nc")