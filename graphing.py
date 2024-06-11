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

    element is either "u1mat","u2mat","v1mat","v2mat","h1mat","h2mat"
    """   
    rootgroup = Dataset(data_name, "r") 
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
    
    ani = animation.FuncAnimation(fig, animate, interval=ani_interval, frames=frameslen)
    plt.show()
    rootgroup.close()


def animate_zeta(data_name, i):
    """
    Animates zeta1 or zeta2
    """
    rootgroup = Dataset(data_name, "r") 
    umat = np.asarray(rootgroup.variables[f'u{i}mat'])
    vmat = np.asarray(rootgroup.variables[f'v{i}mat'])
    rootgroup.close()
    zetaimat = []
    for j in range(len(umat)):
        zetai = 1 - Bt * rdist**2 + (1 / dx) * (vmat[j] - vmat[j][:,l] + umat[j][l,:] - umat[j])
        zeta = zetai
        zetaimat.append(zeta)
    frames = zetaimat
    frameslen = len(zetaimat)

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
    
    ani = animation.FuncAnimation(fig, animate, interval=ani_interval, frames=frameslen)
    plt.show()
    return zetaimat


def animate_two(data_name1, data_name2, element):
    """
    Takes two files and produces a single smooth animation 
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
#animate_data("data.nc", "u1mat")
#animate_data("data2.nc", "u1mat")






#animate_zeta("data.nc", 2)
#last_timestep("data.nc")
#u1, u2, v1, v2, h1, h2, locs, lasttime = last_timestep("data2.nc")