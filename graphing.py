import numpy as np
import math
import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from numba import jit, objmode, threading_layer, config
import psutil
from netCDF4 import Dataset
from name_list import *


def animate(files, element):
    """
    Takes multple files (i.e files = ['data1.nc', 'data2.nc', ...]) and produces a single smooth animation 
    This has taken the place of all the animate, animate_zeta1/zeta2, and animate_two. Since it will do all those
    
    possible elements are: "u1mat","u2mat","v1mat","v2mat","h1mat","h2mat", "zeta1", "zeta2"

    Note the zetas are relative vorticity.
    """
    rootgroups = []
    for file in files:
        rootgroups.append(Dataset(file, "r"))

    if element == "zeta1":
        u1mat = rootgroups[0].variables['u1mat']
        v1mat = rootgroups[0].variables['v1mat']
        for i in range(1,len(rootgroups)):
            u1 = np.array(rootgroups[i].variables["u1mat"]) 
            v1 = np.array(rootgroups[i].variables["v1mat"])

            u1mat = np.append(u1mat, u1, axis=0)
            v1mat = np.append(v1mat, v1, axis=0)

        zeta1 = (1 / dx) * (v1mat[:] - v1mat[:,:,l] + u1mat[:,l,:] - u1mat[:])
        frames = zeta1
    
    elif element == "zeta2":
        u2mat = rootgroups[0].variables['u2mat']
        v2mat = rootgroups[0].variables['v2mat']
        for i in range(1,len(rootgroups)):
            u2 = np.array(rootgroups[i].variables["u2mat"]) 
            v2 = np.array(rootgroups[i].variables["v2mat"])

            u2mat = np.append(u2mat, u2, axis=0)
            v2mat = np.append(v2mat, v2, axis=0)
            
        zeta2 = (1 / dx) * (v2mat[:] - v2mat[:,:,l] + u2mat[:,l,:] - u2mat[:])
        frames = zeta2

    else:
        frames = rootgroups[0].variables[element]
        for i in range(1,len(rootgroups)):
            data = np.array(rootgroups[i].variables[element])

            frames = np.append(frames, data, axis=0)
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
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
    tx = ax.set_title(f"time: {0}")

    def animate(i):
        arr = frames[i] #frames[7 * i + (i - 1)]
        vmax = np.max(arr)
        vmin = np.min(arr)
        tx.set_text(f"time: {i}")
        im.set_data(arr)

    ani = animation.FuncAnimation(fig, animate, interval=ani_interval, frames=frameslen)
    plt.show()
    
    ani.save("Sponge0_1.mp4")

    for group in rootgroups:
        group.close()



#### examples ####
data = ['Sponge0_1.nc']

animate(data, "zeta2")