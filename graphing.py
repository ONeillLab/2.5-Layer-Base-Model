import numpy as np
import math
import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.axes_grid1 import make_axes_locatable
from netCDF4 import Dataset
from name_list_uranus import *


def animate(files, element):
    """
    Takes multple files (i.e files = ['data1.nc', 'data2.nc', ...]) and produces a single smooth animation 
    This has taken the place of all the animate, animate_zeta1/zeta2, and animate_two. Since it will do all those
    
    possible elements are: "u1mat","u2mat","v1mat","v2mat","h1mat","h2mat", "zeta1", "zeta2"

    Note the zetas are relative vorticity.
    """
    rootgroups = []
    for file in files:
        set = Dataset(file, "r")
        rootgroups.append(set)
        print(set.__dict__["time"])

    if element == "zeta1":
        u1mat = rootgroups[0].variables['u1mat']
        v1mat = rootgroups[0].variables['v1mat']
        for i in range(1,len(rootgroups)):
            u1 = np.array(rootgroups[i].variables["u1mat"]) 
            v1 = np.array(rootgroups[i].variables["v1mat"])

            u1mat = np.append(u1mat, u1, axis=0)
            v1mat = np.append(v1mat, v1, axis=0)

        zeta1 = (1 / dx) * (v1mat[:] - v1mat[:,:,lg] + u1mat[:,lg,:] - u1mat[:])
        frames = zeta1
    
    elif element == "zeta2":
        u2mat = rootgroups[0].variables['u2mat']
        v2mat = rootgroups[0].variables['v2mat']
        for i in range(1,len(rootgroups)):
            u2 = np.array(rootgroups[i].variables["u2mat"]) 
            v2 = np.array(rootgroups[i].variables["v2mat"])

            u2mat = np.append(u2mat, u2, axis=0)
            v2mat = np.append(v2mat, v2, axis=0)
        #print(N)
        #print(np.shape(rdist))
        #print(np.shape(1 - Bt + (1/dx) * (v2mat[1] - np.roll(v2mat[1], 1, axis=1) + np.roll(u2mat[1], 1, axis=0) - u2mat[1])))
        zeta2 = (1 / dx) * (v2mat[:] - v2mat[:,:,lg] + u2mat[:,lg,:] - u2mat[:])
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

    print("animating")
    ani = animation.FuncAnimation(fig, animate, interval=ani_interval, frames=frameslen)
    plt.show()
    
    for group in rootgroups:
        group.close()

def view_slice(data, slice):
    rootgroup = Dataset(data, "r")
    u1 = rootgroup.variables['u1mat'][slice]
    u2 = rootgroup.variables['u2mat'][slice]
    v1 = rootgroup.variables['v1mat'][slice]
    v2 = rootgroup.variables['v2mat'][slice]
    h1 = rootgroup.variables['h1mat'][slice]
    h2 = rootgroup.variables['h2mat'][slice]
    zeta1 = (1 / dx) * (v1 - v1[:,l] + u1[l,:] - u1) #1 - Bt * rdist**2 +
    zeta2 = (1 / dx) * (v2 - v2[:,l] + u2[l,:] - u2) 

    fig, axs = plt.subplots(2, 4, constrained_layout=True)
    u1_plot = axs[0,0]
    u2_plot = axs[0,1]
    v1_plot = axs[0,2]
    v2_plot = axs[0,3]
    h1_plot = axs[1,0]
    h2_plot = axs[1,1]
    zeta1_plot = axs[1,2]
    zeta2_plot = axs[1,3]

    im1 = u1_plot.imshow(u1, cmap="bwr")
    im2 = u2_plot.imshow(u2, cmap="bwr")
    im3 = v1_plot.imshow(v1, cmap="bwr")
    im4 = v2_plot.imshow(v2, cmap="bwr")
    im5 = h1_plot.imshow(h1, cmap="bwr")
    im6 = h2_plot.imshow(h2, cmap="bwr")
    im7 = zeta1_plot.imshow(zeta1, cmap="bwr")
    im8 = zeta2_plot.imshow(zeta2, cmap="bwr")

    fig.colorbar(im1, ax=u1_plot)
    fig.colorbar(im2, ax=u2_plot)
    fig.colorbar(im3, ax=v1_plot)
    fig.colorbar(im4, ax=v2_plot)
    fig.colorbar(im5, ax=h1_plot)
    fig.colorbar(im6, ax=h2_plot)
    fig.colorbar(im7, ax=zeta1_plot)
    fig.colorbar(im8, ax=zeta2_plot)

    u1_plot.set_title("u1")
    u2_plot.set_title("u2")
    v1_plot.set_title("v1")
    v2_plot.set_title("v2")
    h1_plot.set_title("h1")
    h2_plot.set_title("h2")
    zeta1_plot.set_title("zeta1")
    zeta2_plot.set_title("zeta2")

    #fig.tight_layout()
    plt.show()
    rootgroup.close()



#### examples ####
data = ['testing_1.nc']

animate(data, "zeta2")


"""
root = Dataset(data[0], 'r')

h2mat = np.array(root.variables['h2mat'])

root.close()

plt.imshow(h2mat[0])
plt.colorbar()
plt.show()
"""
