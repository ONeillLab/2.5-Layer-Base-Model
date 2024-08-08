import numpy as np
import matplotlib.pyplot as plt
from name_list_general import *
import helper_functions_MPI as hf
from netCDF4 import Dataset


def zonal_winds(files, layer, reverse, rdist):
    rootgroups = []
    for file in files:
        rootgroups.append(Dataset(file, "r"))
        
    if layer == 1:
        u = np.array(rootgroups[0].variables['u1mat'])
        for i in range(1,len(rootgroups)):
            data = np.array(rootgroups[i].variables['u1mat'])

            u = np.append(u, data[1:], axis=0)
        
        
        v = np.array(rootgroups[0].variables['v1mat'])
        for i in range(1,len(rootgroups)):
            data = np.array(rootgroups[i].variables['v1mat'])

            v = np.append(v, data[1:], axis=0)
    
    if layer == 2:
        u = np.array(rootgroups[0].variables['u2mat'])
        for i in range(1,len(rootgroups)):
            data = np.array(rootgroups[i].variables['u2mat'])

            u = np.append(u, data[1:], axis=0)


        v = np.array(rootgroups[0].variables['v2mat'])
        for i in range(1,len(rootgroups)):
            data = np.array(rootgroups[i].variables['v2mat'])
            
            v = np.append(v, data[1:], axis=0)
    
    x, y = np.meshgrid(np.arange(0, N) * dx - L / 2, np.arange(0, N) * dx - L / 2)

    zonal = 1/np.sqrt(x**2 + y**2) * (-y*u + x*v)
    
    
    rdist = np.round(rdist, 1)
    dists = np.unique(rdist)
    
    zonals = []
    mean_zonals = []
    for i in range(reverse):
        mean_zonal = []
        for dist in dists:
            inds = np.array(list(zip(*np.where(rdist == dist))))

            zonals.append(zonal[zonal.shape[0]-i-1, inds[:,0],inds[:,1]])

            mean_winds = np.mean(zonal[zonal.shape[0]-i-1, inds[:,0],inds[:,1]])

            mean_zonal.append(mean_winds)
        
        mean_zonals.append(mean_zonal)
    
    return zonal, mean_zonals, zonals, dists



def fourier_coeff(zonal, inds):

    poss = inds - N/2

    thetas = []
    for pos in poss:
        thetas.append(np.arctan2(pos[1], pos[0]))


    ns = np.arange(0,20)
    cs = []
    for n in ns:
        first = 0
        second = 0
        for i in range(len(inds)):
            first += zonal[-1][inds[i][0]][inds[i][1]] * np.cos(n*thetas[i])
            second += zonal[-1][inds[i][0]][inds[i][1]] * np.sin(n*thetas[i])


        cs.append( 1/(4*np.pi**2) * (first**2 + second**2) )

    return cs


layer = 2
data = ['testing_1.nc']

zonal, mean_zonals, zonals, dists = zonal_winds(data, layer, 10, rdist)


rdist = np.round(rdist, 0).astype('int')
dists = np.unique(rdist)



allcs = []
for dist in dists:
    inds = np.array(list(zip(*np.where(rdist == dist))))
    cs = fourier_coeff(zonal, inds)

    allcs.append(cs)

allcs = np.array(allcs)

print(allcs.shape[1])

fig, axs = plt.subplots(1, 2, layout='constrained')
ax = axs[0]
x = np.arange(0,allcs.shape[1])
y = dists
im = ax.pcolormesh(x, y, allcs, cmap='hot', shading='nearest')
ax.set_xlabel("wavenumber")
ax.set_ylabel("Radial distance [LD2]")
tx = ax.set_title("Wavenumer vs radial distance")
fig.colorbar(im)


x, y = np.meshgrid(np.arange(0, N) * dx - L / 2, np.arange(0, N) * dx - L / 2)

ax2 = axs[1]
im2 = ax2.pcolormesh(x,y, zonal[-1], cmap='hot', shading='auto')
ax2.set_xlabel("x position [LD2]")
ax2.set_ylabel("y position [LD2]")
ax2.set_title("Zonal winds at last timestep")
fig.colorbar(im2)

plt.show()
"""
for i in range(len(inds)):
    zonal[-1][inds[i][0]][inds[i][1]] = 100

plt.imshow(zonal[-1])
plt.show()
"""