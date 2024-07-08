import numpy as np
import math
import helper_functions_MPI as hf
import time
from name_list import *
import psutil
from netCDF4 import Dataset
import access_data as ad
from mpi4py import MPI
import sys
import os


import matplotlib.pyplot as plt


""" 
Initialization of MPI and each thread.

Each u1,u2,v1,... are subdivided on the 0th thread in to even square chunks of side length N/sqrt(num)
where num is the number of threads used (minus the 0th thread as that is the "master" thread). 

Note for this to work the number of threads used has to be a square number and the side length of the domain must have no remainder
with the square root of the number of threads. This is easy to do if everything is in powers of 2, i.e domain size = 1024 and num 
threads = 64, then the domain size on each thread is 128.

"""

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size() - 1


def distribute_domain(domain_size, num_procs):
    subdomain_size = domain_size // np.sqrt(num_procs)
    remainder = domain_size % np.sqrt(num_procs)

    subdomains = [int(subdomain_size + 1) if i < remainder else int(subdomain_size) for i in range(num_procs)]
    return subdomains

subdomains = distribute_domain(N, size)
offset = subdomains[0]

subdomains = np.reshape(subdomains, (int(np.sqrt(size)),int(np.sqrt(size))))
ranks = np.reshape(np.arange(0,size), (int(np.sqrt(size)), int(np.sqrt(size)))) + 1

largeranks = np.zeros((6, 6), dtype=ranks.dtype)

# Step 3: Use nested loops to copy elements with periodic boundary conditions
for i in range(6):
    for j in range(6):
        largeranks[i, j] = ranks[i % 2, j % 2]

if rank == 0:
    if restart_name == None:
        #ad.create_file(new_name)
        lasttime = 0
        locs = hf.genlocs(num, N, 0) ### use genlocs instead of paircount
    else:
        ad.create_file(new_name)
        u1, u2, v1, v2, h1, h2, locs, lasttime = ad.last_timestep(restart_name)

    wlayer = hf.pairshapeN2(locs, 0)
    Wmat = hf.pairfieldN2(L, h1, wlayer)

    WmatSplit = [Wmat]
    u1matSplit = [u1]
    v1matSplit = [v1]
    u2matSplit = [u2]
    v2matSplit = [v2]
    h1matSplit = [h1]
    h2matSplit = [h2]

    spdrag1Split = [spdrag1]
    spdrag2Split = [spdrag2]
    rdistSplit = [rdist]

    for i in range(1,size+1):
        WmatSplit.append(hf.split(Wmat, offset, ranks, i))
        u1matSplit.append(hf.split(u1, offset, ranks, i))
        v1matSplit.append(hf.split(v1, offset, ranks, i))
        u2matSplit.append(hf.split(u2, offset, ranks, i))
        v2matSplit.append(hf.split(v2, offset, ranks, i))
        h1matSplit.append(hf.split(h1, offset, ranks, i))
        h2matSplit.append(hf.split(h2, offset, ranks, i))

        spdrag1Split.append(hf.split(spdrag1, offset, ranks, i))
        spdrag2Split.append(hf.split(spdrag2, offset, ranks, i))
        rdistSplit.append(hf.split(rdist, offset, ranks, i))

else:
    WmatSplit = None
    u1matSplit = None
    v1matSplit = None
    u2matSplit = None
    v2matSplit = None
    h1matSplit = None
    h2matSplit = None

    spdrag1Split = None
    spdrag2Split = None
    rdistSplit = None

    u1 = None
    u2 = None
    v1 = None
    v2 = None
    h1 = None
    h2 = None
    Wmat = None
    lasttime = None

    spdrag1 = None
    spdrag2 = None
    rdist = None

Wmat = np.ascontiguousarray(comm.scatter(WmatSplit, root=0))
u1 = np.ascontiguousarray(comm.scatter(u1matSplit, root=0))
u2 = np.ascontiguousarray(comm.scatter(u2matSplit, root=0))
v1 = np.ascontiguousarray(comm.scatter(v1matSplit, root=0))
v2 = np.ascontiguousarray(comm.scatter(v2matSplit, root=0))
h1 = np.ascontiguousarray(comm.scatter(h1matSplit, root=0))
h2 = np.ascontiguousarray(comm.scatter(h2matSplit, root=0))
spdrag1 = np.ascontiguousarray(comm.scatter(spdrag1Split, root=0))
spdrag2 = np.ascontiguousarray(comm.scatter(spdrag2Split, root=0))
rdist = np.ascontiguousarray(comm.scatter(rdistSplit, root=0))
lasttime = comm.bcast(lasttime, root=0)


#print(f"Rank: {rank}, Shape: {Wmat.shape}")

### END OF INITIALIZATION ###


"""
This is the start of time stepping.

Each thread solves the 2.5 layer shallow water equations on their own grid cell and then sends boundary data to its 
neighbouring cells so that there are no weird boundaries

The time step function...

This function takes in each threads u1,u2,... and then solves the shallow water equations in them. It is the exact 
same code as in other files, just wrapped in a function instead of a while loop. 
"""
def timestep(u1,u2,v1,v2,h1,h2,Wmat, u1_p,u2_p,v1_p,v2_p,h1_p,h2_p):
    broke = False
    if AB == 2:
        tmp = u1.copy()
        u1 = 1.5 * u1 - 0.5 * u1_p
        u1_p = tmp  #
        tmp = u2.copy()
        u2 = 1.5 * u2 - 0.5 * u2_p
        u2_p = tmp  #
        tmp = v1.copy()
        v1 = 1.5 * v1 - 0.5 * v1_p
        v1_p = tmp
        tmp = v2.copy()
        v2 = 1.5 * v2 - 0.5 * v2_p
        v2_p = tmp
        tmp = h1.copy()
        h1 = 1.5 * h1 - 0.5 * h1_p
        h1_p = tmp
        if layers == 2.5:
            tmp = h2.copy()
            h2 = 1.5 * h2 - 0.5 * h2_p
            h2_p = tmp

    # add friction
    du1dt = hf.viscND(u1, Re, n)
    du2dt = hf.viscND(u2, Re, n)
    dv1dt = hf.viscND(v1, Re, n)
    dv2dt = hf.viscND(v2, Re, n)

    if spongedrag1 > 0:
        du1dt = du1dt - spdrag1 * (u1)
        du2dt = du2dt - spdrag2 * (u2)
        dv1dt = dv1dt - spdrag1 * (v1)
        dv2dt = dv2dt - spdrag2 * (v2)

    # absolute vorticity
    zeta1 = 1 - Bt * rdist**2 + (1 / dx) * (v1 - v1[:,l] + u1[l,:] - u1)
    
    zeta2 = 1 - Bt * rdist**2 + (1 / dx) * (v2 - v2[:,l] + u2[l,:] - u2)


    # add vorticity flux, zeta*u
    zv1 = zeta1 * (v1 + v1[:,l])
    zv2 = zeta2 * (v2 + v2[:,l])

    du1dt = du1dt + 0.25 * (zv1 + zv1[r,:])
    du2dt = du2dt + 0.25 * (zv2 + zv2[r,:])

    zu1 = zeta1 * (u1 + u1[l,:])
    zu2 = zeta2 * (u2 + u2[l,:])

    dv1dt = dv1dt - 0.25 * (zu1 + zu1[:,r])
    dv2dt = dv2dt - 0.25 * (zu2 + zu2[:,r])

    ### Cumulus Drag (D) ###
    du1dt = du1dt - (1 / dx) * u1 / dragf
    du2dt = du2dt - (1 / dx) * u2 / dragf
    dv1dt = dv1dt - (1 / dx) * v1 / dragf
    dv2dt = dv2dt - (1 / dx) * v2 / dragf

    B1p, B2p = hf.BernN2(u1, v1, u2, v2, gm, c22h, c12h, h1, h2, ord)

    du1dtsq = du1dt - (1 / dx) * (B1p - B1p[:,l])
    du2dtsq = du2dt - (1 / dx) * (B2p - B2p[:,l])

    dv1dtsq = dv1dt - (1 / dx) * (B1p - B1p[l,:])
    dv2dtsq = dv2dt - (1 / dx) * (B2p - B2p[l,:])

    if AB == 2:
        u1sq = u1_p + dt * du1dtsq
        u2sq = u2_p + dt * du2dtsq

        v1sq = v1_p + dt * dv1dtsq
        v2sq = v2_p + dt * dv2dtsq


    Fx1 = hf.xflux(h1, u1) - kappa / dx * (h1 - h1[:,l])
    Fy1 = hf.yflux(h1, v1) - kappa / dx * (h1 - h1[l,:])
    dh1dt = -(1 / dx) * (Fx1[:,r] - Fx1 + Fy1[r,:] - Fy1)

    if layers == 2.5:
        Fx2 = hf.xflux(h2, u2) - kappa / dx * (h2 - h2[:,l])
        Fy2 = hf.yflux(h2, v2) - kappa / dx * (h2 - h2[l,:])

        dh2dt = -(1 / dx) * (Fx2[:,r] - Fx2 + Fy2[r,:] - Fy2)

    if tradf > 0:
        dh1dt = dh1dt - 1 / tradf * (h1 - 1)
        dh2dt = dh2dt - 1 / tradf * (h2 - 1)

    if mode == 1:
        dh1dt = dh1dt + Wmat.astype(np.float64)
        if layers == 2.5:
            dh2dt = dh2dt - H1H2 * Wmat.astype(np.float64)

    if AB == 2:
        h1 = h1_p + dt * dh1dt
        if layers == 2.5:
            h2 = h2_p + dt * dh2dt

    u1 = u1sq
    u2 = u2sq
    v1 = v1sq
    v2 = v2sq

    if math.isnan(h1[0, 0]):
        print(f"Rank: {rank}, h1 is nan")
        broke = True

    return u1,u2,v1,v2,h1,h2,u1_p,u2_p,v1_p,v2_p,h1_p,h2_p, broke



"""
This is the start of the simulation

"""

mode = 1

# TIME STEPPING
if AB == 2:
    u1_p = u1.copy()
    v1_p = v1.copy()
    h1_p = h1.copy()
    u2_p = u2.copy()
    v2_p = v2.copy()
    h2_p = h2.copy()


ts = []
ii = 0

t = lasttime
tc = round(t/dt)

rem = False

tottimer = time.time()
print("Starting simulation")

sendingTimes = []
simTimes = []
zeroTimes = []
broke = False

while t <= tmax + lasttime + dt / 2:

    ### Running of the simulation on all ranks but the master rank (0) ###

    timer = time.time()

    if rank != 0:
        u1,u2,v1,v2,h1,h2, u1_p,u2_p,v1_p,v2_p,h1_p,h2_p, broke = timestep(u1,u2,v1,v2,h1,h2,Wmat, u1_p,u2_p,v1_p,v2_p,h1_p,h2_p)

    if broke == True:
        MPI.Finalize()

    simTimes.append(time.time()-timer)


    ### Sending boundary conditions to neighbouring cells

    sendtimer = time.time()
    
    if rank != 0:
        ind = np.where(ranks == rank)
        i = ind[0][0] + 2
        j = ind[1][0] + 2
        sendranks, recvranks = hf.get_surrounding_points(largeranks, i, j)
        print(f"rank: {rank}, {sendranks}")

        
        for sendrank in sendranks:
            if (sendrank[0], sendrank[1]) == (-1,-1):
                comm.send(u1[2:4,:][:,2:4], dest=sendrank[2], tag=0)
                comm.send(u2[2:4,:][:,2:4], dest=sendrank[2], tag=1)
                comm.send(v1[2:4,:][:,2:4], dest=sendrank[2], tag=2)
                comm.send(v2[2:4,:][:,2:4], dest=sendrank[2], tag=3)
                comm.send(h1[2:4,:][:,2:4], dest=sendrank[2], tag=4)
                comm.send(h2[2:4,:][:,2:4], dest=sendrank[2], tag=5) #recvbuf=u1[offset+2:offset+4,:][:,offset+2:offset+4], source=rank, recvtag=5)

            if (sendrank[0], sendrank[1]) == (-1,0):
                comm.send(u1[2:4,:][:,2:offset+2], dest=sendrank[2], tag=10)
                comm.send(u2[2:4,:][:,2:offset+2], dest=sendrank[2], tag=11)
                comm.send(v1[2:4,:][:,2:offset+2], dest=sendrank[2], tag=12)
                comm.send(v2[2:4,:][:,2:offset+2], dest=sendrank[2], tag=13)
                comm.send(h1[2:4,:][:,2:offset+2], dest=sendrank[2], tag=14)
                comm.send(h2[2:4,:][:,2:offset+2], dest=sendrank[2], tag=15) #recvbuf=h2[offset+2:offset+4,:][:,2:offset+2])

            if (sendrank[0], sendrank[1]) == (-1,1):
                comm.send(u1[2:4,:][:,offset:offset+2], dest=sendrank[2], tag=20)
                comm.send(u2[2:4,:][:,offset:offset+2], dest=sendrank[2], tag=21)
                comm.send(v1[2:4,:][:,offset:offset+2], dest=sendrank[2], tag=22)
                comm.send(v2[2:4,:][:,offset:offset+2], dest=sendrank[2], tag=23)
                comm.send(h1[2:4,:][:,offset:offset+2], dest=sendrank[2], tag=24)
                comm.send(h2[2:4,:][:,offset:offset+2], dest=sendrank[2], tag=25) #recvbuf=h2[offset+2:offset+4,:][:,0:2])

            if (sendrank[0], sendrank[1]) == (0,-1):
                comm.send(u1[2:offset+2,:][:,2:4], dest=sendrank[2], tag=30)
                comm.send(u2[2:offset+2,:][:,2:4], dest=sendrank[2], tag=31)
                comm.send(v1[2:offset+2,:][:,2:4], dest=sendrank[2], tag=32)
                comm.send(v2[2:offset+2,:][:,2:4], dest=sendrank[2], tag=33)
                comm.send(h1[2:offset+2,:][:,2:4], dest=sendrank[2], tag=34)
                comm.send(h2[2:offset+2,:][:,2:4], dest=sendrank[2], tag=35) #recvbuf=h2[2:offset+2,:][:,offset+2:offset+4])

            if (sendrank[0], sendrank[1]) == (0,1):
                comm.send(u1[2:offset+2,:][:,offset:offset+2], dest=sendrank[2], tag=40)
                comm.send(u2[2:offset+2,:][:,offset:offset+2], dest=sendrank[2], tag=41)
                comm.send(v1[2:offset+2,:][:,offset:offset+2], dest=sendrank[2], tag=42)
                comm.send(v2[2:offset+2,:][:,offset:offset+2], dest=sendrank[2], tag=43)
                comm.send(h1[2:offset+2,:][:,offset:offset+2], dest=sendrank[2], tag=44)
                comm.send(h2[2:offset+2,:][:,offset:offset+2], dest=sendrank[2], tag=45) #recvbuf=h2[2:offset+2,:][:,0:2])

            if (sendrank[0], sendrank[1]) == (1,-1):
                comm.send(u1[offset:offset+2,:][:,2:4], dest=sendrank[2], tag=50)
                comm.send(u2[offset:offset+2,:][:,2:4], dest=sendrank[2], tag=51)
                comm.send(v1[offset:offset+2,:][:,2:4], dest=sendrank[2], tag=52)
                comm.send(v2[offset:offset+2,:][:,2:4], dest=sendrank[2], tag=53)
                comm.send(h1[offset:offset+2,:][:,2:4], dest=sendrank[2], tag=54)
                comm.send(h2[offset:offset+2,:][:,2:4], dest=sendrank[2], tag=55) #recvbuf=h2[0:2,:][:,offset+2:offset+4])

            if (sendrank[0], sendrank[1]) == (1,0):
                comm.send([u1[offset:offset+2,:][:,2:offset+2], u2[offset:offset+2,:][:,2:offset+2], v1[offset:offset+2,:][:,2:offset+2], v2[offset:offset+2,:][:,2:offset+2],
                           h1[offset:offset+2,:][:,2:offset+2], h2[offset:offset+2,:][:,2:offset+2]], dest=sendrank[2], tag=6)
                
                
                comm.send(u2[offset:offset+2,:][:,2:offset+2], dest=sendrank[2], tag=61)
                comm.send(v1[offset:offset+2,:][:,2:offset+2], dest=sendrank[2], tag=62)
                comm.send(v2[offset:offset+2,:][:,2:offset+2], dest=sendrank[2], tag=63)
                comm.send(h1[offset:offset+2,:][:,2:offset+2], dest=sendrank[2], tag=64)
                comm.send(h2[offset:offset+2,:][:,2:offset+2], dest=sendrank[2], tag=65) #recvbuf=h2[0:2,:][:,2:offset+2])
            
            if (sendrank[0], sendrank[1]) == (1,1):
                comm.send(u1[offset:offset+2,:][:,offset:offset+2], dest=sendrank[2], tag=70)
                comm.send(u2[offset:offset+2,:][:,offset:offset+2], dest=sendrank[2], tag=71)
                comm.send(v1[offset:offset+2,:][:,offset:offset+2], dest=sendrank[2], tag=72)
                comm.send(v2[offset:offset+2,:][:,offset:offset+2], dest=sendrank[2], tag=73)
                comm.send(h1[offset:offset+2,:][:,offset:offset+2], dest=sendrank[2], tag=74)
                comm.send(h2[offset:offset+2,:][:,offset:offset+2], dest=sendrank[2], tag=75) #recvbuf=h2[0:2,:][:,0:2])

        for sendrank in sendranks:


    sys.exit()
    sendingTimes.append(time.time()-sendtimer)

    ### Rank 0 checks for if new storms need to be created and sends out the new Wmat ###

    zerotimer = time.time()

    if rank == 0:
        remove_layers = [] # store weather layers that need to be removed here
        rem = False

        if mode == 1:
            for i in range(len(locs)):
                if (t-locs[i][-1]) >= locs[i][3] and t != 0:
                    remove_layers.append(i) # tag layer for removal if a storm's 

            add = len(remove_layers) # number of storms that were removed

            if add != 0:
                newlocs = hf.genlocs(add, N, t)

                for i in range(len(remove_layers)):
                    locs[remove_layers[i]] = newlocs[i]

                wlayer = hf.pairshapeN2(locs, t) ### use pairshapeBEGIN instead of pairshape
                Wmat = hf.pairfieldN2(L, h1, wlayer)

        if len(remove_layers) != 0:
            rem = True
            WmatSplit = [Wmat]

            for i in range(1,size+1):
                WmatSplit.append(hf.split(Wmat, offset, ranks, i))
    
    rem = comm.bcast(rem, root=0)
    if rem == True:
        Wmat = comm.scatter(WmatSplit, root=0)
        rem = False

    zeroTimes.append(time.time()-zerotimer)
    
    tc += 1
    t = tc * dt


### Combining data on rank 0 ###
u1matSplit = comm.gather(u1, root=0)
v1matSplit = comm.gather(v1, root=0)
u2matSplit = comm.gather(u2, root=0)
v2matSplit = comm.gather(v2, root=0)
h1matSplit = comm.gather(h1, root=0)
h2matSplit = comm.gather(h2, root=0)

if rank == 0:
    u1 = hf.combine(u1matSplit, offset, ranks, size)
    u2 = hf.combine(u2matSplit, offset, ranks, size)
    v1 = hf.combine(v1matSplit, offset, ranks, size)
    v2 = hf.combine(v2matSplit, offset, ranks, size)
    h1 = hf.combine(h1matSplit, offset, ranks, size)
    h2 = hf.combine(h2matSplit, offset, ranks, size)

### Post running tests ###

print(f"rank: {rank}, simtime avg: {np.mean(simTimes)}, sendingtime avg: {np.mean(zeroTimes)}, total time: {time.time()-tottimer}, num threads: {os.environ.get('OMP_NUM_THREADS')}")


if rank == 0:
    plt.title(f"Rank: {rank}")
    plt.imshow(h1, cmap='hot')
    plt.colorbar()
    plt.show()

MPI.Finalize()
