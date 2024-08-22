import numpy as np
import math
import helper_functions_MPI as hf
import time
from name_list_jupiter import *
from netCDF4 import Dataset
import access_data as ad
from mpi4py import MPI
import psutil


def rss():
    return psutil.Process().memory_info().rss

""" 
Initialization of MPI and each thread.

Each u1,u2,v1,... are subdivided on the 0th thread in to even square chunks of side length N/sqrt(num)
where num is the number of threads used (minus the 0th thread as that is the "master" thread). 

Note for this to work the number of threads used has to be a square number and the side length of the domain must have no remainder
with the square root of the number of threads. This is easy to do if everything is in powers of 2, i.e domain size = 1024 and num 
threads = 64, then the domain size on each thread is 128.


This program uses rank 0 to compute all "overarching" (needed by all process) variables, along with performing all data saving.
The other ranks are worker ranks, and solve the shallow-water equations on each of their subarray (subdomains). They recieve a subdomain
with padding so that they can solve ODEs on them, this padding is updated at the end of each timestep.

"""

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size() - 1


# Calculating the size of the subdomains given to each process
def distribute_domain(domain_size, num_procs):
    subdomain_size = domain_size // np.sqrt(num_procs)
    remainder = domain_size % np.sqrt(num_procs)

    subdomains = [int(subdomain_size + 1) if i < remainder else int(subdomain_size) for i in range(num_procs)]
    return subdomains

# assign each process a subdomain.
subdomains = distribute_domain(N, size)
offset = subdomains[0]

# Create an array containing which rank has which location of the larger array. This is the ranks array.
subdomains = np.reshape(subdomains, (int(np.sqrt(size)),int(np.sqrt(size))))
ranks = np.reshape(np.arange(0,size), (int(np.sqrt(size)), int(np.sqrt(size)))) + 1

# Make the ranks array periodic for later calculations.
largeranks = np.zeros((int(3*np.sqrt(size)), int(3*np.sqrt(size))), dtype=ranks.dtype)

for i in range(int(3*np.sqrt(size))):
    for j in range(int(3*np.sqrt(size))):
        largeranks[i, j] = ranks[i % int(np.sqrt(size)), j % int(np.sqrt(size))]


# Perform data loading/Saving on rank 0. Also distrubution of initial data and storm locations.
if rank == 0:
    
    # Loads initial data or creates new data file.
    if restart_name == None:
        if saving == True:
            ad.create_file(new_name)
        lasttime = 0
        locs = hf.genlocs(num, N, 0) ### use genlocs instead of paircount
    else:
        ad.create_file(new_name)
        u1, u2, v1, v2, h1, h2, locs, lasttime = ad.last_timestep(restart_name)

    # Initialize dynamical variable arrays before splitting then into subarrays
    Wmat = None
    WmatSplit = None
    u1matSplit = [u1]
    v1matSplit = [v1]
    u2matSplit = [u2]
    v2matSplit = [v2]
    h1matSplit = [h1]
    h2matSplit = [h2]

    spdrag1Split = [spdrag1]
    spdrag2Split = [spdrag2]
    rdistSplit = [rdist]
    xSplit = [x]
    ySplit = [y]


    # Split each dynamical array into subarrays, and append the subarray to the split list. These will be distributed to each process.
    for i in range(1,size+1):
        #WmatSplit.append(hf.split(Wmat, offset, ranks, i))
        u1matSplit.append(hf.split(u1, offset, ranks, i))
        v1matSplit.append(hf.split(v1, offset, ranks, i))
        u2matSplit.append(hf.split(u2, offset, ranks, i))
        v2matSplit.append(hf.split(v2, offset, ranks, i))
        h1matSplit.append(hf.split(h1, offset, ranks, i))
        h2matSplit.append(hf.split(h2, offset, ranks, i))

        spdrag1Split.append(hf.split(spdrag1, offset, ranks, i))
        spdrag2Split.append(hf.split(spdrag2, offset, ranks, i))
        rdistSplit.append(hf.split(rdist, offset, ranks, i))
        xSplit.append(hf.split(x, offset, ranks, i))
        ySplit.append(hf.split(y, offset, ranks, i))

else:
    # Initialize empty data for each process, which will be filled by the split arrays computed on rank 0.
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
    xSplit = None
    ySplit = None

    u1 = None
    u2 = None
    v1 = None
    v2 = None
    h1 = None
    h2 = None
    Wmat = None
    wcorrect = None
    lasttime = None

    spdrag1 = None
    spdrag2 = None
    rdist = None
    x = None
    y = None
    locs = None


# Dynamical data set for each process. comm.scatter sends the ith index of rank 0 Split data to the ith process. So each 
# process recieves its own subarray. 
u1 = comm.scatter(u1matSplit, root=0)
u2 = comm.scatter(u2matSplit, root=0)
v1 = comm.scatter(v1matSplit, root=0)
v2 = comm.scatter(v2matSplit, root=0)
h1 = comm.scatter(h1matSplit, root=0)
h2 = comm.scatter(h2matSplit, root=0)
spdrag1 = comm.scatter(spdrag1Split, root=0)
spdrag2 = comm.scatter(spdrag2Split, root=0)
rdist = comm.scatter(rdistSplit, root=0)
lasttime = comm.bcast(lasttime, root=0)
x = comm.scatter(xSplit, root=0)
y = comm.scatter(ySplit, root=0)
locs = comm.bcast(locs, root=0)


Wsum = None # Current weather layer set to none

# Each process (not 0) generates the storms in their own subarray
if rank != 0:
    wlayer = hf.pairshapeN2(locs, 0, x, y, offset)
    Wsum = np.sum(wlayer) * dx**2

Wsums = comm.gather(Wsum, root=0) # Send all weather layers to rank 0 for subsidence calculation


# Rank 0 calculates subsidence from all the storms and creates a final weather layer
if rank == 0:
    area = L**2
    wcorrect = np.sum(Wsums[1:]) / area

wcorrect = comm.bcast(wcorrect, root=0) # Distribute the subsidence correction to all process

# Each process adds subsidence to the final weather layer. Note this code replaces helper_function pairfieldN2.
if rank != 0:
    Wmat = wlayer - wcorrect


### END OF INITIALIZATION ###


"""
This is the start of time stepping.

Each thread solves the 2.5 layer shallow water equations on their own grid cell and then sends boundary data to its 
neighbouring cells.

The time step function...

This function takes in each threads u1,u2,... and then solves the shallow water equations in them. It is the exact 
same code as in other files, just wrapped in a function instead of a while loop. 
"""
def timestep(u1,u2,v1,v2,h1,h2,Wmat, u1_p,u2_p,v1_p,v2_p,h1_p,h2_p):
    broke = False

    # initialize Adam-Bashforth scheme.
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

    # Add spongedrag
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

    # add vorticity to momentum equations
    du1dt = du1dt + 0.25 * (zv1 + zv1[r,:])
    du2dt = du2dt + 0.25 * (zv2 + zv2[r,:])

    zu1 = zeta1 * (u1 + u1[l,:])
    zu2 = zeta2 * (u2 + u2[l,:])

    dv1dt = dv1dt - 0.25 * (zu1 + zu1[:,r])
    dv2dt = dv2dt - 0.25 * (zu2 + zu2[:,r])

    ### Cumulus Drag ###
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
#print("Starting simulation")

sendingTimes = []
simTimes = []
stormTimes = []
broke = False


initialmem = rss()
clocktimer = time.time()


while t <= tmax + lasttime + dt / 2:
    ### Running of the simulation on all ranks but the master rank (0) ###
    simtimer = time.time()

    if rank != 0:
        u1,u2,v1,v2,h1,h2, u1_p,u2_p,v1_p,v2_p,h1_p,h2_p, broke = timestep(u1,u2,v1,v2,h1,h2,Wmat, u1_p,u2_p,v1_p,v2_p,h1_p,h2_p)

    
    if broke == True:
        print(f"h1 Nan on rank {rank}")
        MPI.Finalize()
        MPI.COMM_WORLD.Abort()

    #simTimes.append(time.time()-simtimer)

    ### Sending boundary conditions to neighbouring cells

    sendtimer = time.time()
    
    if rank != 0:
        ind = np.where(ranks == rank)
        i = ind[0][0] + int(np.sqrt(size))
        j = ind[1][0] + int(np.sqrt(size))
        sendranks, recvranks = hf.get_surrounding_points(largeranks, i, j)
        
        for sendrank in sendranks:
            if (sendrank[0], sendrank[1]) == (-1,-1):
                comm.send([u1[2:4,:][:,2:4],u2[2:4,:][:,2:4],v1[2:4,:][:,2:4],v2[2:4,:][:,2:4],h1[2:4,:][:,2:4],h2[2:4,:][:,2:4]], 
                          dest=sendrank[2], tag=0)

            if (sendrank[0], sendrank[1]) == (-1,0):
                comm.send([u1[2:4,:][:,2:offset+2],u2[2:4,:][:,2:offset+2],v1[2:4,:][:,2:offset+2],v2[2:4,:][:,2:offset+2],h1[2:4,:][:,2:offset+2],h2[2:4,:][:,2:offset+2]], 
                          dest=sendrank[2], tag=1)
      
            if (sendrank[0], sendrank[1]) == (-1,1):
                comm.send([u1[2:4,:][:,offset:offset+2],u2[2:4,:][:,offset:offset+2],v1[2:4,:][:,offset:offset+2],v2[2:4,:][:,offset:offset+2],h1[2:4,:][:,offset:offset+2],h2[2:4,:][:,offset:offset+2]],     
                          dest=sendrank[2], tag=2)

            if (sendrank[0], sendrank[1]) == (0,-1):
                comm.send([u1[2:offset+2,:][:,2:4],u2[2:offset+2,:][:,2:4],v1[2:offset+2,:][:,2:4],v2[2:offset+2,:][:,2:4],h1[2:offset+2,:][:,2:4],h2[2:offset+2,:][:,2:4]],
                          dest=sendrank[2], tag=3)

            if (sendrank[0], sendrank[1]) == (0,1):
                comm.send([u1[2:offset+2,:][:,offset:offset+2],u2[2:offset+2,:][:,offset:offset+2],v1[2:offset+2,:][:,offset:offset+2],v2[2:offset+2,:][:,offset:offset+2],h1[2:offset+2,:][:,offset:offset+2],h2[2:offset+2,:][:,offset:offset+2]],
                          dest=sendrank[2], tag=4)
            
            if (sendrank[0], sendrank[1]) == (1,-1):
                comm.send([u1[offset:offset+2,:][:,2:4],u2[offset:offset+2,:][:,2:4],v1[offset:offset+2,:][:,2:4],v2[offset:offset+2,:][:,2:4],h1[offset:offset+2,:][:,2:4],h2[offset:offset+2,:][:,2:4]],
                          dest=sendrank[2], tag=5)

            if (sendrank[0], sendrank[1]) == (1,0):
                comm.send([u1[offset:offset+2,:][:,2:offset+2],u2[offset:offset+2,:][:,2:offset+2],v1[offset:offset+2,:][:,2:offset+2],v2[offset:offset+2,:][:,2:offset+2],h1[offset:offset+2,:][:,2:offset+2], h2[offset:offset+2,:][:,2:offset+2]], 
                          dest=sendrank[2], tag=6)
            
            if (sendrank[0], sendrank[1]) == (1,1):
                comm.send([u1[offset:offset+2,:][:,offset:offset+2],u2[offset:offset+2,:][:,offset:offset+2],v1[offset:offset+2,:][:,offset:offset+2],v2[offset:offset+2,:][:,offset:offset+2],h1[offset:offset+2,:][:,offset:offset+2],h2[offset:offset+2,:][:,offset:offset+2]],
                          dest=sendrank[2], tag=7)


        for sendrank in sendranks:
            if (sendrank[0], sendrank[1]) == (-1,-1):
                data = comm.recv(source=sendrank[2], tag=7)
                u1[0:2,:][:,0:2] = data[0]
                u2[0:2,:][:,0:2] = data[1]
                v1[0:2,:][:,0:2] = data[2]
                v2[0:2,:][:,0:2] = data[3]
                h1[0:2,:][:,0:2] = data[4]
                h2[0:2,:][:,0:2] = data[5]

            if (sendrank[0], sendrank[1]) == (-1,0):
                data = comm.recv(source=sendrank[2], tag=6)
                u1[0:2,:][:,2:offset+2] = data[0]
                u2[0:2,:][:,2:offset+2] = data[1]
                v1[0:2,:][:,2:offset+2] = data[2]
                v2[0:2,:][:,2:offset+2] = data[3]
                h1[0:2,:][:,2:offset+2] = data[4]
                h2[0:2,:][:,2:offset+2] = data[5]

            if (sendrank[0], sendrank[1]) == (-1,1):
                data = comm.recv(source=sendrank[2], tag=5)
                u1[0:2,:][:,offset+2:offset+4] = data[0]
                u2[0:2,:][:,offset+2:offset+4] = data[1]
                v1[0:2,:][:,offset+2:offset+4] = data[2]
                v2[0:2,:][:,offset+2:offset+4] = data[3]
                h1[0:2,:][:,offset+2:offset+4] = data[4]
                h2[0:2,:][:,offset+2:offset+4] = data[5]

            if (sendrank[0], sendrank[1]) == (0,-1):
                data = comm.recv(source=sendrank[2], tag=4)
                u1[2:offset+2,:][:,0:2] = data[0]
                u2[2:offset+2,:][:,0:2] = data[1]
                v1[2:offset+2,:][:,0:2] = data[2]
                v2[2:offset+2,:][:,0:2] = data[3]
                h1[2:offset+2,:][:,0:2] = data[4]
                h2[2:offset+2,:][:,0:2] = data[5]

            if (sendrank[0], sendrank[1]) == (0,1):
                data = comm.recv(source=sendrank[2], tag=3)
                u1[2:offset+2,:][:,offset+2:offset+4] = data[0]
                u2[2:offset+2,:][:,offset+2:offset+4] = data[1]
                v1[2:offset+2,:][:,offset+2:offset+4] = data[2]
                v2[2:offset+2,:][:,offset+2:offset+4] = data[3]
                h1[2:offset+2,:][:,offset+2:offset+4] = data[4]
                h2[2:offset+2,:][:,offset+2:offset+4] = data[5]
            
            if (sendrank[0], sendrank[1]) == (1,-1):
                data = comm.recv(source=sendrank[2], tag=2)
                u1[offset+2:offset+4,:][:,0:2] = data[0]
                u2[offset+2:offset+4,:][:,0:2] = data[1]
                v1[offset+2:offset+4,:][:,0:2] = data[2]
                v2[offset+2:offset+4,:][:,0:2] = data[3]
                h1[offset+2:offset+4,:][:,0:2] = data[4]
                h2[offset+2:offset+4,:][:,0:2] = data[5]
            
            if (sendrank[0], sendrank[1]) == (1,0):
                data = comm.recv(source=sendrank[2], tag=1)
                u1[offset+2:offset+4,:][:,2:offset+2] = data[0]
                u2[offset+2:offset+4,:][:,2:offset+2] = data[1]
                v1[offset+2:offset+4,:][:,2:offset+2] = data[2]
                v2[offset+2:offset+4,:][:,2:offset+2] = data[3]
                h1[offset+2:offset+4,:][:,2:offset+2] = data[4]
                h2[offset+2:offset+4,:][:,2:offset+2] = data[5]

            if (sendrank[0], sendrank[1]) == (1,1):
                data = comm.recv(source=sendrank[2], tag=0)
                u1[offset+2:offset+4,:][:,offset+2:offset+4] = data[0]
                u2[offset+2:offset+4,:][:,offset+2:offset+4] = data[1]
                v1[offset+2:offset+4,:][:,offset+2:offset+4] = data[2]
                v2[offset+2:offset+4,:][:,offset+2:offset+4] = data[3]
                h1[offset+2:offset+4,:][:,offset+2:offset+4] = data[4]
                h2[offset+2:offset+4,:][:,offset+2:offset+4] = data[5]

    #sendingTimes.append(time.time()-sendtimer)
    
    
    ### Rank 0 checks for if new storms need to be created and sends out the new Wmat ###

    stormtimer = time.time()
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

                #wlayer = hf.pairshapeN2(locs, t) ### use pairshapeBEGIN instead of pairshape
                #Wmat = hf.pairfieldN2(L, h1, wlayer)

        if len(remove_layers) != 0:
            rem = True
            
            #WmatSplit = [Wmat]
            #for i in range(1,size+1):
            #    WmatSplit.append(hf.split(Wmat, offset, ranks, i))
    
    rem = comm.bcast(rem, root=0)
    locs = comm.bcast(locs, root=0)
    if rem == True:
        if rank != 0:
            wlayer = hf.pairshapeN2(locs, 0, x, y, offset)
            #Wmat = hf.pairfieldN2(L, h1, wlayer)
            Wsum = np.sum(wlayer) * dx**2

        Wsums = comm.gather(Wsum, root=0)

        if rank == 0:
            area = L**2
            wcorrect = np.sum(Wsums[1:]) / area

        wcorrect = comm.bcast(wcorrect, root=0)

        if rank != 0:
            Wmat = wlayer - wcorrect

        rem = False

    #stormTimes.append(time.time()-stormtimer)

    if tc % tpl == 0 and saving == True:
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

            print(f"t={t}, time elapsed {time.time()-clocktimer}")

            ad.save_data(u1,u2,v1,v2,h1,h2,locs,t,lasttime,new_name)

    tc += 1
    t = tc * dt

#print(f"rank: {rank}, simtime avg: {round(np.mean(simTimes),4)}, sendingtime avg: {round(np.mean(sendingTimes),4)}, stormtime avg: {round(np.mean(stormTimes), 4)}, total time: {round(time.time()-tottimer,4)}, mem used: {(rss() - initialmem)/10**6} MB")

#print(f"mem used: {(rss() - initialmem)/10**6} MB")