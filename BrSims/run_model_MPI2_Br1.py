import numpy as np
import math
import helper_functions_MPI as hf
import time
from name_list_uranus2_Br1 import *
from netCDF4 import Dataset
import access_data as ad
from mpi4py import MPI
import psutil
import sys


def rss():
    return psutil.Process().memory_info().rss

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

largeranks = np.zeros((int(3*np.sqrt(size)), int(3*np.sqrt(size))), dtype=ranks.dtype)

for i in range(int(3*np.sqrt(size))):
    for j in range(int(3*np.sqrt(size))):
        largeranks[i, j] = ranks[i % int(np.sqrt(size)), j % int(np.sqrt(size))]

if rank == 0:
    #print(largeranks)
    if restart_name == None:
        if saving == True:
            ad.create_file(new_name)
        lasttime = 0
        locs = hf.genlocs(num, N, 0) ### use genlocs instead of paircount
    else:
        ad.create_file(new_name)
        u1, u2, v1, v2, h1, h2, locs, lasttime = ad.last_timestep(restart_name)

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

#Wmat = comm.scatter(WmatSplit, root=0)
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

Wsum = None

if rank != 0:
    wlayer = hf.pairshapeN2(locs, 0, x, y, offset)
    Wsum = np.sum(wlayer) * dx**2

Wsums = comm.gather(Wsum, root=0)

if rank == 0:
    area = L**2
    wcorrect = np.sum(Wsums[1:]) / area

wcorrect = comm.bcast(wcorrect, root=0)

if rank != 0:
    Wmat = wlayer - wcorrect

"""
print(np.array([u1[2:4,:][:,2:4],u2[2:4,:][:,2:4],v1[2:4,:][:,2:4],v2[2:4,:][:,2:4],h1[2:4,:][:,2:4],h2[2:4,:][:,2:4]]).dtype)
if rank != 0:
    print(tlbuf.dtype)
"""

tlbuf = np.array([u1[offset:offset+2,:][:,offset:offset+2],u2[offset:offset+2,:][:,offset:offset+2],v1[offset:offset+2,:][:,offset:offset+2],v2[offset:offset+2,:][:,offset:offset+2],h1[offset:offset+2,:][:,offset:offset+2],h2[offset:offset+2,:][:,offset:offset+2]], dtype=float)
tbuf = np.array([u1[2:4,:][:,2:offset+2],u2[2:4,:][:,2:offset+2],v1[2:4,:][:,2:offset+2],v2[2:4,:][:,2:offset+2],h1[2:4,:][:,2:offset+2],h2[2:4,:][:,2:offset+2]], dtype=float)
trbuf = np.array([u1[2:4,:][:,offset:offset+2],u2[2:4,:][:,offset:offset+2],v1[2:4,:][:,offset:offset+2],v2[2:4,:][:,offset:offset+2],h1[2:4,:][:,offset:offset+2],h2[2:4,:][:,offset:offset+2]], dtype=float)
lbuf = np.array([u1[2:offset+2,:][:,2:4],u2[2:offset+2,:][:,2:4],v1[2:offset+2,:][:,2:4],v2[2:offset+2,:][:,2:4],h1[2:offset+2,:][:,2:4],h2[2:offset+2,:][:,2:4]], dtype=float)
rbuf = np.array([u1[2:offset+2,:][:,offset:offset+2],u2[2:offset+2,:][:,offset:offset+2],v1[2:offset+2,:][:,offset:offset+2],v2[2:offset+2,:][:,offset:offset+2],h1[2:offset+2,:][:,offset:offset+2],h2[2:offset+2,:][:,offset:offset+2]], dtype=float)
blbuf = np.array([u1[offset:offset+2,:][:,2:4],u2[offset:offset+2,:][:,2:4],v1[offset:offset+2,:][:,2:4],v2[offset:offset+2,:][:,2:4],h1[offset:offset+2,:][:,2:4],h2[offset:offset+2,:][:,2:4]], dtype=float)
bbuf = np.array([u1[offset:offset+2,:][:,2:offset+2],u2[offset:offset+2,:][:,2:offset+2],v1[offset:offset+2,:][:,2:offset+2],v2[offset:offset+2,:][:,2:offset+2],h1[offset:offset+2,:][:,2:offset+2], h2[offset:offset+2,:][:,2:offset+2]], dtype=float)
brbuf = np.array([u1[offset:offset+2,:][:,offset:offset+2],u2[offset:offset+2,:][:,offset:offset+2],v1[offset:offset+2,:][:,offset:offset+2],v2[offset:offset+2,:][:,offset:offset+2],h1[offset:offset+2,:][:,offset:offset+2],h2[offset:offset+2,:][:,offset:offset+2]], dtype=float)

### END OF INITIALIZATION ###


"""
This is the start of time stepping.

Each thread solves the 2.5 layer shallow water equations on their own grid cell and then sends boundary data to its 
neighbouring cells so that there are no weird boundaries

The time step function...

This function takes in each threads u1,u2,... and then solves the shallow water equations in them. It is the exact 
same code as in other files, just wrapped in a function instead of a while loop. 
"""
def timestep(u1,u2,v1,v2,h1,h2,Wmat, u1_p,u2_p,v1_p,v2_p,h1_p,h2_p, t):
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

    ### Update the seasonal dependent parameters ###
    #if seasonalsim == True:
    #    tradf = hf.seasonaltrad(t)
    #    H1H2 = hf.seasonalH1H2(t)
    #    #Wsh = Wsh / hf.seasonalH1(t)
    #    h1 = h1 / hf.seasonalH1(t)

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
    zeta1 = 1 - Bt * rdist ** 2 + (1 / dx) * (v1 - np.roll(v1, 1, axis=1) + np.roll(u1, 1, axis=0) - u1)  
    zeta2 = 1 - Bt * rdist ** 2 + (1 / dx) * (v2 - np.roll(v2, 1, axis=1) + np.roll(u2, 1, axis=0) - u2)  

    # add vorticity flux, zeta*u
    zv1 = zeta1 * (v1 + np.roll(v1, 1, axis=1))
    zv2 = zeta2 * (v2 + np.roll(v2, 1, axis=1))

    du1dt = du1dt + 0.25 * (zv1 + np.roll(zv1, -1, axis=0))
    du2dt = du2dt + 0.25 * (zv2 + np.roll(zv2, -1, axis=0))

    zu1 = zeta1 * (u1 + np.roll(u1, 1, axis=0))
    zu2 = zeta2 * (u2 + np.roll(u2, 1, axis=0))

    dv1dt = dv1dt - 0.25 * (zu1 + np.roll(zu1, -1, axis=1))
    dv2dt = dv2dt - 0.25 * (zu2 + np.roll(zu2, -1, axis=1))

    #du1dt = du1dt - u1 / dragf
    #du2dt = du2dt - u2 / dragf
    #dv1dt = dv1dt - v1 / dragf
    #dv2dt = dv2dt - v2 / dragf                       

    B1p, B2p = hf.BernN2(u1, v1, u2, v2, gm, c22h, c12h, h1, h2, ord)

    du1dtsq = du1dt - (1 / dx) * (B1p - np.roll(B1p, 1, axis=1))  
    du2dtsq = du2dt - (1 / dx) * (B2p - np.roll(B2p, 1, axis=1))  

    dv1dtsq = dv1dt - (1 / dx) * (B1p - np.roll(B1p, 1, axis=0))  
    dv2dtsq = dv2dt - (1 / dx) * (B2p - np.roll(B2p, 1, axis=0))  
    
    if AB == 2:
        u1sq = u1_p + dt * du1dtsq
        u2sq = u2_p + dt * du2dtsq

        v1sq = v1_p + dt * dv1dtsq
        v2sq = v2_p + dt * dv2dtsq


    Fx1 = hf.xflux(h1, u1) - kappa / dx * (h1 - np.roll(h1, 1, axis=1))
    Fy1 = hf.yflux(h1, v1) - kappa / dx * (h1 - np.roll(h1, 1, axis=0))
    dh1dt = -(1 / dx) * (np.roll(Fx1, -1, axis=1) - Fx1 + np.roll(Fy1, -1, axis=0) - Fy1)

    if layers == 2.5:
        Fx2 = hf.xflux(h2, u2) - kappa / dx * (h2 - np.roll(h2, 1, axis=1))
        Fy2 = hf.yflux(h2, v2) - kappa / dx * (h2 - np.roll(h2, 1, axis=0))
        dh2dt = -(1 / dx) * (np.roll(Fx2, -1, axis=1) - Fx2 + np.roll(Fy2, -1, axis=0) - Fy2)

    if tradf > 0:
        dh1dt = dh1dt - 1 / tradf * (h1 - 1)
        dh2dt = dh2dt - 1 / trad0f * (h2 - 1)
    
    if mode == 1:
        dh1dt = dh1dt + Wmat
        if layers == 2.5:
            dh2dt = dh2dt - H1H2 * Wmat

    if AB == 2:
        h1 = h1_p + dt * dh1dt
        if layers == 2.5:
            h2 = h2_p + dt * dh2dt

    u1 = u1sq
    u2 = u2sq
    v1 = v1sq
    v2 = v2sq

    if np.isnan(h1).any():
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
        u1,u2,v1,v2,h1,h2, u1_p,u2_p,v1_p,v2_p,h1_p,h2_p, broke = timestep(u1,u2,v1,v2,h1,h2,Wmat, u1_p,u2_p,v1_p,v2_p,h1_p,h2_p, t)

    
    if broke == True:
        print(f"h1 Nan on rank {rank}")
        sys.exit()

    simTimes.append(time.time()-simtimer)

    ### Sending boundary conditions to neighbouring cells
    sendtimer = time.time()
    
    if rank != 0:
        ind = np.where(ranks == rank)
        i = ind[0][0] + int(np.sqrt(size))
        j = ind[1][0] + int(np.sqrt(size))
        sendranks, recvranks = hf.get_surrounding_points(largeranks, i, j)
        
        for sendrank in sendranks:
            if (sendrank[0], sendrank[1]) == (-1,-1):
                comm.Send(np.array([u1[2:4,:][:,2:4],u2[2:4,:][:,2:4],v1[2:4,:][:,2:4],v2[2:4,:][:,2:4],h1[2:4,:][:,2:4],h2[2:4,:][:,2:4]], dtype=np.float64), 
                          dest=sendrank[2], tag=0)

            if (sendrank[0], sendrank[1]) == (-1,0):
                comm.Send(np.array([u1[2:4,:][:,2:offset+2],u2[2:4,:][:,2:offset+2],v1[2:4,:][:,2:offset+2],v2[2:4,:][:,2:offset+2],h1[2:4,:][:,2:offset+2],h2[2:4,:][:,2:offset+2]], dtype=np.float64), 
                          dest=sendrank[2], tag=1)
      
            if (sendrank[0], sendrank[1]) == (-1,1):
                comm.Send(np.array([u1[2:4,:][:,offset:offset+2],u2[2:4,:][:,offset:offset+2],v1[2:4,:][:,offset:offset+2],v2[2:4,:][:,offset:offset+2],h1[2:4,:][:,offset:offset+2],h2[2:4,:][:,offset:offset+2]], dtype=np.float64),     
                          dest=sendrank[2], tag=2)

            if (sendrank[0], sendrank[1]) == (0,-1):
                comm.Send(np.array([u1[2:offset+2,:][:,2:4],u2[2:offset+2,:][:,2:4],v1[2:offset+2,:][:,2:4],v2[2:offset+2,:][:,2:4],h1[2:offset+2,:][:,2:4],h2[2:offset+2,:][:,2:4]], dtype=np.float64),
                          dest=sendrank[2], tag=3)

            if (sendrank[0], sendrank[1]) == (0,1):
                comm.Send(np.array([u1[2:offset+2,:][:,offset:offset+2],u2[2:offset+2,:][:,offset:offset+2],v1[2:offset+2,:][:,offset:offset+2],v2[2:offset+2,:][:,offset:offset+2],h1[2:offset+2,:][:,offset:offset+2],h2[2:offset+2,:][:,offset:offset+2]], dtype=np.float64),
                          dest=sendrank[2], tag=4)
            
            if (sendrank[0], sendrank[1]) == (1,-1):
                comm.Send(np.array([u1[offset:offset+2,:][:,2:4],u2[offset:offset+2,:][:,2:4],v1[offset:offset+2,:][:,2:4],v2[offset:offset+2,:][:,2:4],h1[offset:offset+2,:][:,2:4],h2[offset:offset+2,:][:,2:4]], dtype=np.float64),
                          dest=sendrank[2], tag=5)

            if (sendrank[0], sendrank[1]) == (1,0):
                comm.Send(np.array([u1[offset:offset+2,:][:,2:offset+2],u2[offset:offset+2,:][:,2:offset+2],v1[offset:offset+2,:][:,2:offset+2],v2[offset:offset+2,:][:,2:offset+2],h1[offset:offset+2,:][:,2:offset+2], h2[offset:offset+2,:][:,2:offset+2]], dtype=np.float64), 
                          dest=sendrank[2], tag=6)
            
            if (sendrank[0], sendrank[1]) == (1,1):
                comm.Send(np.array([u1[offset:offset+2,:][:,offset:offset+2],u2[offset:offset+2,:][:,offset:offset+2],v1[offset:offset+2,:][:,offset:offset+2],v2[offset:offset+2,:][:,offset:offset+2],h1[offset:offset+2,:][:,offset:offset+2],h2[offset:offset+2,:][:,offset:offset+2]], dtype=float),
                          dest=sendrank[2], tag=7)


        for sendrank in sendranks:
            if (sendrank[0], sendrank[1]) == (-1,-1):
                req = comm.Irecv(buf=tlbuf, source=sendrank[2], tag=7)
                req.wait()
                u1[0:2,:][:,0:2] = tlbuf[0]
                u2[0:2,:][:,0:2] = tlbuf[1]
                v1[0:2,:][:,0:2] = tlbuf[2]
                v2[0:2,:][:,0:2] = tlbuf[3]
                h1[0:2,:][:,0:2] = tlbuf[4]
                h2[0:2,:][:,0:2] = tlbuf[5]

            if (sendrank[0], sendrank[1]) == (-1,0):
                req = comm.Irecv(buf=tbuf, source=sendrank[2], tag=6)
                req.wait()
                u1[0:2,:][:,2:offset+2] = tbuf[0]
                u2[0:2,:][:,2:offset+2] = tbuf[1]
                v1[0:2,:][:,2:offset+2] = tbuf[2]
                v2[0:2,:][:,2:offset+2] = tbuf[3]
                h1[0:2,:][:,2:offset+2] = tbuf[4]
                h2[0:2,:][:,2:offset+2] = tbuf[5]

            if (sendrank[0], sendrank[1]) == (-1,1):
                req = comm.Irecv(buf=trbuf, source=sendrank[2], tag=5)
                req.wait()
                u1[0:2,:][:,offset+2:offset+4] = trbuf[0]
                u2[0:2,:][:,offset+2:offset+4] = trbuf[1]
                v1[0:2,:][:,offset+2:offset+4] = trbuf[2]
                v2[0:2,:][:,offset+2:offset+4] = trbuf[3]
                h1[0:2,:][:,offset+2:offset+4] = trbuf[4]
                h2[0:2,:][:,offset+2:offset+4] = trbuf[5]

            if (sendrank[0], sendrank[1]) == (0,-1):
                req = comm.Irecv(buf=lbuf, source=sendrank[2], tag=4)
                req.wait()
                u1[2:offset+2,:][:,0:2] = lbuf[0]
                u2[2:offset+2,:][:,0:2] = lbuf[1]
                v1[2:offset+2,:][:,0:2] = lbuf[2]
                v2[2:offset+2,:][:,0:2] = lbuf[3]
                h1[2:offset+2,:][:,0:2] = lbuf[4]
                h2[2:offset+2,:][:,0:2] = lbuf[5]

            if (sendrank[0], sendrank[1]) == (0,1):
                req = comm.Irecv(buf=rbuf, source=sendrank[2], tag=3)
                req.wait()
                u1[2:offset+2,:][:,offset+2:offset+4] = rbuf[0]
                u2[2:offset+2,:][:,offset+2:offset+4] = rbuf[1]
                v1[2:offset+2,:][:,offset+2:offset+4] = rbuf[2]
                v2[2:offset+2,:][:,offset+2:offset+4] = rbuf[3]
                h1[2:offset+2,:][:,offset+2:offset+4] = rbuf[4]
                h2[2:offset+2,:][:,offset+2:offset+4] = rbuf[5]
            
            if (sendrank[0], sendrank[1]) == (1,-1):
                req = comm.Irecv(buf=blbuf, source=sendrank[2], tag=2)
                req.wait()
                u1[offset+2:offset+4,:][:,0:2] = blbuf[0]
                u2[offset+2:offset+4,:][:,0:2] = blbuf[1]
                v1[offset+2:offset+4,:][:,0:2] = blbuf[2]
                v2[offset+2:offset+4,:][:,0:2] = blbuf[3]
                h1[offset+2:offset+4,:][:,0:2] = blbuf[4]
                h2[offset+2:offset+4,:][:,0:2] = blbuf[5]
            
            if (sendrank[0], sendrank[1]) == (1,0):
                req = comm.Irecv(buf=bbuf, source=sendrank[2], tag=1)
                req.wait()
                u1[offset+2:offset+4,:][:,2:offset+2] = bbuf[0]
                u2[offset+2:offset+4,:][:,2:offset+2] = bbuf[1]
                v1[offset+2:offset+4,:][:,2:offset+2] = bbuf[2]
                v2[offset+2:offset+4,:][:,2:offset+2] = bbuf[3]
                h1[offset+2:offset+4,:][:,2:offset+2] = bbuf[4]
                h2[offset+2:offset+4,:][:,2:offset+2] = bbuf[5]

            if (sendrank[0], sendrank[1]) == (1,1):
                req = comm.Irecv(buf=brbuf, source=sendrank[2], tag=0)
                req.wait()
                u1[offset+2:offset+4,:][:,offset+2:offset+4] = brbuf[0]
                u2[offset+2:offset+4,:][:,offset+2:offset+4] = brbuf[1]
                v1[offset+2:offset+4,:][:,offset+2:offset+4] = brbuf[2]
                v2[offset+2:offset+4,:][:,offset+2:offset+4] = brbuf[3]
                h1[offset+2:offset+4,:][:,offset+2:offset+4] = brbuf[4]
                h2[offset+2:offset+4,:][:,offset+2:offset+4] = brbuf[5]

    sendingTimes.append(time.time()-sendtimer)
    
    
    ### Rank 0 checks for if new storms need to be created and sends out the new Wmat ###

    stormtimer = time.time()

    # Rank zero checks which storms need to be removed.
    if rank == 0:
        remove_layers = [] # store weather layers that need to be removed here

        # Check if a storm is past its lifetime, tstpf < lifetime.
        if mode == 1:
            for i in range(len(locs)):
                if (t-locs[i][-1]) >= locs[i][3] and t != 0:
                    remove_layers.append(i) # tag layer for removal if a storm's 

            add = len(remove_layers) # number of storms that were removed

            # If new storms need to be created, create a list of new storms to add.
            if add != 0:
                newlocs = hf.genlocs(add, N, t)

                # update the old list of storms with the new ones.
                for i in range(len(remove_layers)):
                    locs[remove_layers[i]] = newlocs[i]


    locs = comm.bcast(locs, root=0) # Communicate updated storms to all processes.


    # each process (except 0) calculates its own storms
    if tc % tp1 == 0:
        if rank != 0:
            wlayer = hf.pairshapeN2(locs, t, x, y, offset)
            #Wmat = hf.pairfieldN2(L, h1, wlayer)
            Wsum = np.sum(wlayer) * dx**2

        Wsums = comm.gather(Wsum, root=0) # Gather the sums on rank 0 for subsidence calculations.

        # Calculate subsidence correction on rank 0
        if rank == 0:
            area = L**2
            wcorrect = np.sum(Wsums[1:]) / area

        wcorrect = comm.bcast(wcorrect, root=0) # Distribute subsidence calculation to all process.

        # Update weather layer of each process with subsidence.
        if rank != 0:
            Wmat = wlayer - wcorrect

    stormTimes.append(time.time()-stormtimer) # Uncomment to time the storm creation section.

    ### Data saving. Rank 0 gathers all data from processes, combines it, and then saves it to a NETCDF file. ###

    # Check if it is time to save.
    if tc % tpl == 0 and saving == True:
        # Gather all dynamical data to rank 0. These will be a list of each subarray of each process.
        u1matSplit = comm.gather(u1, root=0)
        v1matSplit = comm.gather(v1, root=0)
        u2matSplit = comm.gather(u2, root=0)
        v2matSplit = comm.gather(v2, root=0)
        h1matSplit = comm.gather(h1, root=0)
        h2matSplit = comm.gather(h2, root=0)

        # Combine all the data into one large array on rank 0
        if rank == 0:
            u1 = hf.combine(u1matSplit, offset, ranks, size)
            u2 = hf.combine(u2matSplit, offset, ranks, size)
            v1 = hf.combine(v1matSplit, offset, ranks, size)
            v2 = hf.combine(v2matSplit, offset, ranks, size)
            h1 = hf.combine(h1matSplit, offset, ranks, size)
            h2 = hf.combine(h2matSplit, offset, ranks, size)

            # Print the time the cycle from saving to saving has taken
            print(f"t={t}, time elapsed {time.time()-clocktimer}")

            clocktimer = time.time()
            #print(f"memory used {rss()-initialmem}")

            # Save the data to NETCDF.
            ad.save_data(u1,u2,v1,v2,h1,h2,locs,t,lasttime,new_name)

    # Update times.
    tc += 1 
    t = tc * dt

print(f"rank: {rank}, simtime avg: {round(np.mean(simTimes),4)}, sendingtime avg: {round(np.mean(sendingTimes),4)}, stormtime avg: {round(np.mean(stormTimes), 4)}, total time: {round(time.time()-tottimer,4)}")

#print(f"mem used: {(rss() - initialmem)/10**6} MB")