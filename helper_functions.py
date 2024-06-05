import numpy as np
from numba import jit
from name_list import *


@jit(nopython=True, parallel=True)
def pairfieldN2(L, h1, wlayer):
    """
    Creates the weather matrix for the storms, S_st in paper.
    """
    
    voldw = np.sum(wlayer) * dx**2
    area = L**2
    wcorrect = voldw / area
    Wmat = wlayer - wcorrect
    return Wmat


@jit(nopython=True, parallel=True)
def viscND(vel, Re, n):
    """
    n is exponent of Laplacian operator
    Where visc term is nu*(-1)^(n+1) (\/^2)^n
    so for regular viscosity n = 1, for hyperviscosity n=2

    TODO: for n=1 nu is not defined...
    """

    field = np.zeros_like(vel)

    if n == 2:
   
        field = (2*vel[:,l][l,:] + 2*vel[:,r][l,:] + 2*vel[:,l][r,:] + 2*vel[:,r][r,:]
                 - 8*vel[l,:] - 8*vel[r,:] - 8*vel[:,l] - 8*vel[:,r]
                 + vel[l2,:] + vel[r2,:] + vel[:,l2] + vel[:,r2]
                 + 20*vel
        )

        field = -1 / Re * (1 / dx**4) * field

    
    return field


### New pairshapeN2 function. Generates Gaussians using entire domain instead of creating sub-domains. (Daniel) ###
@jit(nopython=True, parallel=True)
def pairshapeN2(locs, x, y, Br2, Wsh, N):

    wlayer = np.zeros_like(x).astype(np.float64)
    
    for loc in locs:
        layer = Wsh * np.exp( - (Br2*dx**2)/0.3606 * ( (x-loc[0])**2 + (y-loc[1])**2))
        wlayer = wlayer + layer

    return wlayer


@jit(nopython=True, parallel=True)
def BernN2(u1, v1, u2, v2, gm, c22h, c12h, h1, h2, ord):
    """
    Bernoulli
    """
    B1 = c12h * h1 + c22h * h2 + 0.25 * (u1**2 + u1[:,r]**2 + v1**2 + v1[r,:]**2)


    B2 = gm * c12h * h1 + c22h * h2 + 0.25 * (u1**2 + u1[:,r]**2 + v1**2 + v1[r,:]**2)

    return B1, B2


@jit(nopython=True, parallel=True)
def xflux(f, u):  # removed dx, dt from input
    fl = f[:,l]
    fr = f

    fa = 0.5 * u * (fl + fr)

    return fa


@jit(nopython=True, parallel=True)
def yflux(f, v):  # removed dx, dt from input
    fl = f[l,:]
    fr = f

    fa = 0.5 * v * (fl + fr)

    return fa

@jit(nopython=True, parallel=True)
def calculate_KE(u1,u2,v1,v2,h1,h2):
    first = p1p2*H1H2*h1*(u1**2 + v1**2)
    second = h2*(u2**2 + v2**2)

    return 0.5 * np.sum(first + second)


@jit(nopython=True, parallel=True)
def calculate_APE(h1, h2):
    first = 0.5*p1p2*H1H2*c12h*(h1-1)**2
    second = 0.5*c22h*(h2-1)**2
    third = p1p2*H1H2*(c22h/c12h)*c12h*(h1-1)*(h2-1)

    return np.sum(first + second + third)


######### new helper functions for new storm forcing ##########

#@jit(nopython=True, parallel=True)
def pairshapeBEGIN(locs, x, y, Br2, Wsh, N, locslayers):
    rad = int(np.ceil(np.sqrt(1 / Br2) / dx))
    xg, yg = np.meshgrid(range(-rad, rad + 1), range(-rad, rad + 1))
    gaus = Wsh * np.exp(-(Br2 * dx**2) / 0.3606 * ((xg + 0.5) ** 2 + (yg + 0.5) ** 2))

    wlayer = np.zeros(x.shape)

    buf = rad
    bufmat = np.zeros((N + 2 * rad, N + 2 * rad))
    nlocs = locs + rad

    corners = nlocs - rad

    for jj in range(locs.shape[0]):
        tempbufmat = np.zeros((N + 2 * rad, N + 2 * rad))
        tempbufmat[
            corners[jj, 0] : corners[jj, 0] + gaus.shape[0],
            corners[jj, 1] : corners[jj, 1] + gaus.shape[1],
        ] += gaus

        tempmainlayer = tempbufmat[buf : buf + N, buf : buf + N].copy()

        addlayer11 = np.zeros_like(tempmainlayer)
        addlayer22 = np.zeros_like(tempmainlayer)
        addlayer33 = np.zeros_like(tempmainlayer)
        addlayer44 = np.zeros_like(tempmainlayer)
        addcorn11 = np.zeros_like(tempmainlayer)
        addcorn22 = np.zeros_like(tempmainlayer)
        addcorn33 = np.zeros_like(tempmainlayer)
        addcorn44 = np.zeros_like(tempmainlayer)

        addlayer11[:buf, :] = tempbufmat[buf + N :, buf : buf + N].copy()
        addlayer22[:, :buf] = tempbufmat[buf : buf + N, buf + N :].copy()
        addlayer33[-buf:, :] = tempbufmat[:buf, buf : buf + N].copy()
        addlayer44[:, -buf:] = tempbufmat[buf : buf + N, :buf].copy()

        addcorn11[:buf, :buf] = tempbufmat[buf + N :, buf + N :].copy()
        addcorn22[-buf:, -buf:] = tempbufmat[:buf, :buf].copy()
        addcorn33[:buf, -buf:] = tempbufmat[buf + N :, :buf].copy()
        addcorn44[-buf:, :buf] = tempbufmat[:buf, buf + N :].copy()

        tempmainlayer += (
                addlayer11
                + addlayer22
                + addlayer33
                + addlayer44
                + addcorn11
                + addcorn22
                + addcorn33
                + addcorn44
            )

        locslayers.append(tempmainlayer)

    mainlayer = np.zeros((N,N))
    for ll in locslayers:
        mainlayer += ll
    return mainlayer


#@jit(nopython=True, parallel=True)
def newstorm(locs1, mainlayer):
    buf = rad
    mat = np.zeros((N,N)) # matrix of zeros
    newlocx = np.random.randint(0, N) # generate new x coordinate
    newlocy = np.random.randint(0, N) # generate new y coordinate
    #newloc = np.asarray([newlocx, newlocy]) # new location coordinates
    #locs = np.vstack([locs, newloc]) # add to list of locations

    tempbufmat = np.zeros((N + 2 * rad, N + 2 * rad))
    tempbufmat[
            newlocx : newlocx + gaus.shape[0],
            newlocy : newlocy + gaus.shape[1],
        ] += gaus
    
    tempmainlayer = tempbufmat[buf : buf + N, buf : buf + N].copy()

    addlayer11 = np.zeros_like(tempmainlayer)
    addlayer22 = np.zeros_like(tempmainlayer)
    addlayer33 = np.zeros_like(tempmainlayer)
    addlayer44 = np.zeros_like(tempmainlayer)
    addcorn11 = np.zeros_like(tempmainlayer)
    addcorn22 = np.zeros_like(tempmainlayer)
    addcorn33 = np.zeros_like(tempmainlayer)
    addcorn44 = np.zeros_like(tempmainlayer)

    addlayer11[:buf, :] = tempbufmat[buf + N :, buf : buf + N].copy()
    addlayer22[:, :buf] = tempbufmat[buf : buf + N, buf + N :].copy()
    addlayer33[-buf:, :] = tempbufmat[:buf, buf : buf + N].copy()
    addlayer44[:, -buf:] = tempbufmat[buf : buf + N, :buf].copy()

    addcorn11[:buf, :buf] = tempbufmat[buf + N :, buf + N :].copy()
    addcorn22[-buf:, -buf:] = tempbufmat[:buf, :buf].copy()
    addcorn33[:buf, -buf:] = tempbufmat[buf + N :, :buf].copy()
    addcorn44[-buf:, :buf] = tempbufmat[:buf, buf + N :].copy()

    tempmainlayer += (
            addlayer11
            + addlayer22
            + addlayer33
            + addlayer44
            + addcorn11
            + addcorn22
            + addcorn33
            + addcorn44
        )
    ##
    tup = []
    tup.append(newlocx)
    tup.append(newlocy)
    newdur = np.random.randint(1, 15)
    newper = np.random.randint(newdur, 4*newdur+1)
    tup.append(newdur)
    tup.append(newper)
    tup.append(0)
    locs1 = np.vstack([locs1, np.asarray(tup)])
    ##
    
    return tempmainlayer, tup


@jit(nopython=True, parallel=True)
def genlocs(num, N):
    """
    Generates a list of coordinates, storm duration, storm period, and tclock.

        - Made it more pythonic and faster - D
    """
    
    locs = np.random.randint(0,N+1, (num, 2))
    newdur = np.round(np.random.normal(tstf, 2, (num, 1)))
    newper = np.round(np.random.normal(tstpf, 2, (num, 1)))

    final = np.append(locs, newdur, axis=1)
    final = np.append(final, newper, axis=1)
    final = np.append(final, np.zeros((num, 1)), axis=1).astype(np.int64)

    return final
