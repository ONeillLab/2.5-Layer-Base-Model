import numpy as np
from numba import jit
from name_list_jupiter import *


@jit(nopython=True, parallel=True)
def gauss(x, y, L):
    """
    Generates a Gaussian
    """
    g = np.exp(-0.5 * (x**2 + y**2) / L**2)
    return g


@jit(nopython=True, parallel=True)
def paircountN2(num, N):
    """
    Generates a list of coordinate pairs.
    """

    locs = np.ceil(np.random.rand(num, 2) * N).astype(np.float64)
    return locs


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

