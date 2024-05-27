import numpy as np
import math
from name_list import dx
from numba import jit
from name_list import *


@jit(nopython=True, parallel=True)
def gauss(x, y, L):
    """
    Generates a Gaussian.
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


#@jit(nopython=True, parallel=True)
def viscND(vel, Re, n):
    """
    n is exponent of Laplacian operator
    Where visc term is nu*(-1)^(n+1) (\/^2)^n
    so for regular viscosity n = 1, for hyperviscosity n=2

    TODO: for n=1 nu is not defined...
    """

    if n == 1:

        field = (
            -4 * vel
            + np.roll(vel, 1, axis=0)
            + np.roll(vel, -1, axis=0)
            + np.roll(vel, 1, axis=1)
            + np.roll(vel, -1, axis=1)
        )
        field = (n / dx**2) * field

        # in Morgan's code the n here is 'nu', but that's never defined; I think it's a typo

        # replaced n==1 case with my code

    if n == 2:

        field = (
            2 * np.roll(vel, (1, 1), axis=(0, 1))
            + 2 * np.roll(vel, (1, -1), axis=(0, 1))
            + 2 * np.roll(vel, (-1, 1), axis=(0, 1))
            + 2 * np.roll(vel, (-1, -1), axis=(0, 1))
            - 8 * np.roll(vel, 1, axis=0)
            - 8 * np.roll(vel, -1, axis=0)
            - 8 * np.roll(vel, 1, axis=1)
            - 8 * np.roll(vel, -1, axis=1)
            + np.roll(vel, 2, axis=0)
            + np.roll(vel, -2, axis=0)
            + np.roll(vel, 2, axis=1)
            + np.roll(vel, -2, axis=1)
            + 20 * vel
        )

        field = -1 / Re * (1 / dx**4) * field

        return field



### New pairshapeN2 function. Generates Gaussians using entire domain instead of creating sub-domains. (Daniel) ###
@jit(nopython=True, parallel=True)
def pairshapeN2(locs, x, y, Br2, Wsh, N):

    wlayer = np.zeros_like(x).astype(np.float64)
    #x,y = np.meshgrid(np.arange(0,N), np.arange(0,N))
    
    for loc in locs:
        layer = Wsh * np.exp( - (Br2*dx**2)/0.3606 * ( (x-loc[0])**2 + (y-loc[1])**2))
        wlayer = wlayer + layer

    return wlayer


@jit(nopython=True, parallel=True)
def BernN2(u1, v1, u2, v2, gm, c22h, c12h, h1, h2, ord):
    """
    Bernoulli
    """
    #B1 = c12h * h1 + c22h * h2 + 0.25 * (u1**2 + np.roll(u1, -1, axis=1)**2 + v1**2 + np.roll(v1, -1, axis=0)**2)
    B1 = c12h * h1 + c22h * h2 + 0.25 * (u1**2 + u1[:,r]**2 + v1**2 + v1[r,:]**2)


    #B2 = gm * c12h * h1 + c22h * h2 + 0.25 * (u1**2 + np.roll(u1, -1, axis=1)**2 + v1**2 + np.roll(v1, -1, axis=0)**2)
    B2 = gm * c12h * h1 + c22h * h2 + 0.25 * (u1**2 + u1[:,r]**2 + v1**2 + v1[r,:]**2)

    return B1, B2


@jit(nopython=True, parallel=True)
def xflux(f, u):  # removed dx, dt from input
    #fl = np.roll(f, 1, axis=1)
    fl = f[:,l]
    fr = f

    fa = 0.5 * u * (fl + fr)

    return fa


@jit(nopython=True, parallel=True)
def yflux(f, v):  # removed dx, dt from input
    #fl = np.roll(f, 1, axis=0)
    fl = f[l,:]
    fr = f

    fa = 0.5 * v * (fl + fr)

    return fa

