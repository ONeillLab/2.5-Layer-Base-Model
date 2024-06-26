import numpy as np
from numba import jit, prange
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
def pairshapeN2(locs, t):

    wlayer = np.zeros_like(x).astype(np.float64)
    
    for i in prange(len(locs)):
        if (t-locs[i][-1]) <= locs[i][2] or t == 0:
            layer = Wsh * np.exp( - (Br2*dx**2)/0.3606 * ( (x-locs[i][0])**2 + (y-locs[i][1])**2))
            wlayer += layer

    return wlayer


@jit(nopython=True, parallel=True)
def BernN2(u1, v1, u2, v2, gm, c22h, c12h, h1, h2, ord):
    """
    Bernoulli
    """
    B1 = c12h * h1 + c22h * h2 + 0.25 * (u1**2 + u1[:,r]**2 + v1**2 + v1[r,:]**2)


    if fixed == False:
        B2 = gm * c12h * h1 + c22h * h2 + 0.25 * (u1**2 + u1[:,r]**2 + v1**2 + v1[r,:]**2)
    else:
        B2 = gm * c12h * h1 + c22h * h2 + 0.25 * (u2**2 + u2[:,r]**2 + v2**2 + v2[r,:]**2)

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

@jit(nopython=True, parallel=True)
def genlocs(num, N, t):
    """
    Generates a list of coordinates, storm duration, storm period, and tclock.

        - Made it more pythonic and faster - D
    """
    
    choices = np.random.randint(0, len(poslocs), num)

    locs = poslocs[choices]
    
    newdur = np.round(np.random.normal(tstf, 2, (num, 1)))
    newper = np.round(np.random.normal(tstpf, 2, (num, 1)))

    final = np.append(locs, newdur, axis=1)
    final = np.append(final, newper, axis=1)

    if t == 0:
        final = np.append(final, np.round(np.random.normal(0, tstf, (num,1))), axis=1).astype(np.int64)
    else:
        final = np.append(final, np.ones((num, 1)) * t, axis=1).astype(np.int64)

    return final