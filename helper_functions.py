import numpy as np
from numba import jit
from name_list import *


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


"""
    Everything past this point is old/redundant code

def Axl(f, l, r):

    fa = 0.5 * (f + f[:, l - 1])
    return fa


def Ayl(f, l, r):

    fa = 0.5 * (f + f[l - 1, :])
    return fa

    
def pairshapeN2(locs, x, y, Br2, Wsh, N):
    #Create Gaussians on smaller scales and then convolve them with the weather layer.
    rad = int(np.ceil(np.sqrt(1 / Br2) / dx))
    xg, yg = np.meshgrid(range(-rad, rad + 1), range(-rad, rad + 1))
    gaus = Wsh * np.exp(-(Br2 * dx**2) / 0.3606 * ((xg + 0.5) ** 2 + (yg + 0.5) ** 2))

    wlayer = np.zeros(x.shape)

    buf = rad
    bufmat = np.zeros((N + 2 * rad, N + 2 * rad))
    nlocs = locs + rad

    corners = nlocs - rad

    for jj in range(locs.shape[0]):
        bufmat[
            corners[jj, 0] : corners[jj, 0] + gaus.shape[0],
            corners[jj, 1] : corners[jj, 1] + gaus.shape[1],
        ] += gaus

    wlayer = bufmat[buf : buf + N, buf : buf + N]

    addlayer1 = np.zeros_like(wlayer)
    addlayer2 = np.zeros_like(wlayer)
    addlayer3 = np.zeros_like(wlayer)
    addlayer4 = np.zeros_like(wlayer)
    addcorn1 = np.zeros_like(wlayer)
    addcorn2 = np.zeros_like(wlayer)
    addcorn3 = np.zeros_like(wlayer)
    addcorn4 = np.zeros_like(wlayer)

    addlayer1[:buf, :] = bufmat[buf + N :, buf : buf + N]
    addlayer2[:, :buf] = bufmat[buf : buf + N, buf + N :]
    addlayer3[-buf:, :] = bufmat[:buf, buf : buf + N]
    addlayer4[:, -buf:] = bufmat[buf : buf + N, :buf]

    addcorn1[:buf, :buf] = bufmat[buf + N :, buf + N :]
    addcorn2[-buf:, -buf:] = bufmat[:buf, :buf]
    addcorn3[:buf, -buf:] = bufmat[buf + N :, :buf]
    addcorn4[-buf:, :buf] = bufmat[:buf, buf + N :]

    wlayer += (
        addlayer1
        + addlayer2
        + addlayer3
        + addlayer4
        + addcorn1
        + addcorn2
        + addcorn3
        + addcorn4
    )

    layersum = np.sum(wlayer)  # redundant ?
    return wlayer
"""
