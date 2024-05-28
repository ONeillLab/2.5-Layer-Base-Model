import numpy as np
import math
from name_list import dx


def gauss(x, y, L):
    """
    Generates a Gaussian.
    """
    g = np.exp(-0.5 * (x**2 + y**2) / L**2)
    return g


def paircountN2(num, N):
    """
    Generates a list of coordinate pairs.
    """

    locs = np.ceil(np.random.rand(num, 2) * N).astype(int)
    return locs


def pairfieldN2(L, h1, wlayer):
    """
    Creates the weather matrix for the storms, S_st in paper.
    """
    voldw = np.sum(np.sum(wlayer)) * dx**2
    area = L**2
    wcorrect = voldw / area
    Wmat = wlayer - wcorrect
    return Wmat


def Axl(f, l, r):
    """
    Redundant?
    """
    fa = 0.5 * (f + f[:, l - 1])
    return fa


def Ayl(f, l, r):
    """
    Redundant?
    """
    fa = 0.5 * (f + f[l - 1, :])
    return fa


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


def pairshapeN2(locs, x, y, Br2, Wsh, N):
    
    wlayer = np.zeros_like(x).astype(np.float64)
    
    for loc in locs:
        layer = Wsh * np.exp( - (Br2*dx**2)/0.3606 * ( (x-loc[0])**2 + (y-loc[1])**2))
        wlayer = wlayer + layer

    return wlayer


"""
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

def BernN2(u1, v1, u2, v2, gm, c22h, c12h, h1, h2, ord):
    """
    Bernoulli
    """
    if ord == 1:
        B1 = "broke"
        B2 = "broke"
    else:
        B1 = (
            c12h * h1
            + c22h * h2
            + 0.25
            * (
                (u1**2)
                + (np.roll(u1, -1, axis=1) ** 2)
                + (v1**2)
                + (np.roll(v1, -1, axis=0) ** 2)
            )
        )

        B2 = (
            gm * c12h * h1
            + c22h * h2
            + 0.25
            * (
                (u1**2)
                + (np.roll(u1, -1, axis=1) ** 2)
                + (v1**2)
                + (np.roll(v1, -1, axis=0) ** 2)
            )
        )

    return B1, B2


def xflux(f, u):  # removed dx, dt from input
    fl = np.roll(f, 1, axis=1)
    fr = f

    fa = 0.5 * u * (fl + fr)

    return fa


def yflux(f, v):  # removed dx, dt from input
    fl = np.roll(f, 1, axis=0)
    fr = f

    fa = 0.5 * v * (fl + fr)

    return fa
