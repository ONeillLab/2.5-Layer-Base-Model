import numpy as np
from name_list_uranus import *
import time
import sys


"""
Seasonal forcing helperfunctions. These will do all the calculations on how different parameters evolve with time

seasonalH1: Calculates H1 as a function of time

seasonalH1H2: Calculates the new H1H2 ratio for a given time

seasonaltrad: Calculates the new radiative timescale for a given time

"""

def seasonal_forcing(t):
    return np.exp((-(t-seasperf)**2)/(2*seasstdf**2)) + np.exp((-(t)**2)/(2*seasstdf**2))

def seasonalH1(t):
    return deltaH1 * seasonal_forcing(t) + 1

def seasonalH1H2(t):
    return H1H2 + (seasonalH1(t) - 1)

def seasonaltrad(t):
    return (1 - deltatrad * seasonal_forcing(t))*trad0f


def pairfieldN2(L, h1, wlayer):
    """
    Creates the weather matrix for the storms, S_st in paper.
    """
    
    voldw = np.sum(wlayer) * dx**2
    area = L**2
    wcorrect = voldw / area
    Wmat = wlayer - wcorrect
    return Wmat


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
def pairshapeN2(locs, t, x, y, offset):

    wlayer = np.zeros_like(x).astype(np.float64)

    resolution = round(np.sqrt(1/Br2)/dx)
    padding = 3*resolution

    #print(round(len(x)/2))

    xcenter = x[round(offset/2), round(offset/2)]
    ycenter = y[round(offset/2), round(offset/2)]

    for i in range(len(locs)):
        if np.abs(locs[i][0] - xcenter) < len(x)/2 + padding and np.abs(locs[i][1] - ycenter) < len(y)/2 + padding:
            if (t-locs[i][-1]) <= locs[i][2] or t == 0:
                #xloc = locs[i][0]
                #yloc = locs[i][1]
                #zonex = x[yloc-padding:yloc+padding, :][:, xloc-padding:xloc+padding]
                #zoney = y[yloc-padding:yloc+padding, :][:, xloc-padding:xloc+padding]
                layer = Wsh * np.exp( - (Br2*dx**2)/0.3606 * ( (x-locs[i][0])**2 + (y-locs[i][1])**2))
                #wlayer[yloc-padding:yloc+padding, :][:, xloc-padding:xloc+padding] += layer
                wlayer += layer

    return wlayer

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

def xflux(f, u):  # removed dx, dt from input
    fl = f[:,l]
    fr = f

    fa = 0.5 * u * (fl + fr)

    return fa

def yflux(f, v):  # removed dx, dt from input
    fl = f[l,:]
    fr = f

    fa = 0.5 * v * (fl + fr)

    return fa

def calculate_KE(u1,u2,v1,v2,h1,h2):
    first = p1p2*H1H2*h1*(u1**2 + v1**2)
    second = h2*(u2**2 + v2**2)

    return 0.5 * np.sum(first + second)


def calculate_APE(h1, h2):
    first = 0.5*p1p2*H1H2*c12h*(h1-1)**2
    second = 0.5*c22h*(h2-1)**2
    third = p1p2*H1H2*(c22h/c12h)*c12h*(h1-1)*(h2-1)

    return np.sum(first + second + third)


######### new helper functions for new storm forcing ##########
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

### New helper function for MPI, splits an array into even pieces with 2 elements of padding. Includes wrapping ###
def split(arr, offset, ranks, rank):

    #timer = time.time()

    rows, cols = arr.shape
    ind = np.where(ranks == rank)
    i = offset*ind[0][0]
    j = offset*ind[1][0]
    
    n = offset

    # Initialize an (n+4)x(n+4) result array
    result = np.full((n+4, n+4), np.nan, dtype=arr.dtype)
    
    #Fill the center (n x n) part with the original n x n block
    for block_i in range(n):
        for block_j in range(n):
            result[block_i + 2, block_j + 2] = arr[(i + block_i) % rows, (j + block_j) % cols]
    
    #result[2:n+2,:][:,2:n+2] = arr[i:i+n,:][:,j:j+n]

    # Fill the edges 2 cells away
    for block_j in range(n):
        result[0, block_j + 2] = arr[(i - 2) % rows, (j + block_j) % cols]  # Top edge
        result[1, block_j + 2] = arr[(i - 1) % rows, (j + block_j) % cols]  # 1 cell above the block
        result[n + 2, block_j + 2] = arr[(i + n) % rows, (j + block_j) % cols]  # 1 cell below the block
        result[n + 3, block_j + 2] = arr[(i + n + 1) % rows, (j + block_j) % cols]  # Bottom edge
    
    for block_i in range(n):
        result[block_i + 2, 0] = arr[(i + block_i) % rows, (j - 2) % cols]  # Left edge
        result[block_i + 2, 1] = arr[(i + block_i) % rows, (j - 1) % cols]  # 1 cell left of the block
        result[block_i + 2, n + 2] = arr[(i + block_i) % rows, (j + n) % cols]  # 1 cell right of the block
        result[block_i + 2, n + 3] = arr[(i + block_i) % rows, (j + n + 1) % cols]  # Right edge
    
    # Fill the corners 2 cells away
    result[0, 0] = arr[(i - 2) % rows, (j - 2) % cols]  # Top-left corner
    result[0, 1] = arr[(i - 2) % rows, (j - 1) % cols]  # Top-left 1 cell right
    result[1, 0] = arr[(i - 1) % rows, (j - 2) % cols]  # Top-left 1 cell down
    result[1, 1] = arr[(i - 1) % rows, (j - 1) % cols]  # Top-left 1 cell down-right

    result[0, n + 3] = arr[(i - 2) % rows, (j + n + 1) % cols]  # Top-right corner
    result[0, n + 2] = arr[(i - 2) % rows, (j + n) % cols]  # Top-right 1 cell left
    result[1, n + 3] = arr[(i - 1) % rows, (j + n + 1) % cols]  # Top-right 1 cell down
    result[1, n + 2] = arr[(i - 1) % rows, (j + n) % cols]  # Top-right 1 cell down-left

    result[n + 3, 0] = arr[(i + n + 1) % rows, (j - 2) % cols]  # Bottom-left corner
    result[n + 3, 1] = arr[(i + n + 1) % rows, (j - 1) % cols]  # Bottom-left 1 cell right
    result[n + 2, 0] = arr[(i + n) % rows, (j - 2) % cols]  # Bottom-left 1 cell up
    result[n + 2, 1] = arr[(i + n) % rows, (j - 1) % cols]  # Bottom-left 1 cell up-right

    result[n + 3, n + 3] = arr[(i + n + 1) % rows, (j + n + 1) % cols]  # Bottom-right corner
    result[n + 3, n + 2] = arr[(i + n + 1) % rows, (j + n) % cols]  # Bottom-right 1 cell left
    result[n + 2, n + 3] = arr[(i + n) % rows, (j + n + 1) % cols]  # Bottom-right 1 cell up
    result[n + 2, n + 2] = arr[(i + n) % rows, (j + n) % cols]  # Bottom-right 1 cell up-left
    
        
    return result


###
def combine(mats, offset, ranks, size):
    mats = np.array(mats[1:size+1])

    matsnew = np.reshape(mats[:, 2:subdomain_size+2, :][:, :, 2:subdomain_size+2], (int(np.sqrt(size)),int(np.sqrt(size)),subdomain_size,subdomain_size))

    mat = np.zeros((N,N))

    for i in range(1,size+1):
        ind = np.where(ranks == i)
        mat[offset*ind[0][0] : offset*ind[0][0] + offset, :][:, offset*ind[1][0] : offset*ind[1][0] + offset] = matsnew[ind[0][0], ind[1][0]]

    return mat


def get_surrounding_points(arr, i, j):
    surrounding_indices = [(-1, -1), (-1, 0), (-1, 1),
                           (0, -1),         (0, 1),
                           (1, -1), (1, 0), (1, 1)]
    
    rows, cols = arr.shape
    surrounding_points = []
    just_ranks = []
    
    for di, dj in surrounding_indices:
        ni, nj = (i + di) % rows, (j + dj) % cols
        surrounding_points.append((ni-i, nj-j, arr[ni, nj]))
        just_ranks.append(arr[ni,nj])
    
    return surrounding_points, set(just_ranks)
 