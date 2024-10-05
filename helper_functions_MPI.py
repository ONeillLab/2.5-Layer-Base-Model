import numpy as np
from name_list_betadrift import *
import time
import sys


def pairfieldN2(L, h1, wlayer):
    """
    Creates the weather matrix for the storms, S_st in paper.

    Calculates the subsidence correction.
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

    wlayer = np.zeros_like(x).astype(np.float64) # Initializes the weather layer

    resolution = round(np.sqrt(1/Br2)/dx) # The number of grid points per storm
    padding = 3*resolution # The padding outside of which the gaussian will be terminated to 0


    xcenter = x[round(offset/2), round(offset/2)] # x,y center of the current subgrid (used for MPI)
    ycenter = y[round(offset/2), round(offset/2)]

    for i in range(len(locs)):
        #Checking if the current storm placement is close enough to or within the current subgrid and calculates its contribution
        if np.abs(locs[i][0] - xcenter) < len(x)/2 + padding and np.abs(locs[i][1] - ycenter) < len(y)/2 + padding:
            # Check if storm is dormant, tstf < lifetime < tstpf
            if (t-locs[i][-1]) <= locs[i][2] or t == 0:

                layer = Wsh * np.exp( - (Br2*dx**2)/0.3606 * ( (x-locs[i][0])**2 + (y-locs[i][1])**2)) # Adds the storm to the weather layer
                wlayer += layer

    return wlayer


def BernN2(u1, v1, u2, v2, gm, c22h, c12h, h1, h2, ord):
    """
    Bernoulli 
    """
    B1 = c12h * h1 + c22h * h2 + 0.25 * (u1**2 + u1[:,r]**2 + v1**2 + v1[r,:]**2)

    # Allows one to use the old bugged version of the code. Used for testing, actual simulations should always have fixed == True in name_list.
    if fixed == False:
        B2 = gm * c12h * h1 + c22h * h2 + 0.25 * (u1**2 + u1[:,r]**2 + v1**2 + v1[r,:]**2)
    else:
        B2 = gm * c12h * h1 + c22h * h2 + 0.25 * (u2**2 + u2[:,r]**2 + v2**2 + v2[r,:]**2)

    return B1, B2



def calculate_KE(u1,u2,v1,v2,h1,h2, H1H2, p1p2):
    """
    Calculates the kinetic energy of a configuration of the simluation. These equations can be found in O'Neill et al. (2016).
    """
    first = p1p2*H1H2*h1*(u1**2 + v1**2)
    second = h2*(u2**2 + v2**2)

    return 0.5 * np.sum(first + second)


def calculate_APE(h1, h2, H1H2, p1p2, c12h, c22h):
    """
    Calculates the available potential energy of a configuration of the simulation. The equations can be found in O'Neill et al. (2016).
    """
    first = 0.5*p1p2*H1H2*c12h*(h1-1)**2
    second = 0.5*c22h*(h2-1)**2
    third = p1p2*H1H2*(c22h/c12h)*c12h*(h1-1)*(h2-1)

    return np.sum(first + second + third)


######### new helper functions for new storm forcing ##########
def genlocs(num, N, t):
    """
    Generates a list of coordinates, storm duration, storm period, and tclock.

        - Made it more pythonic and faster - D

    The "locs" list contains information about each storm present in the simulation, and all the information they need.
    """
    
    choices = np.random.randint(0, len(poslocs), num) 

    locs = poslocs[choices] # make random choices of loction from all possible storm locations (not in the sponge layer)

    locs = [[100,100]]
    
    newdur = np.round(np.random.normal(tstf, 2, (num, 1)))  # Pick durations for the storms from a normal distribution
    newper = np.round(np.random.normal(tstpf, 2, (num, 1))) # Pick periods for the storms from a normal distribution

    final = np.append(locs, newdur, axis=1) 
    final = np.append(final, newper, axis=1) # Create the final array of all storms.


    # Adding the current time so that the storm knows when it was created.
    if t == 0:
        final = np.append(final, np.round(np.random.normal(0, tstf, (num,1))), axis=1).astype(np.int64)
    else:
        final = np.append(final, np.ones((num, 1)) * t, axis=1).astype(np.int64)

    return final


def xflux(f, u):
    """
    Calculates flux of field f in the x direction
    """
    fl = f[:,l]
    fr = f

    fa = 0.5 * u * (fl + fr)

    return fa

def yflux(f, v):
    """
    Calculates flux of field f in the y direction
    """
    fl = f[l,:]
    fr = f

    fa = 0.5 * v * (fl + fr)

    return fa

### New helper functions for MPI ###

def split(arr, offset, ranks, rank):
    """
    Splits an array in to subarrays based on the number of ranks. It all wraps the subarrays with period boundary conditions on the
    main array. The subarrays have a padding of 2 on each side. These arrays are assigned from top to bottom, left to right.

    This is a key function for the MPI process, as it allows the rank 0 process to split and distribute dynamical data to each subprocess.
    """

    # finding the top left index of each rank
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


def combine(mats, offset, ranks, size):
    """
    Combines subarrays from each rank into one large array. Each rank calculates dynamics in each of its subarray, to get the large
    array back for saving in the NETCDF output they need to be recombined on rank 0.
    """

    mats = np.array(mats[1:size+1]) # discard the rank 0 contribution

    # line below discards the 2 cell padding around the entire subarray.
    matsnew = np.reshape(mats[:, 2:subdomain_size+2, :][:, :, 2:subdomain_size+2], (int(np.sqrt(size)),int(np.sqrt(size)),subdomain_size,subdomain_size))

    mat = np.zeros((N,N)) # initialize full array

    # add each subarray to the correct location in the new combined array, goes from top to bottom left to right by rank.
    for i in range(1,size+1):
        ind = np.where(ranks == i)
        mat[offset*ind[0][0] : offset*ind[0][0] + offset, :][:, offset*ind[1][0] : offset*ind[1][0] + offset] = matsnew[ind[0][0], ind[1][0]]

    return mat


def get_surrounding_points(arr, i, j):
    """
    Returns the points surrounding a point in an array, with periodic boundary conditions. 
    Used with MPI to find which ranks a cell has to communicate its dynamical data too to update subarray paddings.
    """

    # Array of relative index locations.
    surrounding_indices = [(-1, -1), (-1, 0), (-1, 1),
                           (0, -1),         (0, 1),
                           (1, -1), (1, 0), (1, 1)]
    
    rows, cols = arr.shape
    surrounding_points = []
    just_ranks = []
    
    # Appends the elements at each surrounding index, however with boundary wrapping.
    for di, dj in surrounding_indices:
        ni, nj = (i + di) % rows, (j + dj) % cols # Calculate the wrapped index
        surrounding_points.append((ni-i, nj-j, arr[ni, nj])) # append the relative location (index) the rank
        just_ranks.append(arr[ni,nj]) # append just the rank not the location.
    
    return surrounding_points, set(just_ranks) # return the relative locations of each rank, and a set of corresponding ranks.