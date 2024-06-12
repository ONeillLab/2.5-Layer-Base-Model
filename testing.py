import numpy as np
from name_list import *
from numba import jit


@jit(nopython=True, parallel=True)
def testing(num):

    choices = np.random.randint(0, len(possibleLocs), num)

    locs = poslocs[choices]

    return locs


print(testing(10))