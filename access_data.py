import numpy as np
from netCDF4 import Dataset
from name_list import *


def display_data(data_name):
    """
    Display metadata for convenience 
    """
    rootgroup = Dataset(data_name, "r")
    print(rootgroup)
    print(rootgroup.dimensions)
    print(rootgroup.ncattrs)
    print(rootgroup.variables)
    rootgroup.close()


def create_file(data_name):
    """
    Creates a netCDF file on the disk
    """
    rootgroup = Dataset(data_name, "a") # creates the file
    rootgroup.tmax = tmax # creates attributes
    rootgroup.c22h = c22h
    rootgroup.c12h = c12h
    rootgroup.H1H2 = H1H2
    rootgroup.Bt = Bt
    rootgroup.Br2 = Br2
    rootgroup.p1p2 = p1p2
    rootgroup.tstf = tstf
    rootgroup.tstpf = tstpf
    rootgroup.tradf = tradf
    rootgroup.dragf = dragf
    rootgroup.Ar = Ar
    rootgroup.Re = Re
    rootgroup.Wsh = Wsh
    rootgroup.gm = gm 
    rootgroup.aOLd = aOLd 
    rootgroup.L = L 
    rootgroup.num = num 
    rootgroup.deglim = deglim  
    rootgroup.Lst = Lst
    rootgroup.AB = AB  
    rootgroup.layers = layers  
    rootgroup.n = n 
    rootgroup.kappa = kappa
    rootgroup.ord = ord 
    rootgroup.spongedrag1 = spongedrag1
    rootgroup.spongedrag2 = spongedrag2
    rootgroup.dx = dx
    rootgroup.dt = dt
    rootgroup.dtinv = dtinv
    rootgroup.sampfreq = sampfreq
    rootgroup.tpl = tpl
    rootgroup.N = N
    rootgroup.L = L
    rootgroup.EpHat = EpHat

    rootgroup.createDimension("tu1", None) # dimensions
    rootgroup.createDimension("tu2", None) # dimensions
    rootgroup.createDimension("tv1", None) # dimensions
    rootgroup.createDimension("tv2", None) # dimensions
    rootgroup.createDimension("th1", None) # dimensions
    rootgroup.createDimension("th2", None) # dimensions
    rootgroup.createDimension("tlocs", None) # dimensions
    rootgroup.createDimension("x", N) 
    rootgroup.createDimension("y", N)
    rootgroup.createDimension("xlocs", num) 
    rootgroup.createDimension("ylocs", 5)
    rootgroup.createDimension("time", None)

    u1mat = rootgroup.createVariable("u1mat", "f8", ("tu1", "x", "y",),compression='zlib') # variables (list of arrays)
    u2mat = rootgroup.createVariable("u2mat", "f8", ("tu2", "x", "y",),compression='zlib')
    v1mat = rootgroup.createVariable("v1mat", "f8", ("tv1", "x", "y",),compression='zlib')
    v2mat = rootgroup.createVariable("v2mat", "f8", ("tv2", "x", "y",),compression='zlib')
    h1mat = rootgroup.createVariable("h1mat", "f8", ("th1", "x", "y",),compression='zlib')
    h2mat = rootgroup.createVariable("h2mat", "f8", ("th2", "x", "y",),compression='zlib')
    locsmat = rootgroup.createVariable("locsmat", "f8", ("tlocs", "xlocs", "ylocs",),compression='zlib')
    ts = rootgroup.createVariable("ts", "f8", ("time",),compression='zlib')

    rootgroup.close()


### Depricated ? ###
def store_data(data_name, u1mat, u2mat, h1mat, h2mat, v1mat, v2mat, locsmat, ts):
    """
    Stores the output of a simulation
    """
    rootgroup = Dataset(data_name, "a")
    rootgroup.variables["u1mat"][:] = u1mat#.astype("float64") 
    rootgroup.variables["u2mat"][:] = u2mat#.astype("float64") 
    rootgroup.variables["v1mat"][:] = v1mat#.astype("float64") 
    rootgroup.variables["v2mat"][:] = v2mat#.astype("float64") 
    rootgroup.variables["h1mat"][:] = h1mat#.astype("float64") 
    rootgroup.variables["h2mat"][:] = h2mat#.astype("float64") 
    rootgroup.variables["locsmat"][:] = locsmat#.astype("float64") 
    rootgroup.variables["ts"][:] = ts#.astype("float64") 
    rootgroup.time = ts[-1]

    rootgroup.close()


def last_timestep(data_name):
    """
    Takes a file and extracts data of the last timestep
    """
    rootgroup = Dataset(data_name, "r")
    if len(np.asarray(rootgroup.variables["u1mat"])) != 0:
        u1 = np.asarray(rootgroup.variables["u1mat"][-1])
        u2 = np.asarray(rootgroup.variables["u2mat"][-1])
        v1 = np.asarray(rootgroup.variables["v1mat"][-1])
        v2 = np.asarray(rootgroup.variables["v2mat"][-1])
        h1 = np.asarray(rootgroup.variables["h1mat"][-1])
        h2 = np.asarray(rootgroup.variables["h2mat"][-1])
        locs = np.asarray(rootgroup.variables["locsmat"][-1])
        lasttime = rootgroup.__dict__["time"]
    rootgroup.close()
    return u1, u2, v1, v2, h1, h2, locs, lasttime


def storetime(t, file_name):
    rootgroup = Dataset(file_name, "a")
    rootgroup.time = t
    rootgroup.close()

def storedata(xmat, x, file_name):
    rootgroup = Dataset(file_name, "a")
    rootgroup.variables[xmat][rootgroup.variables[xmat].shape[0],:,:] = x.astype("float64") 
    rootgroup.close()


def save_data(u1,u2,v1,v2,h1,h2,locs,t,lasttime,file_name):
    storedata("u1mat", u1, file_name) # storedata takes u1 and appends it to the variable "u1mat"
    storedata("u2mat", u2, file_name)
    storedata("h1mat", h1, file_name)
    storedata("h2mat", h2, file_name)
    storedata("v1mat", v1, file_name)
    storedata("v2mat", v2, file_name)
    storedata("locsmat", locs, file_name)
    storetime(t, file_name)