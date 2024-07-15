import math
import numpy as np

fixed = True
saving = True

num_processors = 5

tmax = 1000
ani_interval = 100
sampfreq = 100
restart_name = None#'jupiter100724_7.nc'
new_name = 'test1.nc'

### Dimensional, collected from papers, used for normalization ###
f0 = 1.0124e-4    # coriolis parameter from Nasa planet facts [s]
a = 24973e3      # planetary radius from Nasa planet facts [m]
g = 9.01         # Uranus' gravity from Nasa planet facts [m/s^2] 
Ld2 = 1200e3      # 2nd baroclinic Rossby deformation radius [m] from O'Neill
trad = 142858080  # 4.53 years from https://pds-atmospheres.nmsu.edu/education_and_outreach/encyclopedia/radiative_time_constant.htm [s]
drag = np.inf     # Cumulus Drag (placeholder)


### Dimensional, Storm parameters ###
Rst = 350e3       # Storm size [m] from Siegelman [m]
tst = 260000      # 3 day storm duration from Siegelman [s]
tstp = tst*1.1   # Period between forced storms (Guess)

### Dimensonal, Atmosphere parameters, these are not known and must be adjusted ###
p1p2 = 0.95
H1H2 = 1

# Dimensional, Derived Parameters ###
c2 = Ld2 * f0 # Second baroclinic gravity wave speed

### ND Derived Parameters ###
tstf = round(tst*f0)
tradf = trad*f0
tstpf = round(tstp*f0)
dragf = drag*f0
Br2 = Ld2**2 / Rst**2   # Burger Number
c22h = 3 # ND 2nd baroclinic gravity wave speed squared
c12h = 4 # ND 1st baroclinic gravity wave speed squared
Bt = (Ld2**2)/(2*a**2) # scaled beta (for beta plane)
Ar = 0.07 # From calcualtions
Re = 5e4
Wsh = 0.012 / 2 #Wst / (H1 * f0) Place holder


#### Derived Quantities ###
gm = p1p2*c22h/c12h*H1H2            # ND reduced gravity
aOLd = a/Ld2;             # ND planetary radius to deformation radius ratio
deglim = np.pi/6  # domain size [degrees]
L = 2*(deglim * a)/Ld2  # domain radius 30 deg from pole, normalized by deformation radius
num = round(Ar*(L**2)*Br2/np.pi)    # number of storms

Lst = L * Ld2/Rst

################## engineering params ##########################
AB = 2  # order of Adams-Bashforth scheme (2 or 3)
layers = 2.5  # of layers (2 or 2.5)
n = 2  # order of Laplacian '2' is hyperviscosity
kappa = 1e-6
ord = 2  # must equal 1 for Glenn's order, otherwise for Sadourney's (squares before avgs)
spongedrag1 = 0.01
spongedrag2 = 0.01

EpHat = (
    ((1 / 2) * p1p2 * c12h + (1 / 2) * H1H2 * c22h - p1p2 * (c22h / c12h) * H1H2 * c12h)
    * H1H2
    * (Wsh * tstf) ** 2
    * (tradf / tstpf)
    * (Ar / np.sqrt(Br2))
)

#dx = 1 / 5 * round(min(1,L/Lst), 3)
N  = 376
dx = round(L/N,4)
dt = dx / (10 * c12h) #1 / (2**8) # CHANGED TO dx/(10*c12h) SO THAT dt CHANGES TO MATCH dx
dtinv = 1 / dt
tpl = round(sampfreq * dtinv)

#N = math.ceil(L / dx)  # resolve
L = N * dx

x, y = np.meshgrid(np.arange(0.5, N + 0.5) * dx - L / 2, np.arange(0.5, N + 0.5) * dx - L / 2)
H = 1 + 0 * x
eta = 0 * x
h1 = (0 * x + 1).astype(np.float64)
h2 = (0 * x + 1).astype(np.float64)

# u grid 
x, y = np.meshgrid(np.arange(0, N) * dx - L / 2, np.arange(0.5, N + 0.5) * dx - L / 2)
u1 = 0 * x * y
u2 = u1

# v grid
x, y = np.meshgrid(np.arange(0.5, N + 0.5) * dx - L / 2, np.arange(0, N) * dx - L / 2)
v1 = 0 * x * y
v2 = v1

# zeta grid
x, y = np.meshgrid(np.arange(0, N) * dx - L / 2, np.arange(0, N) * dx - L / 2)
rdist = np.sqrt((x**2) + (y**2))
outerlim = L / 2 - 0.5
rlim = (rdist <= outerlim).astype(float)  # 1* converts the Boolean values to integers 1 or 0


sponge1 = np.ones(N) * np.maximum(rdist - outerlim, 0)
sponge1 = sponge1 / np.max(sponge1)  
spdrag1 = spongedrag1 * sponge1

sponge2 = np.ones(N) * np.maximum(rdist - outerlim, 0)
sponge2 = sponge2 / np.max(sponge1)
spdrag2 = spongedrag2 * sponge2

x,y = np.meshgrid(np.arange(0,N), np.arange(0,N))

### For rolling the arrays ###
subdomain_size = int(N // np.sqrt(num_processors-1)) + 4

l = np.concatenate((np.array([subdomain_size]), np.arange(1, subdomain_size)), axis=None) - 1
l2 = np.concatenate((np.arange(subdomain_size - 1, subdomain_size + 1), np.arange(1, subdomain_size - 1)), axis=None) - 1 
r = np.concatenate((np.arange(2, subdomain_size + 1), np.array([1])), axis=None) - 1
r2 = np.concatenate((np.arange(3, subdomain_size + 1), np.arange(1, 3)), axis=None) - 1

subdomain_size = int(N // np.sqrt(num_processors-1))

### Storm location picking ###

possibleLocs = np.array(list(zip(*np.where(rlim == 1))))

poslocs = []

for loc in possibleLocs:
    poslocs.append(np.array(loc))

poslocs = np.array(poslocs)


### FOR GRAPHING ###
lg = np.concatenate((np.array([N]), np.arange(1, N)), axis=None) - 1
lg2 = np.concatenate((np.arange(N - 1, N + 1), np.arange(1, N - 1)), axis=None) - 1 
rg = np.concatenate((np.arange(2, N + 1), np.array([1])), axis=None) - 1
rg2 = np.concatenate((np.arange(3, N + 1), np.arange(1, 3)), axis=None) - 1


print(EpHat, aOLd, N, dt)