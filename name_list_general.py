import math
import numpy as np

fixed = True
saving = True
seasonalsim = False
season = "winter" # "summer" for summer settings and "winter" for winter settings

num_processors = 10

tmax = 5000
ani_interval = 100
sampfreq = 100
restart_name = 'testing_1.nc' #None
new_name = 'testing_2.nc'

### ND seasonal parameters ###
deltatrad = 0.1
deltaH1 = 0.1
seasperf = 5000

c22h = 3  # 9  # ND 2nd baroclinic gravity wave speed squared
c12h = 4  # 10  # ND 1st baroclinic gravity wave speed squared
H1H2 = 1  # ND upper to lower layer height ratio
Bt = (1**2) / 2 / (30**2)  # ND scaled beta Ld2^2/4a^2 ### adjust this
Br2 = 1  # 1.5  # ND scaled storm size: Burger number Ld2^2/Rst^2
p1p2 = 0.95  # ND upper to lower layer density ratio
tstf = 6  # 48  # ND storm duration tst*f0
tstpf = 15  # 60  # ND period between forced storms tstp*f0
trad0f = 2000  # ND Newtonian damping of layer thickness trad*f0
dragf = 1000000  # Cumulus drag time scale (Li and O'Neill) (D)
Ar = 0.20  # ND areal storm coverage
Re = 5e4  # ND Reynolds number
Wsh = 0.002 / 2  # ND convective Rossby number

#### Derived Quantities ###  
gm = p1p2 * c22h / c12h * H1H2  # ND reduced gravity
aOLd = np.sqrt(1 / Bt / 2)  # ND planetary radius to deformation radius ratio ### adjust this
deglim = (np.pi/6)*1.25 
L = deglim * aOLd  # ND num = ceil(numfrc.*L.^2./Br2)

num = round((aOLd**2 * np.pi**2 * Ar) / (36 * 1/Br2))

Lst = L * np.sqrt(Br2)  # Convert the length of domain per Ld2 to length of domain per Rst (Daniel)

################## engineering params ##########################

AB = 2  # order of Adams-Bashforth scheme (2 or 3)
layers = 2.5  # # of layers (2 or 2.5)
n = 2  # order of Laplacian '2' is hyperviscosity
kappa = 1e-6
ord = 2  # must equal 1 for Glenn's order, otherwise for Sadourney's (squares before avgs)
spongedrag1 = 0.01
spongedrag2 = 0.01


if season == "summer":
    H1H2 = (1+deltaH1)*H1H2
    Wsh = Wsh / (1+deltaH1)
    tradf = (1-deltatrad)*trad0f
if season == "winter":
    tradf = trad0f


#### Derived Quantities ###
gm = p1p2*c22h/c12h*H1H2            # ND reduced gravity
deglim = (np.pi/6)*1.25 # domain size [degrees]
L = 2*deglim * aOLd  # domain radius from pole, normalized by deformation radius
num = round(Ar*(L**2)*Br2/np.pi)    # number of storms

#Lst = L * Ld2/Rst

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
    * (trad0f / tstpf)
    * (tstf / tstpf)
    * (Ar / np.sqrt(Br2))
)


N  = 156
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
outerlim = L / 2.5 #- 0.5
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

print(EpHat)
