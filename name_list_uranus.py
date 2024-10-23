import math
import numpy as np

fixed = True
saving = True
seasonalsim = False
season = "winter" # "summer" for summer settings and "winter" for winter settings


TSEASON = 42 # Time in Uranian year, 84 will be summer solstice for the north pole, while 42 will be south pole solstice

num_processors = 65 #10

tmax = 20000
ani_interval = 100
sampfreq = 100
restart_name = None #"test1.nc" #'jupiter100724_7.nc'
new_name = 'test1.nc'

### Dimensional, collected from papers, used for normalization ###
f0 = 1.0124e-4    # coriolis parameter from Nasa planet facts [s]
a = 24973e3      # planetary radius from Nasa planet facts [m]
g = 9.01         # Uranus' gravity from Nasa planet facts [m/s^2] 
Ld2 = 1200e3      # 2nd baroclinic Rossby deformation radius [m] from O'Neill
drag = 100000     # Cumulus Drag (placeholder)

### Dimensional Seasonal Forcing Parameters ###
T0 = 70 # From Clement (2024) (at 0.6 Bar) [K]
Tamp = 4.1 # From Milcareck (2024) (at 0.3 Bar) [K]
seasper = 84 # Nasa Facts [year]
seasstd = 10 # Hueso et al. (big estimate) [year]
M = 2.3e-3 # mean molecular weight of the atmosphere (Clement et al.) [kg/mol]
R = 8.316 # ideal gas constant
p0 = 1.1 # Top of upper layer from Sromovsky (2024) [bar]
ptop = 0.7
scaleHeight = (R*T0)/(M*g) # The scale height of the uranian atmosphere [m]
RM = R/M
H2 = scaleHeight * np.log(p0/0.7)
H10 = scaleHeight * np.log(p0/0.45) - H2
deltaH1 = (Tamp*(2*RM))/g
cp = 10200 # Heat capacity of methane at (3 Bar) from Milcareck
sigma = 5.670e-8 # Stefan-Boltzmann constant
eps = 0.3 # emissivity, estimated
trad0 = (cp*ptop*(10**5)) / (4*g*sigma*eps*T0**3)
deltatrad = (3*Tamp/T0)*trad0
Wst = 0.015 # RMS vertical velocity at 0.7 bar. (Clement et al.) [m/s]


TIMESCALING = 50
seasper = seasper/TIMESCALING
seasstd = seasstd/TIMESCALING
trad0 = trad0/TIMESCALING
TSEASON = TSEASON/TIMESCALING

### Dimensionless Seasonal Forcing Parameters ###
seasperf = round((seasper*365*24*60*60)*f0)
seasstdf = round((seasstd*365*24*60*60)*f0)
deltaH1 = deltaH1/H10
trad0f = trad0*f0
deltatrad = (3*Tamp/T0)*trad0 / trad0
TSEASONf = (TSEASON*365*24*60*60)*f0

### Dimensional, Storm parameters ###
Rst = 2*300e3 #350e3       # Storm size [m] calculated from Sromovsky (2024) [m]
tst = 260000      # 3 day storm duration from Sromovsky (2024) [s]
tstp = tst*2 #100*24*60*60 #tst*2   # 100 day Period between forced storms (Clement)

### Dimensonal, Atmosphere parameters, these are not known and must be adjusted ###
p1p2 = 0.95
H1H2 = H10/H2

# Dimensional, Derived Parameters ###
c2 = Ld2 * f0 # Second baroclinic gravity wave speed

### ND Derived Parameters ###
tstf = round(tst*f0)
tstpf = round(tstp*f0)
dragf = drag*f0
Br2 = Ld2**2 / Rst**2   # Burger Number
c22h = 3 # ND 2nd baroclinic gravity wave speed squared
c12h = 4 # ND 1st baroclinic gravity wave speed squared
Bt = (Ld2**2)/(2*a**2) # scaled beta (for beta plane)
Ar = 0.20 # Calculated from Sromovsky
Re = 5e7
Wsh = Wst / (H10 * f0) # Calculated from O'Neill

if season == "summer":
    H1H2 = (1+deltaH1)*H1H2
    Wsh = Wsh / (1+deltaH1)
    tradf = (1-deltatrad)*trad0f
if season == "winter":
    tradf = trad0f

#### Derived Quantities ###
gm = p1p2*c22h/c12h*H1H2  # ND reduced gravity
aOLd = a/Ld2;             # ND planetary radius to deformation radius ratio
deglim = (np.pi/6)*1.25   # domain size [degrees]
L = 2*(deglim * a)/Ld2

num = round( (16*np.pi*aOLd**2 * deglim**2 * Ar) / (25*np.pi* 1/Br2) )

Lst = L * Ld2/Rst

################## engineering params ##########################
AB = 2  # order of Adams-Bashforth scheme (2 or 3)
layers = 2.5  # of layers (2 or 2.5)
n = 2  # order of Laplacian '2' is hyperviscosity
kappa = 1e-6
ord = 2  # must equal 1 for Glenn's order, otherwise for Sadourney's (squares before avgs)
spongedrag1 = 0.05
spongedrag2 = 0.05

EpHat = (
    ((1 / 2) * p1p2 * c12h + (1 / 2) * H1H2 * c22h - p1p2 * (c22h / c12h) * H1H2 * c12h)
    * H1H2
    * (Wsh * tstf) ** 2
    * (trad0f / tstpf)
    * (tstf / tstpf)
    * (Ar / np.sqrt(Br2))
)

#dx = 1 / 5 * round(min(1,L/Lst), 3)
N  = 640
dx = round(L/N,4)
dt = dx / (10 * c12h) #1 / (2**8) # CHANGED TO dx/(10*c12h) SO THAT dt CHANGES TO MATCH dx
dtinv = 1 / dt
tpl = round(sampfreq * dtinv)
tp1 = round(dtinv)

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