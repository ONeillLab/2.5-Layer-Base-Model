import numpy as np
import matplotlib.pyplot as plt
import helper_functions_MPI as hf
from name_list_uranus import *



ts = np.linspace(0,(84*365*24*60*60)*f0, 1000)
ys = hf.seasonaltrad(ts)

print(hf.seasonalH1H2(TSEASONf), TSEASONf)

plt.plot(ts, ys)
plt.show()