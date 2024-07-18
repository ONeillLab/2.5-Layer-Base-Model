import numpy as np
import matplotlib.pyplot as plt
import helper_functions_MPI as hf
from name_list_uranus import *



ts = np.linspace(0,(84*365*24*60*60)*f0, 1000)
ys = hf.seasonaltrad(ts)

print(trad0f, 0.9*trad0f)

plt.plot(ts, ys)
#plt.xlabel("Non-Dimensional time, t")
#plt.ylabel("Non-Dimensional ouput")
#plt.title("Graph of Forcing funciton B(t)")
plt.show()