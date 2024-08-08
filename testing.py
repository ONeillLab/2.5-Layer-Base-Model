import numpy as np
import matplotlib.pyplot as plt
import helper_functions_MPI as hf
from name_list_general import *



ts = np.linspace(0,5000)
ys = hf.seasonaltrad(ts)

plt.plot(ts, ys)
plt.xlabel("Non-Dimensional time, t")
plt.ylabel("Non-Dimensional ouput")
plt.title("Graph of profile ofForcing funciton B(t)")
plt.show()