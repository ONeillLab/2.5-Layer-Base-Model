# 2.5-Layer-Base-Model
Daniel and I's Python translation of Dr. O'Neill's 2.5 layer shallow water model from [https://doi.org/10.1175/JAS-D-15-0314.1]


Bug List:
    - When setting dx really fine, i.e dx = 1/25, the simulation becomes unstable.
        - Caused by floating point errors in the pairshape helper function. 
