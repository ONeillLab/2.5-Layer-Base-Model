# 2.5-Layer-Base-Model
Daniel and I's Python translation of Dr. O'Neill's 2.5 layer shallow water model from [https://doi.org/10.1175/JAS-D-15-0314.1]


Bug List:

    - When setting dx really fine, i.e dx = 1/25, the simulation becomes unstable
        - Caused by the gaussian creation and floating point error in pairshapeND


The "boundary" issue can be fixed by increasing size of the regions on which the gaussians are created. 
Can also be fixed by just using the entire space to create gaussians.
However both of these still leave the problem of the error around each gaussian at about 0-3 std away.
