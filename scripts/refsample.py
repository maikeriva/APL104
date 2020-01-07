"""
 Reference sample calibrated on K. H. Kim et alii report
"""

import numpy as np
import apl104lib as apl104

# Prepare reference test sample
refsample=apl104.Sample1D(5,400,256)
refsample.specienames=['La','Zr','Co','O','Li']
# Species order: La,Zr,Co,O,Li
refsample.z=np.array([3,2,3,-2,1])
# c1 is LCO
refsample.c1_bulk=np.array([0,0,1,2,1])/4
# c2 is LLZO
refsample.c2_bulk=np.array([3,2,0,12,7])/24

# Set phase coefficients from calibration
# ~ refsample.L=1.47e-05
refsample.L=0.0006374499539135993
refsample.W0=50
refsample.fc=0.1

# Set diffusion coefficients from calibration
guess=np.array([3.10695286e-02, 3.71141509e-02, 2.02426242e-02, 5.93923469e-02,
       3.72455688e-03, 9.99775806e-01, 1.16410815e-01, 4.71333483e-02,
       1.00000000e-06, 4.04688847e-03])
refsample.D1=guess[0:5]
refsample.D2=guess[5:10]

# Set initial condition, symmetric for fourier-friendliness
refsample.c[...]=refsample.c1_bulk
refsample.c[int(refsample.res*1/4):int(refsample.res*3/4)]=refsample.c2_bulk
refsample.p[...]=1
refsample.p[int(refsample.res*1/4):int(refsample.res*3/4)]=0

# Perform consistency checks and update structure
refsample.update()
