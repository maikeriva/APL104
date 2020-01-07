"""
 Find diffusion coefficients
"""

import numpy as np
import apl104lib as apl104
from scipy.optimize import minimize
import copy

exec(open('refsample.py').read())
exec(open('expsample.py').read())

# Define problem
def problem(guess,expsample):
	# Prepare test sample
	sample=copy.deepcopy(refsample)
	# Set guesses in sample
	sample.D1=guess[0:5]
	sample.D2=guess[5:10]
	# Solve problem
	apl104.rfftsolve(sample,0.1,int(3600/0.1))
	# Evaluate error
	return np.sqrt(np.sum((sample.c-expsample.c)**2)/sample.c.size) # RMSE
	# ~ return np.sum(np.abs(sample.c-expsample.c))/sample.c.size # MAE
	
# Optimization routine
guess=np.array([7.04998003e-03, 8.69586536e-03, 4.61830549e-03, 5.64390294e-02,
       1.29183677e-03, 1.00000009e+00, 3.18173366e-02, 1.30978963e-02,
       1.00000000e-05, 9.11079843e-04])
bounds=np.repeat(np.array([[1e-6,10]]),10,axis=0)
result=minimize(problem,guess,expsample,method='L-BFGS-B',bounds=bounds,options={'disp':True,'gtol':1e-04})
print(result)
