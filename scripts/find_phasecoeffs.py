"""
 Find phase coefficients
"""

import copy
import numpy as np
import matplotlib.pyplot as plt
import apl104lib as apl104
from scipy.optimize import minimize,minimize_scalar

exec(open('refsample.py').read())

# Define problem
def problem_scalar(guess):
	sample=copy.deepcopy(refsample)
	sample.L=guess
	steps=int(2*3600/0.1)
	clog,plog=apl104.rfftsolve(sample,0.1,steps,log=True)
	ciflength,cifstart,cifend=apl104.cifevallog(clog,sample,tol=1e-3)
	piflength,pifstart,pifend=apl104.pifevallog(plog,sample,tol=1e-3)
	print('Iteration complete')
	return np.sqrt(np.sum((np.amax(ciflength[600::],axis=-1)-piflength[600::])**2)/(steps-600))

# Optimization routine
bounds=np.array([1e-05,1e-3])
result=minimize_scalar(problem_scalar,method='bounded',bounds=bounds)
print(result)
