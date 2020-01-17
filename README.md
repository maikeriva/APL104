# APL104
A python implementation of the phase-field model outlined in Applied Physics Letters, vol. 104, no. 21, p. 213 907, May 2014, DOI: 10.1063/ 1.4879835

This code has been written for my MSc thesis on Energy Technologies at the Hong Kong University of Science and Technology, which can be found here: http://www.diva-portal.org/smash/record.jsf?pid=diva2:1320887

Some examples and calibration scripts can be found in the `scripts` directory

# 1D model

Import the required libraries

`
import numpy as np
import apl104lib as apl104
`

Create a new sample by passing the following input parameters:
- species: number of species
- size: the size of the interface in nm
- resolution: the resolution (for the spectral solver, that's the number of harmonics)

`
sample=apl104.Sample1D(species,size,resolution)
`

Set the valence of the species as well as the elements concentrations in the bulk materials of the interface

`
sample.z=np.array([1,2,3,-1,-2])
sample.c1_bulk=np.array([0,0,1,2,1])/4 # 0+0+1+2+1=4
sample.c2_bulk=np.array([3,2,0,12,7])/24 # 3+2+0+12+7=24
`

Set the phase equation parameters (read paper for thorough understanding)

`
sample.L=0.0006
sample.W0=50
sample.fc=0.1
`

Set the diffusion coefficients in the bulk materials

`
sample.D1=np.array(3.10695286e-02, 3.71141509e-02, 2.02426242e-02, 5.93923469e-02, 3.72455688e-03)
sample.D2=np.array(3.72455688e-03, 3.10695286e-02, 3.71141509e-02, 2.02426242e-02, 5.93923469e-02)
`

Set the initial condition of the interface for both concentration and phase. When using the spectral solver, it is recommended to use a symmetric condition.

`
sample.c[...]=sample.c1_bulk
sample.c[int(sample.res*1/4):int(sample.res*3/4)]=sample.c2_bulk
sample.p[...]=1
sample.p[int(sample.res*1/4):int(sample.res*3/4)]=0
`

Finally, update the sample. This is necessary to recompute the diffusion matrices with the new settings.

`
sample.update()
`

## Solving

I recommend to use the spectral solver due to its higher performances.
- sample: the sample previously set up
- dt: time step
- steps: number of steps
- log: record the interface status at every time step, and return it in two matrices for concentration and phase respectively. The matrices are indexed as f[step][concentration].

`
apl104.rfftsolve(sample,dt,steps,log=False)
`

There is also a basic euler-forward solver which I only used in the first stages of development. It has not been tested for long though and I don't expect it to work.

## Evaluation of diffused interface length

It is possible to evaluate the interface length of a processed sample with the following functions, both returning three vectors with values in nm: interface length, interface start coordinate, interface end coordinate. The tol parameter sets the sensitivity for the interface detection.

`
cifeval(sample,tol=1e-3)
pifeval(sample,tol=1e-3)
`

The following two functions perform the same task on interface evolution logs (read section "Solving") and similarly return three matrices with time-arranged information on interface length, interface start coordinate, interface end coordinate.

`
cifevallog(log,sample,tol=1e-3)
pifevallog(log,sample,tol=1e-3)
`

## Plotting

Some predefined plotting functions are available for convenience. They all accept similar parameters as:
- sample: the processed sample to plot
- interface: display vertical lines to mark the interface extension
- ifspecie: set the id if the specie to be used for interface detection
- iftol: tolerance for the interface extension detection
- save: save plot to a file
- filename: name of the file

Before working with plots, I suggest to define the names of the species to get pretty labels

`
sample.specienames=['La','Zr','Co','O','Li']
`

Each function call merely prepares the plot in memory. To display prepared plots, call:

`
apl104.showplots()
`

### Summary plot
`
apl104.summaryplot(sample,interface=True,ifspecie=0,iftol=1e-3,save=False,filename="summaryplot.pdf")
`
### Phase plot
`
apl104.pplot(sample,interface=True,iftol=1e-3,save=False,filename="pplot.pdf")
`
### Concentrations plot
`
apl104.cplot(sample,interface=True,ifspecie=0,iftol=1e-3,save=False,filename="cplot.pdf")
`
### Energy function plot
`
apl104.fplot(sample,interface=True,ifspecie=0,iftol=1e-3,save=False,filename="fplot.pdf")
`
### Energy function derivative plot
`
apl104.dfdpplot(sample,interface=True,ifspecie=0,iftol=1e-3,save=False,filename="dfdpplot.pdf")
`
# 2D model

The library has been designed to be eventually extended to multi-dimensional samples. While there is some code in place, further research is necessary to obtain a reasonably quick solver in this matter. I expose some ideas on how to achieve it in my MSc thesis mentioned at the beginning.