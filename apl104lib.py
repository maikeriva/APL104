"""
 apl104lib.py

 Python library implementation of:
 Interdiffusion across solid electrolyte-electrode interface
 Applied Physics Letters 104 (2014)

 Tested on:
 - python 3.6.5
 - numpy 1.14.2
 - matplotlib 2.2.2
"""

import numpy as np
import matplotlib.pyplot as plt
import time

"""
 1D model
"""
class Sample1D:
    def __init__(sample,species=1,dim=200,res=64):
        sample.species=species # Number of chemical species
        sample.dim=dim # Sample dimension (nm)
        sample.res=res # Harmonic resolution of the domain
        sample.z=np.zeros(species) # Ionic valence vector
        sample.c1_bulk=np.zeros(species) # Bulk concentrations vector (material 1)
        sample.c2_bulk=np.zeros(species) # Bulk concentrations vector (material 2)
        sample.D1=np.zeros(species) # Diffusion coefficients vector (material 1, nm²/s)
        sample.D2=np.zeros(species) # Diffusion coefficients vector (material 2, nm²/s)
        sample.L=0 # Phase evolution coefficient (nm³/(J*s))
        sample.W0=0 # Phase gradient coefficient (J/nm)
        sample.fc=0 # Free energy coefficient (J/(nm^3))
        sample.c=np.zeros((res,species)) # Concentrations domain
        sample.p=np.zeros(res) # Phase domain
        # Plotting parameters
        sample.name=' '
        sample.specienames=[' ' for specie in range(sample.species)]
    
    def h(sample):
        return sample.p**3*(6*sample.p**2-15*sample.p+10)
    
    def D(sample):
        return sample.h()[...,np.newaxis,np.newaxis]*sample.D1mat+(1-sample.h())[...,np.newaxis,np.newaxis]*sample.D2mat
    
    def f(sample):
        return  np.sum((sample.c[...,0:-1]-sample.c1_bulk[0:-1])**2,axis=-1)*sample.h()+\
        np.sum((sample.c[...,0:-1]-sample.c2_bulk[0:-1])**2,axis=-1)*(1-sample.h())+\
        2*(sample.p**4-2*sample.p**3+sample.p**2)
    
    def dfdp(sample):
        return (30*sample.p**4-60*sample.p**3+30*sample.p**2)*\
        (np.sum((sample.c[...,0:-1]-sample.c1_bulk[0:-1])**2,axis=-1)-\
        np.sum((sample.c[...,0:-1]-sample.c2_bulk[0:-1])**2,axis=-1))+\
        (8*sample.p**3-12*sample.p**2+4*sample.p)
    
    def update(sample):
        # Update diffusion matrices. Call after editing the sample.
        sample.D1mat=(sample.res/sample.dim)**2*(np.diag(sample.D1[0:-1])-(sample.z[0:-1]*sample.c1_bulk[0:-1]*sample.D1[0:-1])[:,np.newaxis]*\
            (sample.z[0:-1]*(sample.D1[0:-1]-sample.D1[-1]))[np.newaxis,:]/np.sum(sample.z**2*sample.c1_bulk*sample.D1))
        sample.D2mat=(sample.res/sample.dim)**2*(np.diag(sample.D2[0:-1])-(sample.z[0:-1]*sample.c2_bulk[0:-1]*sample.D2[0:-1])[:,np.newaxis]*\
            (sample.z[0:-1]*(sample.D2[0:-1]-sample.D2[-1]))[np.newaxis,:]/np.sum(sample.z**2*sample.c2_bulk*sample.D2))

###
# Finite-difference explicit Euler FW solver
###
def efwsolve(sample,dt,steps):
    # Prepare scaled coefficients
    sample.L_s=sample.L*(sample.res/sample.dim)**3 # nm³/(J*s)
    sample.W0_s=sample.W0*(sample.dim/sample.res) # J/(nm)
    sample.fc_s=sample.fc*(sample.dim/sample.res)**3 # J/(nm³)
    # Update sample internal data
    sample.update()
    # Performance improvement
    matmuldest=np.ndarray((sample.res,sample.species-1,1))
    # Euler forward explicit algorithm
    for step in range(steps):
        sample.c[...,0:-1]=sample.c[...,0:-1]+dt*np.gradient(np.matmul(sample.D(),np.gradient(sample.c[...,0:-1],axis=0)[...,np.newaxis],matmuldest)[...,0],axis=0)
        sample.p=sample.p+dt*sample.L_s*(sample.W0_s*np.gradient(np.gradient(sample.p))-sample.fc_s*sample.dfdp())
    # Compute last species once cycle has finished
    sample.c[...,-1]=1-np.sum(sample.c[...,0:-1],axis=-1)

###
# Spectral semi-implicit Fourier solver with real numbers optimization
###
def rfftsolve(sample,dt,steps,log=False):
    # Prepare scaled coefficients
    sample.L_s=sample.L*(sample.res/sample.dim)**3 # nm³/(J*s)
    sample.W0_s=sample.W0*(sample.dim/sample.res) # J/(nm)
    sample.fc_s=sample.fc*(sample.dim/sample.res)**3 # J/(nm³)
    # Update sample internal data
    sample.update()
    # Performance improvement
    matmuldest=np.ndarray((sample.res,sample.species-1,1))
    # Prepare fourier coefficients for real FFT
    kx=np.linspace(0,sample.res/2,sample.res/2+1)*2*np.pi/sample.res
    k=np.sqrt(kx**2)
    if log:
        # Prepare arrays for storage
        clog=np.zeros((steps,sample.res,sample.species))
        plog=np.zeros((steps,sample.res))
        # Prepare first FFT
        c_fft=np.fft.rfft(sample.c[...,0:-1],axis=0)
        p_fft=np.fft.rfft(sample.p)
        # Solving cycle
        for step in range(steps):
            # Phase evolution equation in fourier space
            p_fft=(p_fft-dt*sample.L_s*sample.fc_s*np.fft.rfft(sample.dfdp()))/(1+dt*sample.L_s*sample.W0_s*k**2)
            sample.p=np.fft.irfft(p_fft)
            # Concentration evolution equation in fourier space
            c_fft=c_fft+dt*1j*k[...,np.newaxis]*np.fft.rfft(np.matmul(sample.D(),np.fft.irfft(1j*k[...,np.newaxis]*c_fft,axis=0)[...,np.newaxis],matmuldest)[...,0],axis=0)
            sample.c[...,0:-1]=np.fft.irfft(c_fft,axis=0)
            # Compute last specie
            sample.c[...,-1]=1-np.sum(sample.c[...,0:-1],axis=-1)
            # Store logs
            clog[step]=sample.c
            plog[step]=sample.p
        # Return logs
        return clog,plog
    else:
        # Prepare first FFT
        c_fft=np.fft.rfft(sample.c[...,0:-1],axis=0)
        p_fft=np.fft.rfft(sample.p)
        # Solving cycle
        for step in range(steps):
            # Phase evolution equation in fourier space
            p_fft=(p_fft-dt*sample.L_s*sample.fc_s*np.fft.rfft(sample.dfdp()))/(1+dt*sample.L_s*sample.W0_s*k**2)
            sample.p=np.fft.irfft(p_fft)
            # Concentration evolution equation in fourier space
            c_fft=c_fft+dt*1j*k[...,np.newaxis]*np.fft.rfft(np.matmul(sample.D(),np.fft.irfft(1j*k[...,np.newaxis]*c_fft,axis=0)[...,np.newaxis],matmuldest)[...,0],axis=0)
            sample.c[...,0:-1]=np.fft.irfft(c_fft,axis=0)
        # Compute last species once cycle has finished
        sample.c[...,-1]=1-np.sum(sample.c[...,0:-1],axis=-1)

"""
 Interface evaluation
"""
def cifeval(sample,tol=1e-3):
	ifstart=np.argmin(np.abs(sample.c[0:int(sample.res/2)]-sample.c1_bulk)<tol,axis=0)*sample.dim/sample.res
	ifend=(sample.res/2-np.argmin(np.abs(np.flip(sample.c[0:int(sample.res/2)],axis=0)-sample.c2_bulk)<tol,axis=0))*sample.dim/sample.res
	return ifend-ifstart,ifstart,ifend

def pifeval(sample,tol=1e-3):
	ifstart=np.argmin(np.abs(sample.p[0:int(sample.res/2)]-1)<tol,axis=0)*sample.dim/sample.res
	ifend=(sample.res/2-np.argmin(np.abs(np.flip(sample.p[0:int(sample.res/2)],axis=0)-0)<tol,axis=0))*sample.dim/sample.res
	return ifend-ifstart,ifstart,ifend

def cifevallog(log,sample,tol=1e-3):
	ifstart=np.argmin(np.abs(log[:,0:int(sample.res/2),:]-sample.c1_bulk)<tol,axis=1)*sample.dim/sample.res
	ifend=(sample.res/2-np.argmin(np.abs(np.flip(log[:,0:int(sample.res/2),:],axis=1)-sample.c2_bulk)<tol,axis=1))*sample.dim/sample.res
	return ifend-ifstart,ifstart,ifend

def pifevallog(log,sample,tol=1e-3):
	ifstart=np.argmin(np.abs(log[:,0:int(sample.res/2)]-1)<tol,axis=1)*sample.dim/sample.res
	ifend=(sample.res/2-np.argmin(np.abs(np.flip(log[:,0:int(sample.res/2)],axis=1)-0)<tol,axis=1))*sample.dim/sample.res
	return ifend-ifstart,ifstart,ifend

###
# Plotting functions
###
def summaryplot(sample,interface=True,ifspecie=0,iftol=1e-3,save=False,filename="summaryplot.pdf"):
    fig,ax1=plt.subplots()
    ax1.plot(np.linspace(0,sample.dim,sample.res),sample.p,linestyle='dashed')
    ax1.set_ylim([-0.1,1.1])
    ax1.set_title("Results overview")
    ax1.set_xlabel("Coordinate (nm)")
    ax1.set_ylabel("Phase (dashed line)")
    ax1.grid()
    ax2=ax1.twinx()
    ax2.plot(np.linspace(0,sample.dim,sample.res),sample.c)
    ax2.set_ylim([-0.1,1.1])
    ax2.set_ylabel("Molar concentration (solid line)")
    ax2.legend(sample.specienames)
    if interface:
        iflength,ifstart,ifend=cifeval(sample,iftol)
        ax2.vlines([ifstart[ifspecie],ifend[ifspecie],sample.dim-ifend[ifspecie],sample.dim-ifstart[ifspecie]],0,1,linestyle='dotted')
    if save:
        fig.savefig(filename)
    return fig

def pplot(sample,interface=True,iftol=1e-3,save=False,filename="pplot.pdf"):
    fig=plt.figure()
    plt.plot(np.linspace(0,sample.dim,sample.res),sample.p)
    plt.ylim([-0.1,1.1])
    plt.grid()
    plt.title("Phase profile")
    plt.xlabel("Coordinate (nm)")
    if interface:
        iflength,ifstart,ifend=pifeval(sample,iftol)
        plt.vlines([ifstart,ifend,ifstart+sample.dim/2,ifend+sample.dim/2],0,1,linestyle='dotted')
    if save:
        fig.savefig(filename)
    return fig

def cplot(sample,interface=True,ifspecie=0,iftol=1e-3,save=False,filename="cplot.pdf"):
    fig=plt.figure()
    plt.plot(np.linspace(0,sample.dim,sample.res),sample.c)
    plt.ylim([-0.1,1.1])
    plt.grid()
    plt.title("Concentration profiles")
    plt.xlabel("Coordinate (nm)")
    plt.ylabel("Molar concentration")
    plt.legend(sample.specienames)
    if interface:
        iflength,ifstart,ifend=cifeval(sample,ifspecie,iftol)
        plt.vlines([ifstart[...,ifspecie],ifend[...,ifspecie],ifstart[...,ifspecie]+sample.dim/2,ifend[...,ifspecie]+sample.dim/2],0,1,linestyle='dotted')
    if save:
        fig.savefig(filename)
    return fig

def fplot(sample,interface=True,ifspecie=0,iftol=1e-3,save=False,filename="fplot.pdf"):
    fig=plt.figure()
    plt.plot(np.linspace(0,sample.dim,sample.res),sample.f())
    plt.grid()
    plt.title("f plot")
    plt.xlabel("Coordinate (nm)")
    plt.ylabel("Energy (J/nm^3)")
    if interface:
        iflength,ifstart,ifend=cifeval(sample,ifspecie,iftol)
        plt.vlines([ifstart[...,ifspecie],ifend[...,ifspecie],ifstart[...,ifspecie]+sample.dim/2,ifend[...,ifspecie]+sample.dim/2],0,1,linestyle='dotted')
    if save:
        fig.savefig(filename)
    return fig

def dfdpplot(sample,interface=True,ifspecie=0,iftol=1e-3,save=False,filename="dfdpplot.pdf"):
    fig=plt.figure()
    plt.plot(np.linspace(0,sample.dim,sample.res),sample.dfdp())
    plt.grid()
    plt.title("dfdp plot")
    plt.xlabel("Coordinate (nm)")
    plt.ylabel("Energy variation (J/nm^3)")
    if interface:
        iflength,ifstart,ifend=cifeval(sample,ifspecie,iftol)
        plt.vlines([ifstart[...,ifspecie],ifend[...,ifspecie],ifstart[...,ifspecie]+sample.dim/2,ifend[...,ifspecie]+sample.dim/2],0,1,linestyle='dotted')
    if save:
        fig.savefig(filename)
    return fig

def showplots():
    plt.show()

"""
2D model
"""
class Sample2D:
    def __init__(sample,species=1,dim=200,res=64):
        sample.species=species
        sample.dim=dim
        sample.res=res
        sample.z=np.zeros(species)
        sample.c1_bulk=np.zeros(species)
        sample.c2_bulk=np.zeros(species)
        sample.D1=np.zeros(species) # nm^2/s
        sample.D2=np.zeros(species) # nm^2/s
        sample.L=0 # nm^3/(J*s)
        sample.W0=0 # J/nm
        sample.fc=0 # J/(nm^3)
        sample.c=np.zeros([res,res,species])
        sample.p=np.zeros([res,res])
        # Plotting parameters
        sample.name=' '
        sample.specienames=[' ' for specie in range(sample.species)]
    
    def h(sample):
        return sample.p**3*(6*sample.p**2-15*sample.p+10)
    
    def D(sample):
        return sample.h()[...,np.newaxis,np.newaxis]*sample.D1mat+(1-sample.h())[...,np.newaxis,np.newaxis]*sample.D2mat
    
    def f(sample):
        return  np.sum((sample.c[...,0:-1]-sample.c1_bulk[0:-1])**2,axis=-1)*sample.h()+\
        np.sum((sample.c[...,0:-1]-sample.c2_bulk[0:-1])**2,axis=-1)*(1-sample.h())+\
        2*(sample.p**4-2*sample.p**3+sample.p**2)

    def dfdp(sample):
        return (30*sample.p**4-60*sample.p**3+30*sample.p**2)*\
        (np.sum((sample.c[...,0:-1]-sample.c1_bulk[0:-1])**2,axis=-1)-\
        np.sum((sample.c[...,0:-1]-sample.c2_bulk[0:-1])**2,axis=-1))+\
        (8*sample.p**3-12*sample.p**2+4*sample.p)
    
    def update(sample):
        sample.D1mat=(sample.res/sample.dim)**2*(np.diag(sample.D1[0:-1])-(sample.z[0:-1]*sample.c1_bulk[0:-1]*sample.D1[0:-1])[:,np.newaxis]*\
            (sample.z[0:-1]*(sample.D1[0:-1]-sample.D1[-1]))[np.newaxis,:]/np.sum(sample.z**2*sample.c1_bulk*sample.D1))
        sample.D2mat=(sample.res/sample.dim)**2*(np.diag(sample.D2[0:-1])-(sample.z[0:-1]*sample.c2_bulk[0:-1]*sample.D2[0:-1])[:,np.newaxis]*\
            (sample.z[0:-1]*(sample.D2[0:-1]-sample.D2[-1]))[np.newaxis,:]/np.sum(sample.z**2*sample.c2_bulk*sample.D2))

###
# Spectral semi-implicit Fourier solver without real numbers optimization
###
def fftsolve2D(sample,dt,steps,log=False):
    # Prepare scaled coefficients
    sample.L_s=sample.L*(sample.res/sample.dim)**3 # nm³/(J*s)
    sample.W0_s=sample.W0*(sample.dim/sample.res) # J/(nm)
    sample.fc_s=sample.fc*(sample.dim/sample.res)**3 # J/(nm³)
    # Update sample internal data
    sample.update()
    # Performance improvement
    matmuldest=np.ndarray((sample.res,sample.res,sample.species-1,1),dtype=complex)
    # Prepare fourier coefficients for real FFT
    kx=np.concatenate([np.linspace(0,sample.res/2,sample.res/2,False),np.linspace(-sample.res/2,0,sample.res/2,False)])*2*np.pi/sample.res
    ky=np.concatenate([np.linspace(0,sample.res/2,sample.res/2,False),np.linspace(-sample.res/2,0,sample.res/2,False)])*2*np.pi/sample.res
    k=np.sqrt(kx[np.newaxis,:]**2+ky[:,np.newaxis]**2)
    if log:
        # Prepare arrays for storage
        clog=np.zeros((steps,sample.res,sample.res,sample.species))
        plog=np.zeros((steps,sample.res,sample.res))
        # Prepare first FFT
        c_fft=np.fft.fftn(sample.c[...,0:-1],axes=(0,1))
        p_fft=np.fft.fftn(sample.p)
        # Solving cycle
        for step in range(steps):
            # Phase evolution equation in fourier space
            p_fft=(p_fft-dt*sample.L_s*sample.fc_s*np.fft.fftn(sample.dfdp()))/(1+dt*sample.L_s*sample.W0_s*k**2)
            sample.p=np.real(np.fft.ifftn(p_fft))
            # Concentration evolution equation in fourier space
            c_fft=c_fft+dt*1j*k[...,np.newaxis]*np.fft.fftn(np.matmul(sample.D(),np.fft.ifftn(1j*k[...,np.newaxis]*c_fft,axes=(0,1))[...,np.newaxis],matmuldest)[...,0],axes=(0,1))
            sample.c[...,0:-1]=np.real(np.fft.ifftn(c_fft,axes=(0,-1)))
            # Compute last specie
            sample.c[...,-1]=1-np.sum(sample.c[...,0:-1],axis=-1)
            # Store logs
            clog[step]=sample.c
            plog[step]=sample.p
        # Return logs
        return clog,plog
    else:
        # Prepare first FFT
        c_fft=np.fft.fftn(sample.c[...,0:-1],axes=(0,1))
        p_fft=np.fft.fftn(sample.p)
        # Solving cycle
        for step in range(steps):
            # Phase evolution equation in fourier space
            p_fft=(p_fft-dt*sample.L_s*sample.fc_s*np.fft.fftn(sample.dfdp()))/(1+dt*sample.L_s*sample.W0_s*k**2)
            sample.p=np.real(np.fft.ifftn(p_fft))
            # Concentration evolution equation in fourier space
            c_fft=c_fft+dt*1j*k[...,np.newaxis]*np.fft.fftn(np.matmul(sample.D(),np.fft.ifftn(1j*k[...,np.newaxis]*c_fft,axes=(0,1))[...,np.newaxis],matmuldest)[...,0],axes=(0,1))
            sample.c[...,0:-1]=np.real(np.fft.ifftn(c_fft,axes=(0,1)))
        # Compute last species once cycle has finished
        sample.c[...,-1]=1-np.sum(sample.c[...,0:-1],axis=-1)

"""
 2D contour plot functions
"""
def contourcplot2d(sample,specie=0,save=False,filename="contourcplot2d.pdf"):
    speciename=sample.specienames(specie)
    plt.figure()
    plt.contourf(np.linspace(0,sample.dim,sample.res),np.linspace(0,sample.dim,sample.res),sample.c[...,specie])
    plt.colorbar()
    plt.title("Molar concentration of {}".format(speciename))
    plt.xlabel("Coordinate (nm)")
    plt.ylabel("Coordinate (nm)")
    if save:
        fig.savefig(filename)

def contourpplot2d(sample,save=False,filename="contourpplot.pdf"):
    pfig=plt.figure()
    pplot=plt.contourf(np.linspace(0,sample.dim,sample.res),np.linspace(0,sample.dim,sample.res),sample.p)
    plt.colorbar(pplot)
    plt.title("Phase")
    plt.xlabel("Coordinate (nm)")
    plt.ylabel("Coordinate (nm)")
    if save:
        fig.savefig(filename)

