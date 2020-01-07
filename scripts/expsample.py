"""
 Experimental sample made with data from K. H. Kim et alii report
"""

import numpy as np
import pandas as pd
import apl104lib as apl104

# Settings
expsample=apl104.Sample1D(5,400,256)
expsample.specienames=['La','Zr','Co','O','Li']
padding=-4

# La
Ladata=pd.read_csv('Datasets/La',header=None)
Ladata=Ladata[(Ladata[1]<1020)&(Ladata[0]>0)&(Ladata[1]>0)] # Filtering
Ladata=Ladata.sort_values(0,axis=0) # Sorting
Ladata[0]=Ladata[0]+padding # Padding
Ladata[2]=Ladata[1]/Ladata[Ladata[0]>150][1].mean()*3/24 # Normalization
Ladata[2]=Ladata[2]-Ladata[2].min() # Flooring

# Zr
Zrdata=pd.read_csv('Datasets/Zr',header=None)
Zrdata=Zrdata[(Zrdata[1]<430)&(Zrdata[0]>0)&(Zrdata[1]>0)] # Filtering
Zrdata=Zrdata.sort_values(0,axis=0) # Sorting
Zrdata[0]=Zrdata[0]+padding # Padding
Zrdata[2]=Zrdata[1]/Zrdata[Zrdata[0]>150][1].mean()*2/24 # Normalization
Zrdata[2]=Zrdata[2]-Zrdata[2].min() # Flooring

# Co
Codata=pd.read_csv('Datasets/Co',header=None)
Codata=Codata[(Codata[1]<1100)&(Codata[0]>0)&(Codata[1]>0)] # Filtering
Codata=Codata.sort_values(0,axis=0) # Sorting
Codata[0]=Codata[0]+padding # Padding
Codata[2]=Codata[1]/Codata[Codata[0]<50][1].mean()*1/4 # Normalization
Codata[2]=Codata[2]-Codata[2].min() # Flooring

# O
Odata=pd.read_csv('Datasets/O',header=None)
Odata=Odata[(Odata[1]<390)&(Odata[0]>0)&(Odata[1]>0)] # Filtering
Odata=Odata.sort_values(0,axis=0) # Sorting
Odata[0]=Odata[0]+padding # Padding
Odata[2]=Odata[1]/Odata[1].mean()*2/4 # Normalization
# ~ Odata[2]=Odata[1]-Odata[1].min() # Flooring

# Define experimental sample
Lapoints=np.interp(np.linspace(0,int(expsample.dim/2),int(expsample.res/2)),Ladata[0],Ladata[2])
Zrpoints=np.interp(np.linspace(0,int(expsample.dim/2),int(expsample.res/2)),Zrdata[0],Zrdata[2])
Copoints=np.interp(np.linspace(0,int(expsample.dim/2),int(expsample.res/2)),Codata[0],Codata[2])
Opoints=np.interp(np.linspace(0,int(expsample.dim/2),int(expsample.res/2)),Odata[0],Odata[2])

expsample.c[:,0]=np.concatenate([Lapoints,np.flip(Lapoints,axis=0)]) #La
expsample.c[:,1]=np.concatenate([Zrpoints,np.flip(Zrpoints,axis=0)]) #Zr
expsample.c[:,2]=np.concatenate([Copoints,np.flip(Copoints,axis=0)]) #Co
expsample.c[:,3]=np.concatenate([Opoints,np.flip(Opoints,axis=0)])	 #O
expsample.c[:,4]=1-np.sum(expsample.c,axis=-1)						 #Li
expsample.c[:,4][expsample.c[:,4]<0]=0								 #Filter negative values
