#!/usr/bin/env python
# -*- coding: utf-8 -*-
from irff.dft.siesta import SinglePointEnergies,single_point
from ase.io import read
from ase.io.trajectory import TrajectoryWriter,Trajectory
from ase import Atoms
import numpy as np


images = Trajectory('md.traj')
tframe = len(images)
frame  = 50
E = []
ind   = [i for i in range(tframe)]


ind_ = []
energies = []
dEs = []
dE = 0.0
d2Es     = []
d2E      = 0.0

for i,atoms in enumerate(images):
    energy = atoms.get_potential_energy()
    if i>0 :
       if i<(tframe-1):
          deltEl =  energy - energies[-1]
          deltEr =  images[i+1].get_potential_energy() - energy
          dE     = abs(deltEl)
          d2E    = abs(deltEr-deltEl)
       else:
          deltEl =  energy - energies[-1]
          dE     = abs(deltEl)

    dEs.append(dE)
    d2Es.append(d2E)
    energies.append(energy)
    print('step:',i,'dE:',dE,'d2E:',d2E)

e_mean = np.mean(energies)
maxDiff = np.max(energies) - np.min(energies)

i = np.argmax(dEs)
i_= np.argmax(d2Es)
print(' * dEmax: ',i,dEs[i],' d2Emax: ',i_,d2Es[i_],'maxDiff:',maxDiff)
# print(' * dEmax: ',np.max(dEs),' d2Emax: ',np.max(d2Es))

# for e in energies:
#     print('energies:',e,'ave:',e_mean,'diff:',abs(e-e_mean))


