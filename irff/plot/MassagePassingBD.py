#!/usr/bin/env python
# coding: utf-8

# In[5]:


import numpy as np
from ase.optimize import BFGS
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.verlet import VelocityVerlet
from ase.io.trajectory import Trajectory,TrajectoryWriter
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import read,write
from ase import units
from ase.visualize import view
from irff.irff_np import IRFF_NP
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from irff.AtomOP import AtomOP


# In[7]:


atoms=Trajectory('C2H4.traj')
view(atoms)


# In[2]:


def bde(ffield='ffield.json',nn='T',gen='poscar.gen',traj='C2H4.traj'):
    images = Trajectory(traj)
    atoms = images[0]
    nn_=True if nn=='T'  else False
    
    ir = IRFF_NP(atoms=atoms,
                 libfile=ffield,
                 rcut=None,
                 nn=nn_)
    natom = ir.natom
    
    Empnn,Esiesta = [],[]
    eb,eb_ = [],[]
    for atoms in images:
        ir.calculate(atoms)
        Empnn.append(ir.E)
        Esiesta.append(atoms.get_potential_energy())
        # eb.append(ir.ebond[i][j])
        
    fig, ax = plt.subplots() 

    plt.plot(Empnn[0:14],label=r'$E_{MPNN}$', color='blue', linewidth=2, linestyle='-.')
    plt.plot(Esiesta[0:14],label=r'$E_{SIESTA}$', color='r', linewidth=2, linestyle='-')
    plt.legend(loc='best',edgecolor='yellowgreen')

    plt.savefig('Ecompare.pdf') 
    plt.show()
    plt.close()

bde(traj='C2H4.traj')


# In[8]:


atom1 = read('c2h6.gen')
view(atom1)


# In[7]:


ao = AtomOP(atom1)
# pairs = [[1,2],[13,7],[5,26]]
pairs = [[1,3]]
images = ao.stretch(pairs,nbin=50,wtraj=True)
ao.close()
view(images)


# In[10]:


atom2 = read('c2h6.gen')
ao2 = AtomOP(atom2)
# pairs = [[1,2],[13,7],[5,26]]
pairs = [[0,1]]
images2 = ao2.stretch(pairs,nbin=50,wtraj=True)
ao2.close()
view(images2)


# In[14]:


e_c2h4,e_c2h6 = [],[]
for atoms in images:
    e_c2h4.append(atoms.get_potential_energy())
for atoms in images2:
    e_c2h6.append(atoms.get_potential_energy())

fig, ax = plt.subplots() 
plt.plot(e_c2h4[0:40]- min(e_c2h4),label=r'$C_{2}H_4$', color='blue', linewidth=2, linestyle='-.')
plt.plot(e_c2h6[0:40]- min(e_c2h6),label=r'$C_{2}H_6$', color='r', linewidth=2, linestyle='-')
plt.legend(loc='best',edgecolor='yellowgreen')

plt.savefig('Estretch.pdf') 
plt.show()
plt.close()

