from .siesta import single_point
from ase.io import read,write
from ase.io.trajectory import TrajectoryWriter,Trajectory
from ase.calculators.singlepoint import SinglePointCalculator
from ase.units import Ry
from ase import Atoms
import numpy as np


def SinglePointEnergies(traj,label='aimd',xcf='VDW',xca='DRSLL',basistype='DZP',
                        EngTole=0.0000001,frame=50,cpu=4,
                        dE=0.2,d2E=0.1,select=False):
    ''' get single point energy and labeling data '''
    images = Trajectory(traj)
    tframe = len(images)
    E,E_,dEs = [],[],[]
    if tframe>frame:
       if frame>1:
          ind_  = list(np.linspace(0,tframe-1,num=frame,dtype=np.int32))
       else:
       	  ind_  = [tframe-1]
    else:
       ind_  = [i for i in range(tframe)]   

    if len(ind_)>1 and 0 in ind_:
       ind_.pop(0)

    his = TrajectoryWriter(label+'.traj',mode='w')
    # print(frame,ind_)
    energies = []
    d2Es     = []
    dE_      = 0.0
    d2E_     = 0.0

    for i,atoms in enumerate(images):
        energy = atoms.get_potential_energy()
        if i>0: 
           if i<(tframe-1):
              deltEl = energy - energies[-1]
              deltEr = images[i+1].get_potential_energy() - energy
              dE_ = abs(deltEl)
              d2E_= abs(deltEr-deltEl)
           else:
              deltEl =  energy - energies[-1]
              dE_ = abs(deltEl)

        energies.append(energy)
        dEs.append(dE_)
        d2Es.append(d2E_)
        if select:
           if dE_>dE or d2E_>d2E:
              if i not in ind_:
                 ind_.append(i)

    ide  = np.argmax(dEs)
    id2e = np.argmax(d2Es)
    if (ide not in ind_) and (ide+1 not in ind_) and (ide-1 not in ind_): 
       ind_.append(ide)
    if id2e not in ind_ and (id2e+1 not in ind_) and (id2e-1 not in ind_): 
       ind_.append(id2e)
    ind_.sort()

    for i in ind_:
        atoms = images[i]
        e_    = atoms.get_potential_energy()
        dE_   = dEs[i]
        d2E_  = d2Es[i]

        atoms_= single_point(atoms,xcf=xcf,xca=xca,basistype=basistype,cpu=cpu)
        e     = atoms_.get_potential_energy()
        E.append(e)
        E_.append(e_)

        diff_ = abs(e-e_)

        print(' * %d Energies from MLP: %9.5f DFT: %9.5f Diff: %6.6f dE: %5.4f d2E: %5.4f' %(i,
                      e_,e,diff_,dE_,d2E_))
        with open('SinglePointEnergies.log','a') as fs:  
             fs.write('%d MLP: %9.5f DFT: %9.5f Diff: %6.6f dE: %5.4f d2E: %5.4f\n' %(i,
                      e_,e,diff_,dE_,d2E_))
            
        if diff_>EngTole:          # or i==ind_[-1]
           his.write(atoms=atoms_)

    his.close()
    images = None
    dEmax  = dEs[ide]
    d2Emax = d2Es[id2e]
    return E,E_,dEmax,d2Emax

