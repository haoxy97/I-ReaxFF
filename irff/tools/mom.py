#!/usr/bin/env python
# coding: utf-8
from ase.visualize import view
from ase.io import read
from ase.io.trajectory import TrajectoryWriter,Trajectory
from irff.irff import IRFF
from irff.AtomDance import AtomDance
from irff.md.irmd import IRMD


atoms = read('f4.gen',index=-1)
ad = AtomDance(atoms=atoms)
# pairs = [[1,2],[13,7],[5,26]]
# images = ad.stretch([3,2],nbin=30,st=0.85,ed=1.25,scale=1.2,traj='md.traj')
atoms = ad.set_bond_momenta(1,0,atoms,sign=1.0)
ad.close()

irmd  = IRMD(atoms=atoms,time_step=0.1,totstep=2,Tmax=10000,
             ro=0.8,rtole=0.5,initT=700)
irmd.run()
mdsteps= irmd.step
Emd  = irmd.Epot
irmd.close()


