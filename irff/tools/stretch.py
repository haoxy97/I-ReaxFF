#!/usr/bin/env python
# coding: utf-8
from ase.visualize import view
from ase.io import read
from ase.io.trajectory import TrajectoryWriter,Trajectory
from irff.AtomDance import AtomDance


atoms = read('poscar.gen')
ad = AtomDance(atoms=atoms)
# pairs = [[1,2],[13,7],[5,26]]
pairs = [[0,1]]
images = ad.stretch(pairs,nbin=30,st=0.7,ed=1.4,scale=1.35,wtraj=True)
ad.close()
view(images)



