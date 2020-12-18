#!/usr/bin/env python
# coding: utf-8
from ase.visualize import view
from ase.io import read
from ase.io.trajectory import TrajectoryWriter,Trajectory
from irff.AtomDance import AtomDance


atoms = read('md.traj',index=-1)
ad = AtomDance(atoms)
images = ad.swing([3,0,2],st=90.0,ed=120.0,nbin=20,wtraj=True)
# images = ad.bend([7,3,4],rang=20.0,nbin=50,wtraj=True)
ad.close()
view(images)
