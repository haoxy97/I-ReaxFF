#!/usr/bin/env python
# coding: utf-8

from ase.optimize import BFGS,QuasiNewton
from ase.io import read,write
from irff.irff import IRFF

def geo_opt():
    # parameters
    atoms = read('nm.gen')
    atoms.calc = IRFF(atoms=atoms,nn=True,libfile='ffield.json')
    optimizer = BFGS(atoms,trajectory="opt.traj")
    optimizer.run(0.02,100)


if __name__ == '__main__':
   geo_opt()




