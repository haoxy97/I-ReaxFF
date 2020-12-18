#!/usr/bin/env python
# coding: utf-8
from os import system
from ase.optimize import BFGS,QuasiNewton
from ase.constraints import StrainFilter
from ase.vibrations import Vibrations
from ase.io import read,write
from irff.irff import IRFF
from irff.AtomDance import AtomDance
from irff.md.irmd import IRMD
from ase.visualize import view
from ase.io.trajectory import Trajectory,TrajectoryWriter
import numpy as np
import argh
import argparse


def opt(atoms=None,gen='poscar.gen',fmax=0.003,step=50):
    if atoms is None:
       atoms = read(gen)
    atoms.calc = IRFF(atoms=atoms,libfile='ffield.json',rcut=None,nn=True,massage=1)

    def check(atoms=atoms):
        epot_      = atoms.get_potential_energy()
        r          = atoms.calc.r.detach().numpy()
        i_         = np.where(np.logical_and(r<0.5,r>0.0001))
        n          = len(i_[0])

        try:
           assert not np.isnan(epot_), '-  Energy is NaN!'
        except:
           atoms.write('poscarN.gen')
           raise ValueError('-  Energy is NaN!' )

    optimizer = BFGS(atoms,trajectory="opt.traj")
    optimizer.attach(check,interval=1)
    optimizer.run(fmax,step)
    # images = Trajectory('opt.traj')
    # return images


def optl(atoms=None,gen='poscar.gen',fmax=0.003,step=50):
    if atoms is None:
       atoms = read(gen)
    atoms.calc = IRFF(atoms=atoms,libfile='ffield.json',rcut=None,
                      nn=True,CalStress=True)

    sf  =  StrainFilter(atoms)

    optimizer = BFGS(sf,trajectory="opt.traj")
    optimizer.run(fmax,step)

    # images = Trajectory('opt.traj')
    # return images


def freq():
    atoms = read('md.traj',index=0)
    atoms.calc = IRFF(atoms=atoms,libfile='ffield.json',rcut=None,nn=True,massage=2)
    # Compute frequencies
    frequencies = Vibrations(atoms, name='freq')
    frequencies.run()
    # Print a summary
    frequencies.summary()
    frequencies.write_dos()

    # Write jmol file if requested
    # if write_jmol:
    frequencies.write_jmol()


def md(atoms=None,gen='poscar.gen',step=100,model='mpnn',T=700):
    # opt(gen=gen)
    atoms = read(gen)
    ad    = AtomDance(atoms,bondTole=1.35)
    atoms = ad.bond_momenta_bigest(atoms)
    # atoms = ao.set_bond_momenta(0,5,atoms,-1.0)
    irmd  = IRMD(atoms=atoms,time_step=0.1,totstep=step,gen=gen,Tmax=10000,
                 ro=0.8,rtole=0.5,initT=T)
    irmd.run()
    mdsteps= irmd.step
    Emd  = irmd.Epot
    irmd.close()
    ad.close()
    # images = Trajectory('md.traj')
    # view(images) # ,viewer='x3d'


def mom(atoms=None,i=0,j=3,gen='poscar.gen',step=100,T=800,vdwnn=False):
    atoms = read(gen,index=-1)
    ad    = AtomDance(atoms,bondTole=1.35)
    atoms = ad.set_bond_momenta(i,j,atoms,sign=1.0)
    irmd  = IRMD(atoms=atoms,time_step=0.1,totstep=step,gen=gen,Tmax=10000,
                 ro=0.8,rtole=0.5,initT=T)
    irmd.run()
    mdsteps= irmd.step
    Emd  = irmd.Epot
    irmd.close()
    ad.close()
        


if __name__ == '__main__':
   ''' use commond like ./mpnn.py <opt> to run it
       use --h to see options
   '''
   parser = argparse.ArgumentParser()
   argh.add_commands(parser, [opt,optl,md,freq,mom])
   argh.dispatch(parser)
