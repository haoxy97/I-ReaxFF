#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argh
import argparse
from os import environ,system,getcwd
from os.path import exists
# from .mdtodata import MDtoData
from ase.io import read,write
from ase.io.trajectory import Trajectory
from ase import Atoms
import matplotlib.pyplot as plt
import numpy as np
from irff.irff import IRFF



def e(traj='eos_dft.traj',batch_size=100):
    images = Trajectory(traj)
    e,ei,ei_,diff = [],[],[],[]

    ir_mpnn = IRFF(atoms=images[0],
                 libfile='ffield.json',
                 nn=True)
    ir_mpnn.get_potential_energy(images[0])

    ir_reax = IRFF(atoms=images[0],
                 libfile='ffield',
                 nn=False)
    ir_reax.get_potential_energy(images[0])
    
    v = []
    for i,atoms in enumerate(images):
        e.append(atoms.get_potential_energy())
        ei.append(ir_mpnn.get_potential_energy(atoms))
        ei_.append(ir_reax.get_potential_energy(atoms))
        diff.append(abs(e[-1]-ei[-1]))
        print(' * energy: ',e[-1],ei[-1],ei_[-1],diff[-1])
        v.append(atoms.get_volume())
        # stress = atoms.get_stress()
        # print(stress)

    print(' * mean difference: ',np.mean(diff))
    e_min = min(e)
    e_max = max(e)
    e = np.array(e) - e_min
    e_min = min(ei)
    ei = np.array(ei) - e_min

    e_min = min(ei_)
    ei_ = np.array(ei_) - e_min

    plt.figure()   
    plt.ylabel(r'$Energy$ ($eV$)')
    plt.xlabel(r'$Volume$ ($\AA^3$)')
    # plt.xlim(0,i)
    # plt.ylim(0,np.max(hist)+0.01)

    plt.plot(v,ei,alpha=0.9,
             linestyle='-',marker='o',markerfacecolor='k',markersize=5,
             color='k',label='IRFF(MPNN)')

    # plt.plot(v,ei,alpha=0.9,
    #          linestyle='-',marker='^',markerfacecolor='none',
    #          markeredgewidth=1,markeredgecolor='blue',markersize=5,
    #          color='blue',label='IRFF')

    err = np.abs(ei - e)
    err_= np.abs(ei_ - e)

    plt.errorbar(v, e, yerr=err,
                 fmt='s', ecolor='r', color='r', ms=6, markerfacecolor='none', mec='r',
                 elinewidth=2, capsize=2, label='DFT (SIESTA)')

    plt.plot(v,ei_,alpha=0.9,
             linestyle='-',marker='^',markerfacecolor='b',markersize=5,
             color='b',label='ReaxFF(trained)')

    plt.fill_between(v, e - err, e + err, color='darkorange',
                     alpha=0.2)
    
    plt.fill_between(v, e - err_, e + err_, color='palegreen', # palegreen
                     alpha=0.2)

    plt.text( 0.0, e_max, '%.3f' %e_min, fontdict={'size':10.5, 'color': 'k'})
    plt.legend(loc='best',edgecolor='yellowgreen') # lower left upper right
    plt.savefig('Energy.svg',transparent=True) 
    plt.close() 


if __name__ == '__main__':
   ''' use commond like ./cp.py scale-md --T=2800 to run it'''
   parser = argparse.ArgumentParser()
   argh.add_commands(parser, [e])
   argh.dispatch(parser)
