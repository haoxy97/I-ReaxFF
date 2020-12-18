#!/usr/bin/env python
# coding: utf-8
import networkx as nx
from ase.io import read,write
from irff.reaxfflib import read_lib
from irff.irff_np import IRFF_NP
from irff.structures import structure
from ase.visualize import view
import matplotlib.pyplot as plt
import numpy as np
import json as js


colors = ['darkviolet','darkcyan','fuchsia','chartreuse',
          'midnightblue','red','deeppink','blue',
          'cornflowerblue','orangered','lime','magenta',
          'mediumturquoise','aqua','cyan','deepskyblue',
          'firebrick','mediumslateblue','khaki','gold','k']


def init_bonds(p_):
    spec,bonds,offd,angs,torp,hbs = [],[],[],[],[],[]
    for key in p_:
        # key = key.encode('raw_unicode_escape')
        # print(key)
        k = key.split('_')
        if k[0]=='bo1':
           bonds.append(k[1])
        elif k[0]=='rosi':
           kk = k[1].split('-')
           # print(kk)
           if len(kk)==2:
              offd.append(k[1])
           elif len(kk)==1:
              spec.append(k[1])
        elif k[0]=='theta0':
           angs.append(k[1])
        elif k[0]=='tor1':
           torp.append(k[1])
        elif k[0]=='rohb':
           hbs.append(k[1])
    return spec,bonds,offd,angs,torp,hbs


def get_p(ffield):
    if ffield.endswith('.json'):
       lf = open(ffield,'r')
       j = js.load(lf)
       p  = j['p']
       m       = j['m']
       spec,bonds,offd,angs,torp,hbs= init_bonds(p)
    else:
       p,zpe_,spec,bonds,offd,Angs,torp,Hbs=read_lib(libfile=ffield,zpe=False)
    return p,bonds


def get_bo(r,rosi=1.3935,bo1=-0.075,bo2=5.0):
    bo   = np.exp(bo1*(r/rosi)**bo2)
    return bo


def plbo(ffield='ffield.json'):
    p,bonds = get_p(ffield)
    r = np.arange(0.0001,2.6,0.02)

    for bt in ['si','pi','pp']: 
        plt.figure()
        plt.ylabel( 'Uncorrected Bond Order (%s)' %bt)
        plt.xlabel(r'$Radius$ $(Angstrom)$')
        plt.xlim(0,2.5)
        plt.ylim(0,1.01)
        for i,bd in enumerate(bonds):
            b = bd.split('-')
            bdn = b[0] if b[0]==b[1] else bd 
            
            if bt=='si':
               # print(bd,'rosi:',p['rosi_'+bdn],'bo1:',p['bo1_'+bd],'bo2:',p['bo2_'+bd])
               bosi=get_bo(r,rosi=p['rosi_'+bdn],bo1=p['bo1_'+bd],bo2=p['bo2_'+bd])
               plt.plot(r,bosi,label=r'$%s$' %bd, 
                        color=colors[i%len(colors)], linewidth=2, linestyle='-')
            if bt=='pi':
               # print(bd,'ropi:',p['ropi_'+bdn],'bo3:',p['bo3_'+bd],'bo4',p['bo4_'+bd])
               bopi=get_bo(r,rosi=p['ropi_'+bdn],bo1=p['bo3_'+bd],bo2=p['bo4_'+bd])

               plt.plot(r,bopi,label=r'$%s$' %bd, 
                        color=colors[i%len(colors)], linewidth=2, linestyle='-')
            if bt=='pp':
               # print(bd,'ropp',p['ropp_'+bdn],'bo5:',p['bo5_'+bd],'bo6',p['bo6_'+bd])
               bopp=get_bo(r,rosi=p['ropp_'+bdn],bo1=p['bo5_'+bd],bo2=p['bo6_'+bd])

               plt.plot(r,bopp,label=r'$%s$' %bd, 
                        color=colors[i%len(colors)], linewidth=2, linestyle='-')

        plt.legend()
        plt.savefig('bo_%s.svg' %bt) 
        plt.close()


def plot_bondorder():
    #atoms = structure('HMX')
    atoms = read('poscar.gen',index=-1)
    # atoms = atoms*[2,2,2]
    # view(atoms)


    # atoms = read('md.traj',index=-1)
    atom_name = atoms.get_chemical_symbols()
    ir = IRFF_NP(atoms=atoms,
                libfile='ffield.json',
                rcut=None,
                nn=True)
    ir.calculate_Delta(atoms)

    fig = plt.figure()
    # set figure information
    # plt.set_title("Bond-Order")
    plt.xlabel("Atom ID")
    plt.ylabel("Atom ID")

    plt.imshow(ir.bop,cmap='jet') # atom conection matrix
    plt.colorbar()
    # plt.grid()
    plt.savefig('bop.eps')        
    plt.close()

    fig = plt.figure()
    plt.xlabel("Atom ID")
    plt.ylabel("Atom ID")
    plt.imshow(ir.bo0,cmap='jet') # atom conection matrix
    plt.colorbar()
    # plt.grid()
    plt.savefig('bo.eps')        
    plt.close()


if __name__ == '__main__':
   plbo()

