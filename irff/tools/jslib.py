#!/usr/bin/env python
from __future__ import print_function
from irff.reaxfflib import read_lib,write_lib
# from irff.irnnlib_new import write_lib
from irff.qeq import qeq
from ase.io import read
import argh
import argparse
import json as js
from os import environ,system


def q(gen='packed.gen'):
    p,zpe,spec,bonds,offd,angs,torp,hbs= read_lib(libfile='ffield')
    A = read(gen)
    q = qeq(p=p,atoms=A)
    q.calc()



def i():
    p,zpe,spec,bonds,offd,angs,torp,hbs= read_lib(libfile='ffield')

    fj = open('ffield.json','w')
    # j = {'p':p,'m':[],'bo_layer':[],'zpe':[]}
    j = {'p':p,'m':None,
         'EnergyFunction':0,
         'MessageFunction':0, 
         'messages':1,
         'bo_layer':None,
         'bf_layer':None,
         'be_layer':None,
         'vdw_layer':None,
         'MolEnergy':None}
    js.dump(j,fj,sort_keys=True,indent=2)
    fj.close()


def ii():
    p,zpe,spec,bonds,offd,angs,torp,hbs= read_lib(libfile='ffield')
    write_lib(p_,spec,bonds,offd,angs,torp,hbs,libfile='ffield_')



def j():
    lf = open('ffield.json','r')
    j = js.load(lf)
    p_ = j['p']
    m_ = j['m']
    bo_layer_ = j['bo_layer']
    lf.close()

    spec,bonds,offd,angs,torp,hbs = init_bonds(p_)
    write_lib(p_,spec,bonds,offd,angs,torp,hbs,libfile='ffield')


def jj():
    lf = open('ffield.json','r')
    j = js.load(lf)
    p_ = j['p']
    m_ = j['m']
    bo_layer = j['bo_layer']
    lf.close()

    spec,bonds,offd,angs,torp,hbs = init_bonds(p_)
    p,zpe,spec,bonds,offd,angs,torp,hbs= read_lib(libfile='ffield')

    system('mv ffield.json ffield_.json')
    fj = open('ffield.json','w')
    j = {'p':p,'m':m_,'bo_layer':bo_layer,'zpe':None}
    js.dump(j,fj,sort_keys=True,indent=2)
    fj.close()


def t():
    lf = open('ffield.json','r')
    j = js.load(lf)
    p_ = j['p']
    m_ = j['m']
    bo_layer = j['bo_layer']
    lf.close()

    p = {}
    for key in p_:
        k = key.split('_')[0]
        if k=='n.u.':
           continue
        if k in ['V1','V2','V3','tor1','cot1']:
           k_ = key.split('_')[1]
           k2 = 'X' + k_[1:-1] + 'X'
           key_ = k + '_' + k2
           p[key_] = p_[key]
        else:
           p[key] = p_[key]


    system('mv ffield.json ffield_.json')
    fj = open('ffield.json','w')
    j = {'p':p,'m':m_,'bo_layer':bo_layer,'zpe':None}
    js.dump(j,fj,sort_keys=True,indent=2)
    fj.close()


def s():
    lf = open('ffield.json','r')
    j = js.load(lf)
    p  = j['p']
    m = j['m']
    bo_layer = j['bo_layer']
    lf.close()
    # spec,bonds,offd,angs,torp,hbs = init_bonds(p)

    m_ = {}
    for key in m:
        if key[1] == '1':
           m_[key] = m[key]

    system('mv ffield.json ffield_.json')
    fj = open('ffield.json','w')
    j = {'p':p,'m':m_,'bo_layer':bo_layer,'zpe':None}
    js.dump(j,fj,sort_keys=True,indent=2)
    fj.close()


def ct():
    with open('ffield.json','r') as lf:
         j = js.load(lf)
         p  = j['p']
         m = j['m']
         zpe=j['zpe']
         bo_layer = j['bo_layer']
         
    with open('ffield10t2D316.json','r') as lf:
         j_ = js.load(lf)
         mt2= j_['m']

    for key in mt2:
        if key[1] == '2':
           m[key] = mt2[key]

    system('mv ffield.json ffield_.json')

    with open('ffield.json','w') as fj:
         j = {'p':p,'m':m,'bo_layer':bo_layer,'zpe':zpe}
         js.dump(j,fj,sort_keys=True,indent=2)
 

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
              if kk[0]!=kk[1]:
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


if __name__ == '__main__':
   ''' use commond like ./gmd.py nvt --T=2800 to run it'''
   parser = argparse.ArgumentParser()
   argh.add_commands(parser, [q,i,ii,j,jj,s,ct,t])
   argh.dispatch(parser)

