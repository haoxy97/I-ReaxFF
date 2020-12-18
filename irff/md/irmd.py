#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argh
import argparse
from ..irff import IRFF
from ase.io import read,write
import numpy as np
from ase.optimize import BFGS
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.verlet import VelocityVerlet
from ase.io.trajectory import Trajectory
from ase import units


def fvr(x):
    xi  = np.expand_dims(x,axis=0)
    xj  = np.expand_dims(x,axis=1) 
    vr  = xj - xi
    return vr


def fr(vr):
    R   = np.sqrt(np.sum(vr*vr,axis=2))
    return R


def getBonds(natom,r,rcut):
    bonds = [] 
    for i in range(natom-1):
        for j in range(i+1,natom):
            if r[i][j]<rcut[i][j]:
               bonds.append((i,j))
    return bonds


class IRMD(object):
  ''' Intelligent Reactive Molecular Dynamics '''
  def __init__(self,atoms=None,gen='poscar.gen',ffield='ffield.json',
               index=-1,totstep=100,vdwnn=False,nn=True,
               initT=300,Tmax=25000,time_step=0.1,
               ro=None,rtole=0.6,Iter=0,bondTole=1.3,
               CheckDE=False,dEstop=0.5):
      self.Epot    = []
      self.epot    = 0.0
      self.ekin    = 0.0
      self.T       = 0.0
      self.initT   = initT
      self.Tmax    = Tmax
      self.totstep = totstep
      self.ro      = ro
      self.rtole   = rtole
      self.Iter    = Iter
      self.atoms   = atoms
      self.time_step = time_step
      self.step    = 0
      self.bondTole= bondTole
      self.CheckDE   = CheckDE
      self.dEstop    = dEstop

      if self.atoms is None:
         self.atoms   = read(gen,index=index)
      
      self.atoms.calc = IRFF(atoms=self.atoms,
                             libfile=ffield,
                             rcut=None,
                             nn=nn,vdwnn=vdwnn)
      self.natom     = len(self.atoms)
      self.re        = self.atoms.calc.re
      self.dyn       = None
      self.atoms.calc.calculate(atoms=self.atoms)
      self.InitBonds = getBonds(self.natom,self.atoms.calc.r,self.bondTole*self.re)

      if (self.atoms is None) and gen.endswith('.gen'):
         MaxwellBoltzmannDistribution(self.atoms, self.initT*units.kB)
      else:
         temp = self.atoms.get_temperature()
         if temp>0.0000001:
            scale = np.sqrt(self.initT/temp)
            p    = self.atoms.get_momenta()
            p    = scale * p
            self.atoms.set_momenta(p)
         else:
            MaxwellBoltzmannDistribution(self.atoms, self.initT*units.kB)


  def run(self):
      self.dyn = VelocityVerlet(self.atoms, self.time_step*units.fs,trajectory='md.traj')  
      def printenergy(a=self.atoms):
          epot_      = a.get_potential_energy()
          r          = a.calc.r.detach().numpy()
          i_         = np.where(np.logical_and(r<self.rtole*self.ro,r>0.0001))
          n          = len(i_[0])

          if len(self.Epot)==0:
             dE_ = 0.0
          else:
             dE_ = abs(epot_ - self.Epot[-1])
          self.Epot.append(epot_)

          self.epot  = epot_/self.natom
          self.ekin  = a.get_kinetic_energy()/self.natom
          self.T     = self.ekin/(1.5*units.kB)
          self.step  = self.dyn.nsteps

          print('Step %d Epot = %.3feV  Ekin = %.3feV (T=%3.0fK)  '
                'Etot = %.3feV' % (self.step,self.epot,self.ekin,self.T,
                                   self.epot + self.ekin))
          try:
             if self.CheckDE:
                assert n==0 and dE_<self.dEstop,'Atoms too closed or Delta E too high!' 
             else:
                assert n==0 and self.T<self.Tmax,'Atoms too closed or Temperature goes too high!' 
          except:
             # for _ in i_:
             #     print('atoms pair',_)
             print('Atoms too closed or Temperature goes too high, stop at %d.' %self.step)
             self.dyn.max_steps = self.dyn.nsteps-1
  
      # traj = Trajectory('md.traj', 'w', self.atoms)
      self.dyn.attach(printenergy,interval=1)
      # self.dyn.attach(traj.write,interval=1)
      self.dyn.run(self.totstep)
      # bonds      = getBonds(self.natom,self.atoms.calc.r,self.bondTole*self.re)
      # bondBroken = self.checkBond(bonds)
      # return bondBroken


  def opt(self):
      self.dyn = BFGS(self.atoms,trajectory='md.traj')
      def check(a=self.atoms):
          epot_      = a.get_potential_energy()
          r          = a.calc.r.detach().numpy()
          i_         = np.where(np.logical_and(r<self.rtole*self.ro,r>0.0001))
          n          = len(i_[0])
          
          self.Epot.append(epot_)
          self.epot  = epot_/self.natom

          self.step  = self.dyn.nsteps
          try:
             assert n==0 and (not np.isnan(epot_)),'Atoms too closed!'
          except:
             for _ in i_:
                 print('atoms pair',_)
             print('Atoms too closed or temperature too high, stop at %d.' %self.step)
             self.dyn.max_steps = self.dyn.nsteps-1
             # raise ValueError('-  Energy is NaN!' )
  
      self.dyn.attach(check,interval=1)
      self.dyn.run(0.00001,self.totstep)
      
      bonds      = getBonds(self.natom,self.atoms.calc.r,self.bondTole*self.re)
      bondBroken = self.checkBond(bonds)
      return bondBroken


  def checkBond(self,bonds):
      bondBroken = False
      for bd in self.InitBonds:
          bd_ = (bd[1],bd[0])
          if (bd not in bonds) and (bd_ not in bonds):
             bondBroken = True
             return bondBroken 
      return bondBroken


  def logout(self):
      with open('md.log','w') as fmd:
         fmd.write('\n------------------------------------------------------------------------\n')
         fmd.write('\n-       Energies From Machine Learning MD Iteration %4d               -\n' %self.Iter)
         fmd.write('\n------------------------------------------------------------------------\n')

         fmd.write('-  Ebond=%f  ' %self.atoms.calc.Ebond)
         fmd.write('-  Elone=%f  ' %self.atoms.calc.Elone)
         fmd.write('-  Eover=%f  ' %self.atoms.calc.Eover)
         fmd.write('-  Eunder=%f  \n' %self.atoms.calc.Eunder)
         fmd.write('-  Eang=%f  ' %self.atoms.calc.Eang)
         fmd.write('-  Epen=%f  ' %self.atoms.calc.Epen)
         fmd.write('-  Etcon=%f  \n' %self.atoms.calc.Etcon)
         fmd.write('-  Etor=%f  ' %self.atoms.calc.Etor)
         fmd.write('-  Efcon=%f  ' %self.atoms.calc.Efcon)
         fmd.write('-  Evdw=%f  ' %self.atoms.calc.Evdw)
         fmd.write('-  Ecoul=%f  \n' %self.atoms.calc.Ecoul)
         fmd.write('-  Ehb=%f  ' %self.atoms.calc.Ehb)
         fmd.write('-  Eself=%f  ' %self.atoms.calc.Eself)
         fmd.write('-  Ezpe=%f \n' %self.atoms.calc.zpe)
         
         fmd.write('\n------------------------------------------------------------------------\n')
         fmd.write('\n-              Atomic Information  (Delta and Bond order)              -\n')
         fmd.write('\n------------------------------------------------------------------------\n')
         fmd.write('\n  AtomID Sym  Delta      NLP      DLPC      Bond-Order  \n')

         for i in range(self.natom):
             fmd.write('%6d  %2s %9.6f %9.6f %9.6f' %(i,
                                      self.atoms.calc.atom_name[i],
                                      self.atoms.calc.Delta[i],
                                      self.atoms.calc.nlp[i],
                                      self.atoms.calc.Delta_lpcorr[i]))
             for j in range(self.natom):
                   if self.atoms.calc.bo0[i][j]>self.atoms.calc.botol:
                      fmd.write(' %3d %2s %9.6f' %(j,self.atoms.calc.atom_name[j],
                                                 self.atoms.calc.bo0[i][j]))
             fmd.write(' \n')

         fmd.write('\n------------------------------------------------------------------------\n')
         fmd.write('\n-                    Atomic Energies & Forces                          -\n')
         fmd.write('\n------------------------------------------------------------------------\n')

         fmd.write('\n  AtomID Sym Delta_lp     Elone      Eover      Eunder      Fx        Fy        Fz\n')
         for i in range(self.natom):
             fmd.write('%6d  %2s  %9.6f  %9.6f  %9.6f  %9.6f ' %(i,
                       self.atoms.calc.atom_name[i],
                       self.atoms.calc.Delta_lp[i],
                       self.atoms.calc.elone[i],
                       self.atoms.calc.eover[i],
                       self.atoms.calc.eunder[i]))

             for f in self.atoms.calc.results['forces'][i]:
                   fmd.write(' %9.6f ' %f)
             fmd.write(' \n')

         fmd.write('\n------------------------------------------------------------------------\n')
         fmd.write('\n- Machine Learning MD Completed!\n')


  def close(self):
      self.logout()
      self.dyn   = None
      self.atoms = None


if __name__ == '__main__':
   ''' use commond like ./irmd.py md --T=2800 to run it'''
   parser = argparse.ArgumentParser()
   argh.add_commands(parser, [md])
   argh.dispatch(parser)

   
