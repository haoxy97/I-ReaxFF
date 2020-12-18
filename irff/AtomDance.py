from __future__ import print_function
from os.path import isfile
from irff.irff_np import IRFF_NP
from ase.io import read,write
from ase.io.trajectory import TrajectoryWriter
from ase.calculators.singlepoint import SinglePointCalculator
import numpy as np


def getAtomsToMove(i,j,j_,ToBeMove,neighbors,ring=False):
    ToBeMove.append(j_)
    for n in neighbors[j_]:
        if n!=i:
           if n not in ToBeMove:
              ToBeMove,ring = getAtomsToMove(i,j,n,ToBeMove,neighbors)
        elif j_!=j and n==i:
           ring = True
    return ToBeMove,ring


def getNeighbor(natom,r,rcut,bo,botol=0.0):
    neighbors = [[] for _ in range(natom)]
    for i in range(natom-1):
        for j in range(i+1,natom):
            if r[i][j]<rcut[i][j] and bo[i][j]>=botol:
               neighbors[i].append(j)
               neighbors[j].append(i)
    return neighbors


def getBonds(natom,r,rcut,bo,botol=0.0):
    bonds = [] 
    for i in range(natom-1):
        for j in range(i+1,natom):
            # print(r[i][j],rcut[i][j],bo[i][j],botol)
            if r[i][j]<rcut[i][j] and bo[i][j]>=botol:
               bonds.append((i,j))
    return bonds


class AtomDance(object):
  def __init__(self,atoms=None,poscar=None,nn=True,rtole=0.5,bondTole=1.25,botol=0.0):
      self.rtole         = rtole
      self.bondTole      = bondTole
      self.botol         = botol
      self.BondDistrubed = []
      if atoms is None:
         if poscar is None:
            atoms  = read('poscar.gen')
         else:
            atoms  = read(poscar)

      self.ir = IRFF_NP(atoms=atoms,
                        libfile='ffield.json',
                        rcut=None,
                        nn=nn)
      self.natom     = self.ir.natom
      self.atom_name = self.ir.atom_name
      spec           = self.ir.spec
      self.mass      = atoms.get_masses()
     
      label_dic      = {}
      for sp in self.atom_name:
          if sp in label_dic:
             label_dic[sp] += 1
          else:
             label_dic[sp]  = 1
      self.label = ''
      for sp in spec:
          self.label += sp+str(label_dic[sp])

      self.ir.calculate_Delta(atoms)
      self.InitBonds = getBonds(self.natom,self.ir.r,self.bondTole*self.ir.re,self.ir.bo0,
                                botol=self.botol)

      self.neighbors = getNeighbor(self.natom,self.ir.r,self.bondTole*self.ir.re,self.ir.bo0,
                                   botol=self.botol)


  def bond_momenta_bigest(self,atoms):
      ratio = []
      s     = []
      for bd in self.InitBonds:
          i,j = bd
          ratio_     = self.ir.r[i][j]/self.ir.re[i][j]
          s_         = ratio_ -1.0
          s.append(s_)
          ratio.append(abs(s_))

      m_  = np.argmax(ratio)
      i,j = self.InitBonds[m_]
      s_  = s[m_] 
      if s_>=0.0:
         sign = 1.0
      else:
         sign = -1.0
      atoms = self.set_bond_momenta(i,j,atoms,sign=sign)
      return atoms


  def bond_momenta(self,atoms):
      ratio = []
      for bd in self.InitBonds:
          i,j = bd
          if bd not in self.BondDistrubed:
             s_ = self.ir.r[i][j]/self.ir.re[i][j] -1.0
             if s_>=0.0:
                sign = 1.0
             else:
                sign = -1.0
             self.BondDistrubed.append(bd)
             atoms = self.set_bond_momenta(i,j,atoms,sign=sign)
             return atoms,True
      return atoms,False


  def set_bond_momenta(self,i,j,atoms,sign=1.0):
      ha      = int(0.5*self.natom)
      # x     = atoms.get_positions()
      v       = np.zeros([self.natom,3])

      group_j = []
      group_j,ring = getAtomsToMove(i,j,j,group_j,self.neighbors)
      jg      = len(group_j)

      group_i = []
      group_i,ring = getAtomsToMove(j,i,i,group_i,self.neighbors)
      ig      = len(group_i)

      if ring:
         group_i = [i]
         group_j = [j]

      vij   = self.ir.vr[j][i]/self.ir.r[i][j]
      massi = 0.0
      massj = 0.0

      for a in group_i:
          massi += self.mass[a] 
      for a in group_j:
          massj += self.mass[a] 

      vi  = 1.0/massi
      vj  = 1.0/massj

      for a in group_i:
          v[a] = sign*vi*vij

      for a in group_j:
          v[a] = -sign*vj*vij
      atoms.set_velocities(v)
      return atoms


  def check_bond(self,atoms=None,mdtraj=None,bondTole=1.3):
      if atoms is None:
         atoms = self.ir.atoms
      if not bondTole is None:
         self.bondTole = bondTole
      self.ir.calculate_Delta(atoms,updateP=True)
      bkbd       = None
      bB_        = 0
      
      bondBroken = False
      bondTole_  = self.bondTole - 0.015
      bonds      = getBonds(self.natom,self.ir.r,bondTole_*self.ir.re,
                            self.ir.bo0,botol=self.botol*0.5)
      
      if len(bonds) == len(self.InitBonds):
         for bd in self.InitBonds:
             bd_ = (bd[1],bd[0])
             if (bd not in bonds) and (bd_ not in bonds):
                bkbd = bd
                bondBroken = True
                break
      else:
         bondBroken = True
      if bondBroken:
         bB_ += 1

      bondBroken = False
      bondTole_  = self.bondTole  
      bonds      = getBonds(self.natom,self.ir.r,bondTole_*self.ir.re,
                            self.ir.bo0,botol=self.botol)
      if len(bonds) == len(self.InitBonds):
         for bd in self.InitBonds:
             bd_ = (bd[1],bd[0])
             if (bd not in bonds) and (bd_ not in bonds):
                bondBroken = True
                break
      else:
         bondBroken = True
      if bondBroken:
         bB_ += 1
      return bB_,bkbd


  def check(self,wcheck=2,i=0,atoms=None,rtole=None):
      if atoms is None:
         atoms = self.ir.atoms
      if not rtole is None:
         self.rtole = rtole

      self.ir.calculate_Delta(atoms,updateP=True)

      fc = open('check.log','w')
      if i%wcheck==0:
         atoms = self.checkLoneAtoms(atoms,fc)
      else:
         atoms = self.checkLoneAtom(atoms,fc)

      atoms = self.checkClosedAtom(atoms,fc)
      fc.close()
      return atoms


  def checkLoneAtom(self,atoms,fc):
      for i in range(self.natom):
          if self.ir.Delta[i]<=self.ir.atol:
             print('- find an lone atom',i,self.atom_name[i],file=fc)
             sid = np.argsort(self.ir.r[i])
             for j in sid:
                 if self.ir.r[i][j]>0.0001:
                    print('  move lone atom to nearest neighbor: %d' %j,file=fc)
                    vr = self.ir.vr[i][j]
                    u = vr/np.sqrt(np.sum(np.square(vr)))
                    atoms.positions[i] = atoms.positions[j] + u*0.64*self.ir.r_cuta[i][j]
                    break
             self.ir.calculate_Delta(atoms)
      return atoms


  def checkLoneAtoms(self,atoms,fc):
      for i in range(self.natom):
          if self.ir.Delta[i]<=self.ir.atol:
             print('- find an lone atom',i,self.atom_name[i],file=fc)
             mid = np.argmin(self.ir.ND)
             
             if mid == i:
                continue

             print('- find the most atractive atom:',mid,file=fc)
             print('\n- neighors of atom %d %s:' %(i,self.atom_name[i]),end='',file=fc)
             neighs = []
             for j,bo in enumerate(self.ir.bo0[mid]):
                 if bo>self.ir.botol:
                    neighs.append(j)         
                    print(j,self.atom_name[j],end='',file=fc)
             print(' ',file=fc)

             if len(neighs)==0:
                vr = self.ir.vr[mid][i]
                u = vr/np.sqrt(np.sum(np.square(vr)))
                atoms.positions[i] = atoms.positions[mid] + u*0.64*self.ir.r_cuta[i][mid]
             elif len(neighs)==1:
                j = neighs[0]
                vr = self.ir.vr[mid][j]
                u = vr/np.sqrt(np.sum(np.square(vr)))
                atoms.positions[i] = atoms.positions[mid] + u*0.64*self.ir.r_cuta[i][mid]
             elif len(neighs)==2:
                i_,j_ = neighs
                xj = atoms.positions[mid]
                xi = 0.5*(atoms.positions[i_]+atoms.positions[j_])
                vr = xj - xi
                u = vr/np.sqrt(np.sum(np.square(vr)))
                vij = atoms.positions[j_]-atoms.positions[i_]
                rij = np.sqrt(np.sum(np.square(vij)))
                r_  = np.dot(vij,u)
                if r_!=rij:
                   atoms.positions[i] = atoms.positions[mid] + u*0.64*self.ir.r_cuta[i][mid]
             elif len(neighs)==3:
                i_,j_,k_ = neighs
                vi = atoms.positions[i_] - atoms.positions[j_]
                vj = atoms.positions[i_] - atoms.positions[k_]
                # cross product
                vr = np.cross(vi,vj)
                c  = (atoms.positions[i_]+atoms.positions[j_]+atoms.positions[k_])/3
                v  = atoms.positions[mid] - c
                u = vr/np.sqrt(np.sum(np.square(vr)))
                # dot product
                dot = np.dot(v,u)
                if dot<=0:
                   u = -u
                atoms.positions[i] = atoms.positions[mid] + u*0.64*self.ir.r_cuta[i][mid]

             self.ir.calculate_Delta(atoms)
      return atoms


  def checkClosedAtom(self,atoms,fc):
      self.ir.calculate_Delta(atoms)
      neighbors = getNeighbor(self.natom,self.ir.r,self.ir.r_cuta,self.ir.bo0)
      for i in range(self.natom-1):
          for j in range(i+1,self.natom):
              if self.ir.r[i][j]<self.rtole*self.ir.r_cuta[i][j]:
                 print('- atoms %d and %d too closed' %(i,j),file=fc)

                 moveDirection = self.ir.vr[j][i]/self.ir.r[i][j]
                 moveD         = self.ir.r_cuta[i][j]*(self.rtole+0.01) - self.ir.r[i][j]
                 moveV         = moveD*moveDirection
                                                               
                 ToBeMove = []
                 ToBeMove,ring = getAtomsToMove(i,j,j,ToBeMove,neighbors)
                 print('  atoms to to be moved:',ToBeMove,file=fc)
                 for m in ToBeMove:
                     newPos = atoms.positions[m] + moveV
                     r = np.sqrt(np.sum(np.square(newPos-atoms.positions[i])))
                     if r>self.ir.r[i][m]:
                        atoms.positions[m] = newPos
                 self.ir.calculate_Delta(atoms)
                 neighbors = getNeighbor(self.natom,self.ir.r,self.ir.r_cuta,self.ir.bo0)
      return atoms


  def bend(self,ang=None,rang=20.0,nbin=10,scale=1.2,wtraj=False):
      i,j,k = ang
      axis = [i,k]
      images = self.rotate(atms=[i,k],axis=axis,o=j,rang=rang,nbin=nbin,wtraj=wtraj,scale=scale)
      return images


  def bend_axis(self,axis=None,group=None,rang=20,nbin=30,scale=1.2,wtraj=False):
      images = self.rotate(atms=group,axis=axis,o=axis[0],rang=rang,nbin=nbin,wtraj=wtraj,scale=scale)
      return images


  def swing_group(self,ang=None,group=None,rang=20,nbin=30,scale=1.2,wtraj=False):
      i,j,k = ang
      atoms = self.ir.atoms
      self.ir.calculate_Delta(atoms)

      vij = atoms.positions[i] - atoms.positions[j] 
      vjk = atoms.positions[k] - atoms.positions[j]
      r   = self.ir.r[j][k]
      ujk = vjk/r
      ui  = vij/self.ir.r[i][j]
      uk  = np.cross(ui,ujk)
      rk  = np.sqrt(np.sum(uk*uk))
      
      if rk<0.0000001:
         uk = np.array([1.0,0.0,0.0])
      else:
         uk  = uk/rk       
      images = self.rotate(atms=group,axis_vector=uk,o=j,rang=rang,nbin=nbin,
                            wtraj=wtraj,scale=scale)
      return images


  def rotate(self,atms=None,axis=None,axis_vector=None,o=None,rang=20.0,nbin=10,wtraj=False,scale=1.2):
      da = 2.0*rang/nbin
      atoms = self.ir.atoms
      self.ir.calculate_Delta(atoms)
      neighbors = getNeighbor(self.natom,self.ir.r,scale*self.ir.re,self.ir.bo0)

      images = []
      if wtraj: his = TrajectoryWriter('rotate.traj',mode='a')

      if axis_vector is None:
         i,j   = axis
         vaxis = atoms.positions[j] - atoms.positions[i] 
         uk    = vaxis/self.ir.r[i][j]
      else:
         uk    = axis_vector

      a_     = -rang
      while a_<rang:
          atoms_ = atoms.copy()
          for atomk in atms:
              vo  = atoms.positions[atomk] - atoms.positions[o] 
              r_  = np.dot(vo,uk)

              o_  = atoms.positions[o] + r_*uk
              vi  = atoms.positions[atomk] - o_

              r   = np.sqrt(np.sum(np.square(vi)))
              ui  = vi/r
              uj  = np.cross(uk,ui)

              a   = a_*3.14159/180.0
              p   = r*np.cos(a)*ui + r*np.sin(a)*uj

              atoms_.positions[atomk] = o_ + p
              self.ir.calculate(atoms_)

              calc = SinglePointCalculator(atoms_,energy=self.ir.E)
              atoms_.set_calculator(calc)

          images.append(atoms_)
          if wtraj: his.write(atoms=atoms_)
          a_ += da
      return images


  def swing(self,ang,st=60.0,ed=180.0,nbin=50,scale=1.2,wtraj=False):
      da = (ed - st)/nbin
      i,j,k = ang
      atoms = self.ir.atoms
      self.ir.calculate_Delta(atoms)
      neighbors = getNeighbor(self.natom,self.ir.r,scale*self.ir.re,self.ir.bo0)
      images = []
      if wtraj: his = TrajectoryWriter('swing.traj',mode='w')

      vij = atoms.positions[i] - atoms.positions[j] 
      vjk = atoms.positions[k] - atoms.positions[j]
      r   = self.ir.r[j][k]
      ujk = vjk/r
      ui  = vij/self.ir.r[i][j]
      uk  = np.cross(ui,ujk)
      rk  = np.sqrt(np.sum(uk*uk))
      
      if rk<0.0000001:
         uk = np.array([1.0,0.0,0.0])
      else:
         uk  = uk/rk

      uj = np.cross(uk,ui)
      a_ = st

      while a_<ed:
            atoms_ = atoms.copy()
            a = a_*3.14159/180.0
            p = r*np.cos(a)*ui + r*np.sin(a)*uj
            atoms_.positions[k] = atoms_.positions[j]+p
            self.ir.calculate(atoms_)

            calc = SinglePointCalculator(atoms_,energy=self.ir.E)
            atoms_.set_calculator(calc)

            images.append(atoms_)
            if wtraj: his.write(atoms=atoms_)
            a_ += da

      if wtraj: his.close()
      return images


  def stretch(self,pair,atoms=None,nbin=20,st=0.7,ed=1.3,scale=1.25,traj=None,ToBeMoved=None):
      if atoms is None:
         atoms = self.ir.atoms
      self.ir.calculate_Delta(atoms)
      neighbors = getNeighbor(self.natom,self.ir.r,scale*self.ir.re,self.ir.bo0)
      images = []
 
      if not traj is None: his = TrajectoryWriter(traj,mode='w')
      #for pair in pairs:
      i,j = pair
      if ToBeMoved is None:
         ToBeMove = []
         ToBeMove,ring = getAtomsToMove(i,j,j,ToBeMove,neighbors)
      else:
         ToBeMove = ToBeMoved

      bin_     = (self.ir.re[i][j]*ed - self.ir.re[i][j]*st)/nbin
      moveDirection = self.ir.vr[j][i]/self.ir.r[i][j]

      if ring:
         return None

      for n in range(nbin):
          atoms_ = atoms.copy()
          moveV  = atoms.positions[i] + moveDirection*(self.ir.re[i][j]*st+bin_*n)-atoms.positions[j]
          # print(self.ir.re[i][j]*self.rtole+bin_*n)
          for m in ToBeMove:
              # sPos   = atoms.positions[i] + self.ir.re[i][m]*self.rtole*moveDirection
              newPos = atoms.positions[m] + moveV
              r = np.sqrt(np.sum(np.square(newPos-atoms.positions[i])))   
              atoms_.positions[m] = newPos

          self.ir.calculate(atoms_)
          i_= np.where(np.logical_and(self.ir.r<self.ir.re[i][j]*self.rtole-bin_,self.ir.r>0.0001))
          n = len(i_[0])
          try:
             assert n==0,'Atoms too closed!'
          except:
             print('Atoms too closed.')
             break
             
          calc = SinglePointCalculator(atoms_,energy=self.ir.E)
          atoms_.set_calculator(calc)
          images.append(atoms_)
          if not traj is None: his.write(atoms=atoms_)

      if not traj is None: his.close()
      return images


  def close(self):
      self.ir        = None
      self.atom_name = None

