from __future__ import print_function
from os.path import exists,isfile
from os import system,getcwd,chdir
from ase.io import read,write
from ase.io.trajectory import Trajectory
import matplotlib.pyplot as plt
from math import ceil
from .molecule import packmol
from .md.gmd import nvt_wt as gulp_nvt
from .dft.mdtodata import MDtoData
from .dft.prep_data import prep_data
from .reaxfflib import read_lib
import tensorflow as tf
from .md.irmd import IRMD
from .training import train
from .training import train_mpnn
from .AtomDance import AtomDance
from .dingtalk import send_msg
import numpy as np
import json as js
    

class ColData(object):
  def __init__(self,direcs=None,
               aimd='aimd',
               max_batch=100,
               batch=50,
               label=None,
               data_label='data',
               covergence=0.1,
               dft='siesta',
               rodic=None):
      ''' max_batch: max number of batch 
              batch: batch size
      '''
      self.batch        = batch
      self.direcs       = direcs
      self.aimd         = aimd
      self.max_batch    = max_batch   # max number in direcs to train

      self.get_ro(rodic)
      self.c = AtomDance(rtole=0.55)

      if label is None:
         self.label     = self.c.label
      else:
         self.label     = label

      self.data_label   = data_label
      self.dft          = dft


  def get_data(self):
      ''' recursive training loop '''
      it       = []
      e_gulp   = []
      e_siesta = []
      cwd      = getcwd()
      gen      = 'poscar.gen'

      if self.direcs is None:
         self.direcs = {}

      i        = 0
      data_dir = {}
      running  = True
      while running:
          run_dir = self.aimd+'/'+self.label+'-'+str(i)
          if exists(run_dir):
             i += 1
             data_dir[self.label+'-'+str(i)] = cwd+'/'+run_dir+'/'+self.label+'.traj'
          else:
             running = False

      trajs_ = prep_data(label=self.data_label,direcs=data_dir,
                         split_batch=self.batch,max_batch=self.max_batch,
                         frame=1000,dft=self.dft)              # get trajs for training
      # print(data_dir)
      return trajs_


  def get_ro(self,rodic):
      if rodic is None:
         self.rodic= {'C-C':1.35,'C-H':1.05,'C-N':1.45,'C-O':1.35,
                      'N-N':1.35,'N-H':1.05,'N-O':1.30,
                      'O-O':1.35,'O-H':1.05,
                      'H-H':0.8,
                      'others':1.35} 
      else:
      	 self.rodic= rodic

      atoms     = read('poscar.gen')
      self.atom_name = atoms.get_chemical_symbols()
      self.natom     = len(self.atom_name)
      self.ro   = np.zeros([self.natom,self.natom])

      for i in range(self.natom):
          for j in range(self.natom):
              bd  = self.atom_name[i] + '-' + self.atom_name[j]
              bdr = self.atom_name[j] + '-' + self.atom_name[i]
              if bd in self.rodic:
                 self.ro[i][j] = self.rodic[bd]
              elif bdr in self.rodic:
                 self.ro[i][j] = self.rodic[bdr]
              else:
                 self.ro[i][j] = self.rodic['others']


  def close(self):
      print(' * Data collection compeleted.')
      self.atom_name = None
      self.ro        = None
      self.c         = None


if __name__ == '__main__':
   gd = ColData(direcs=None,
                batch=50,
                data_label='case')
   gd.get_data()
   gd.close()


