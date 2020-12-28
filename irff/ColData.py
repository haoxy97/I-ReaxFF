from os.path import exists,isfile
from os import system,getcwd,chdir
from .dft.prep_data import prep_data


class ColData(object):
  def __init__(self,max_batch=100):
      ''' max_batch: max number of batch 
              batch: batch size
      '''
      self.max_batch    = max_batch   # max number in direcs to train
      # self.get_ro(rodic)


  def __call__(self,label=None,dft='ase',batch=50):
      self.label  = label
      cwd         = getcwd()
      gen         = self.label+'.gen'
      self.direcs = {}
      i           = 0
      data_dir    = {}
      running     = True

      while running:
          run_dir = 'aimd_'+self.label+'/'+self.label+'-'+str(i)
          if exists(run_dir):
             i += 1
             data_dir[self.label+'-'+str(i)] = cwd+'/'+run_dir+'/'+self.label+'.traj'
          else:
             running = False

      trajs_ = prep_data(label=self.label,direcs=data_dir,
                         split_batch=batch,max_batch=self.max_batch,
                         frame=100000,dft=dft)              # get trajs for training
      return trajs_


#   def get_ro(self,rodic):
#       if rodic is None:
#          self.rodic= {'C-C':1.35,'C-H':1.05,'C-N':1.45,'C-O':1.35,
#                       'N-N':1.35,'N-H':1.05,'N-O':1.30,
#                       'O-O':1.35,'O-H':1.05,
#                       'H-H':0.8,
#                       'others':1.35} 
#       else:
#       	 self.rodic= rodic

#       atoms     = read('poscar.gen')
#       self.atom_name = atoms.get_chemical_symbols()
#       self.natom     = len(self.atom_name)
#       self.ro   = np.zeros([self.natom,self.natom])

#       for i in range(self.natom):
#           for j in range(self.natom):
#               bd  = self.atom_name[i] + '-' + self.atom_name[j]
#               bdr = self.atom_name[j] + '-' + self.atom_name[i]
#               if bd in self.rodic:
#                  self.ro[i][j] = self.rodic[bd]
#               elif bdr in self.rodic:
#                  self.ro[i][j] = self.rodic[bdr]
#               else:
#                  self.ro[i][j] = self.rodic['others']


#   def close(self):
#       print(' * Data collection compeleted.')
#       self.atom_name = None
#       self.ro        = None
#       self.c         = None


if __name__ == '__main__':
   getdata = ColData(batch=100)
   traj    = getdata(label='nm2')


