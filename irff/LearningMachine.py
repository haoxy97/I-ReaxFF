from __future__ import print_function
from os.path import exists,isfile
from os import getcwd,chdir,mkdir,listdir,popen #,system,
from ase.io import read,write
from ase.io.trajectory import Trajectory
import matplotlib.pyplot as plt
from math import ceil
from .molecule import packmol
from .dft.siesta import siesta_md 
from .dft.SinglePointEnergy import SinglePointEnergies
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


def plot_energies(it,edft,eamp,label_dft='SIESTA',label_ml='IRFF'):
    plt.figure()                                     # test
    mine = np.min(edft)

    plt.ylabel('Energies + %f (eV)' %(-mine))
    plt.xlabel('Iterations')
    
    plt.plot(edft-mine,linestyle='-',marker='o',markerfacecolor='snow',
             markeredgewidth=1,markeredgecolor='k',
             ms=5,color='k',alpha=0.8,label=label_dft)

    plt.plot(eamp-mine,linestyle='-',marker='^',markerfacecolor='snow',
             markeredgewidth=1,markeredgecolor='r',
             ms=5,color='r',alpha=0.8,label=label_ml)

    plt.legend(loc='best',edgecolor='yellowgreen')
    plt.savefig('energies.pdf')                      # transparent=True
    plt.close() 


def get_zpe(p,atom_name):
    zpe_ = 0.0
    for atom in atom_name:
        zpe_ += p['atomic_'+atom]
    return zpe_
    

class LearningMachine(object):
  ''' recursive learning machine constructed by I-ReaxFF and siesta '''
  def __init__(self,aimd=None,initConfig='poscar.gen',
               direcs=None,
               ncpu=40,
               maxiter=1000,
               dt_mlmd=0.1,
               dt_aimd=0.3,
               max_batch=10,
               dft_step=5,
               batch=50,
               step=20000,
               md_step=20,
               MaxMDstep=500,
               mpopt=1,
               messages=1,
               T=300,
               Tmax=1000,
               label=None,
               convergence=0.01,
               nconvergence=3,
               accCriteria=0.95,
               lossCriteria=1000.0,
               lossTole=1.0,
               accTole=0.005,
               accMax=0.96,
               accInc=1.0,
               resetAcc=5,
               learnWay=3,
               FreeAtoms=None,
               colFrame=10,
               mdInc=1.2,
               rodic=None,
               rtole=0.55,
               bondTole=1.3,
               CheckBond=False,
               EngTole=0.05,
               dEtole=0.2,
               d2Etole=0.05,
               dEstop=1.0,
               nn=True,vdwnn=False,
               bo_layer=[4,1],
               bf_layer=[9,2],
               be_layer=[9,2],
               EnergyFunction=3,
               MessageFunction=1,
               bom={'others':1.0},
               bore={'others':0.45},
               weight={'others':2.0},
               bo_penalty=10000.0,
               xcf='VDW',
               xca='DRSLL',
               basistype='DZP'):
      ''' max_batch: max number of batch 
              batch: batch size
      '''
      self.initConfig   = initConfig
      if aimd is None:
         lab_           = initConfig.split('.')[0]
         self.aidir     = 'aimd_' + lab_
      else:
         self.aidir     = aimd
      self.accCriteria  = accCriteria # accuracy convergence criteria
      self.lossCriteria = lossCriteria    
      self.lossTole     = lossTole    
      self.accTole      = accTole      # Minimum accuracy convergence after a training process
      self.accMax       = accMax 
      self.accInc       = accInc      # accuracy inrease factor
      self.resetAcc     = resetAcc    # reset accuracy criteria after this train cycle
      self.rtole        = rtole       # minimum bond length used to supervise ML-MD simulation
      self.bondTole     = bondTole
      self.CheckBond    = CheckBond
      self.convergence  = convergence
      self.nconvergence = nconvergence
      self.learnWay     = learnWay    # the way of learing
      self.ncpu         = ncpu
      self.step         = step
      self.batch        = batch
      self.direcs       = direcs
      self.maxiter      = maxiter
      self.max_batch    = max_batch   # max number in direcs to train
      self.xcf          = xcf
      self.xca          = xca
      self.basistype    = basistype
      self.dft_step     = dft_step
      self.dt_mlmd      = dt_mlmd
      self.dt_aimd      = dt_aimd
      self.T            = T
      self.Tmax         = Tmax
      self.md_step      = md_step
      self.colFrame     = colFrame
      self.EngTole      = EngTole
      self.dEtole       = dEtole
      self.d2Etole      = d2Etole
      self.dEstop       = dEstop
      self.MaxMDstep    = MaxMDstep
      self.mpopt        = mpopt
      self.messages     = messages
      self.mdInc        = mdInc       # MD step increase factor
      self.nn           = nn
      self.vdwnn        = vdwnn
      self.bo_layer     = bo_layer
      self.bf_layer     = bf_layer
      self.be_layer     = be_layer
      self.bore         = bore
      self.bom          = bom
      self.weight       = weight
      self.bo_penalty   = bo_penalty
      self.EnergyFunction = EnergyFunction
      self.MessageFunction= MessageFunction
      self.FreeAtoms    = FreeAtoms
      
      self.c = AtomDance(poscar=self.initConfig,nn=self.nn,
                         rtole=self.rtole,bondTole=self.bondTole)
      self.natom     = self.c.natom
      self.atom_name = self.c.atom_name
      self.get_ro(rodic)
      
      if self.mpopt==1:
         self.messages  = 1
      elif self.mpopt==2 or self.mpopt==3:
         self.messages  = 2

      if self.nn:
         self.trainer = train_mpnn
      else:
         self.trainer = train
      
      if label is None:
         aimd_ = self.aidir.split('_')
         if len(aimd_)>1:
            self.label  = aimd_[-1]
         else:
            self.label  = self.c.label
      else:
         self.label     = label

      if not isfile('learning.log'):
         with open('learning.log','w') as l:
           print('------------------------------------------------------------------------',file=l)
           print('-                                                                      -',file=l)
           print('-            On-the-Fly Learning-Machine by SIESTA and IRFF            -',file=l)
           print('-                          Author: Feng Guo                            -',file=l)
           print('-                     Email: gfeng.alan@gmail.com                      -',file=l)
           print('-                  https://github.com/fenggo/I-ReaxFF                  -',file=l)
           print('-                                                                      -',file=l)
           print('-    Please Cite: Computational Materials Science 172 (2020) 109393    -',file=l)
           print('-                                                                      -',file=l)
           print('------------------------------------------------------------------------',file=l)
           print('\n',file=l)


  def run(self):
      ''' recursive training loop '''
      it       = []
      e_gulp   = []
      e_siesta = []
      cwd      = getcwd()
      gen      = 'poscar.gen'

      if self.direcs is None:
         self.direcs = {}
      if not exists(self.aidir):
         mkdir(self.aidir)

      i            = 0
      step         = 0
      mdsteps      = self.md_step
      data_dir     = {}
      running      = True
      bondBroken   = 0
      self.save_config()
      converg      = False
      nconvergence = 0
      lw4          = True

      while running:
          learnWay = self.learnWay
          run_dir  = self.aidir+'/'+self.label+'-'+str(i)
          data_dir[self.label+'-'+str(i)] = cwd+'/'+run_dir+'/'+self.label+'.traj'

          if exists(run_dir):
             i += 1
             continue

          if isfile('md.traj'): 
             atoms = read('md.traj',index=-1)
          else:
             atoms = read(self.initConfig)
             atoms = self.c.bond_momenta_bigest(atoms)
             e_gmd,mdsteps = self.mlmd(cwd,run_dir,atoms,100000,i,learnWay) 
 
          if learnWay==4 and lw4:
             atoms,lw4 = self.c.bond_momenta(atoms)
          else:
             bondBroken,bkbd = self.c.check_bond(atoms=atoms,bondTole=self.bondTole)
          if not lw4: learnWay = 3

          atoms.write('poscar.gen')                                   # for aimd
          if learnWay>=3: 
             dft_step = int(mdsteps/self.colFrame)+1
          else:
             dft_step = self.dft_step

          e_aimd,eml_,dEmax_,d2Emax_ = self.aimd(cwd,run_dir,dft_step,gen,learnWay)
          e_siesta.append(e_aimd[0])                                   # get siesta results              
                                                                       # start training  
          trajs_ = prep_data(label=self.label,direcs=data_dir,
                             split_batch=self.batch,max_batch=self.max_batch,
                             frame=1000,dft='siesta')              # get trajs for training
          with open('training.log','w') as fe:                         # 
               fe.write('## training loss and accuries\n')             #
          trajs_.update(self.direcs)                                   #
          
          self.load_config()
          tf.compat.v1.disable_eager_execution()                       # training untill
          accu = -1000.0                                               # acc >= accMin
          tIter= 1   
          loss = self.lossCriteria+0.1      
          training = True                      
          while (accu<=self.accCriteria or loss>=self.lossCriteria) and training:        
                self.load_config()
                loss,accu,accMax,p,zpe,tstep = self.trainer(direcs=trajs_,
                                                     step=self.step,
                                                     batch=self.batch,
                                                     convergence=self.accCriteria,
                                                     lossConvergence=self.lossCriteria,
                                                     nn=self.nn,vdwnn=self.vdwnn,
                                                     mpopt=self.mpopt,
                                                     bo_layer=self.bo_layer,
                                                     bf_layer=self.bf_layer,
                                                     be_layer=self.be_layer,
                                                     EnergyFunction=self.EnergyFunction,
                                                     MessageFunction=self.MessageFunction,
                                                     bore=self.bore,
                                                     bom=self.bom,
                                                     weight=self.weight,
                                                     bo_penalty=self.bo_penalty)
                self.load_config()
                if tIter>=self.resetAcc and tstep>=self.step:
                   if accu>self.accCriteria-self.accTole and loss<self.lossCriteria+self.lossTole:
                      self.accCriteria = float(accMax)
                      self.lossCriteria= float(loss)
                      self.save_config()
                      training = False
                tIter += 1

          self.lossCriteria = float(loss)

          if bondBroken>=1 and self.CheckBond and (not bkbd is None):
             images = self.c.stretch(bkbd,atoms=atoms,nbin=20,st=self.bondTole-0.015,ed=1.0,scale=1.2,traj='md.traj')
             if images is None:
                atoms = self.c.bond_momenta_bigest(atoms)
                e_gmd,mdsteps,bondBroken = self.mlmd(cwd,run_dir,atoms,100000,i,learnWay) 
          else:
             e_gmd,mdsteps = self.mlmd(cwd,run_dir,atoms,100000,i,learnWay) 

          step += mdsteps                                     # run ML-MD test training results     

          if accu >= self.accCriteria and tstep<self.step and tIter<=2: # get md steps
             if self.md_step<self.MaxMDstep and mdsteps>=self.md_step:
                self.md_step = int(np.ceil(self.md_step*self.mdInc))    # 
             if self.accCriteria<self.accMax:                           # accracy adjusted dynamicly
                self.accCriteria = min(self.accCriteria*self.accInc,self.accMax)  
          elif accu < self.accCriteria:                                 # adjust MD steps according 
             if self.md_step>1:                                         # training results
                self.md_step  = int(self.md_step/self.mdInc)            #                 

          if self.md_step>=self.MaxMDstep*0.5:
             if self.T<self.Tmax:
                self.T = self.T*self.mdInc

          self.save_config()

          if learnWay>=3: e_gmd = eml_
          e_gulp.append(e_gmd[0])
          diff  = abs(e_gmd[0]-e_aimd[0])
          it.append(i)

          plot_energies(it,e_siesta,e_gulp)                           # plot learning status 
                                                                      #
          with open('energies.log','a') as fe:                        # write learning status 
               for i_,e_ in enumerate(e_gmd):                         # to file
                   if i_==0:                                          #
                      fe.write(' %d %f %f diff: %f \n' %(step+i_,e_gmd[i_],e_aimd[0],diff))
                   else:                                              #
                      fe.write(' %d %f \n' %(step+i_,e_gmd[i_]))      #
          
          with open('learning.log','a') as l:
               l.write('Iter: %d loss: %f acc: %f mds: %d Eirff: %f Esiesta: %f dEmax: %f d2Emax: %f\n' %(i,
                        loss,accu,mdsteps,e_gmd[i_],e_aimd[0],dEmax_,d2Emax_))

          converg_ = converg
          if diff<=self.convergence: 
             converg = True
          else:
             converg = False

          if converg and converg_: 
             nconvergence += 1
          else:
             nconvergence  = 0
          if nconvergence > self.nconvergence:
             print('-  Convergence occured.')
             send_msg('-  Convergence occured.')
             return diff

          if i>self.maxiter:
             print('-  Max iteration reached,the loss %7.4f and accuracy %7.4f.' %(loss,accu))
             send_msg('-  Max iteration reached, the loss %7.4f and accuracy %7.4f.' %(loss,accu))
             return diff
          i += 1


  def aimd(self,cwd,run_dir,tstep,gen,learnWay):
      mkdir(run_dir)
      chdir(cwd+'/'+run_dir)
      popen('cp ../../*.psf ./')
      emlmd = []
      dEmax = 0.0
      d2Emax = 0.0

      if learnWay<=2:
         images = siesta_md(label=self.label,ncpu=self.ncpu,T=self.T,dt=self.dt_aimd,us='F',tstep=tstep,
                            gen=cwd+'/'+gen,FreeAtoms=self.FreeAtoms,
                            xcf=self.xcf,xca=self.xca,basistype=self.basistype)
         eaimd = [images[0].get_potential_energy()]
      elif learnWay>=3:
         popen('cp ../../md.traj ./')
         popen('cp ../../ffield.json ./')
         # print(' * files in this dir \n',listdir())
         E,E_,dEmax,d2Emax = SinglePointEnergies('md.traj',label=self.label,EngTole=self.EngTole,
                                                 frame=tstep,select=True,
                                                 dE=self.dEtole,d2E=self.d2Etole,
                                                 xcf=self.xcf,xca=self.xca,basistype=self.basistype,
                                                 cpu=self.ncpu)
         eaimd = E
         emlmd = E_
      chdir(cwd)
      return eaimd,emlmd,dEmax,d2Emax


  def mlmd(self,cwd,run_dir,atoms,Tmax,Iter,learnWay):        # run classic MD to test training results
      irmd = IRMD(atoms=atoms,time_step=self.dt_mlmd,totstep=self.md_step,Tmax=Tmax,
                  ro=self.ro,rtole=self.rtole,Iter=Iter,initT=self.T,
                  bondTole=self.bondTole,
                  CheckDE=True,dEstop=self.dEstop,
                  nn=self.nn,vdwnn=self.vdwnn)
      if learnWay==2:
         irmd.opt()
      else:
         irmd.run()
      mdsteps= irmd.step
      Emd  = irmd.Epot
      irmd.close()
      return Emd,mdsteps


  def get_ro(self,rodic):
      if rodic is None:
         self.rodic= {'C-C':1.35,'C-H':1.05,'C-N':1.45,'C-O':1.35,
                      'N-N':1.35,'N-H':1.05,'N-O':1.30,
                      'O-O':1.35,'O-H':1.05,
                      'H-H':0.8,
                      'others':1.35} 
      else:
      	 self.rodic= rodic

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


  def save_config(self):
      with open('options.json','w') as fj:
           InPut = {'accCriteria':self.accCriteria,
                    'lossCriteria':self.lossCriteria,
                    'lossTole':self.lossTole,
                    'accTole':self.accTole,
                    'accMax':self.accMax,
                    'accInc':self.accInc,
                    'resetAcc':self.resetAcc,
                    'step':self.step,
                    'dft_step':self.dft_step,
                    'md_step':self.md_step,
                    'MaxMDstep':self.MaxMDstep,
                    'EngTole':self.EngTole,
                    'dEtole':self.dEtole,
                    'd2Etole':self.d2Etole,
                    'dEstop':self.dEstop,
                    'mdInc':self.mdInc,
                    'dt_aimd':self.dt_aimd,
                    'dt_mlmd':self.dt_mlmd,
                    'rtole':self.rtole,
                    'Tmax':self.Tmax,
                    'T':self.T,
                    'maxiter':self.maxiter,
                    'max_batch':self.max_batch,
                    'mpopt':self.mpopt,
                    'learnWay':self.learnWay,
                    'colFrame':self.colFrame,
                    'bore':self.bore}
           js.dump(InPut,fj,sort_keys=True,indent=2)


  def load_config(self):
      with open('options.json','r') as fj:
           InPut = js.load(fj)
           self.accCriteria  = InPut['accCriteria']
           self.lossCriteria = InPut['lossCriteria']
           self.lossTole     = InPut['lossTole']
           self.accTole      = InPut['accTole']
           self.accMax       = InPut['accMax']
           self.accInc       = InPut['accInc']
           self.resetAcc     = InPut['resetAcc']
           self.step         = InPut['step'] 
           self.dft_step     = InPut['dft_step'] 
           self.md_step      = InPut['md_step'] 
           self.MaxMDstep    = InPut['MaxMDstep'] 
           self.EngTole      = InPut['EngTole'] 
           self.dEtole       = InPut['dEtole'] 
           self.d2Etole      = InPut['d2Etole'] 
           self.dEstop       = InPut['dEstop'] 
           self.mdInc        = InPut['mdInc'] 
           self.dt_aimd      = InPut['dt_aimd'] 
           self.dt_mlmd      = InPut['dt_mlmd'] 
           self.rtole        = InPut['rtole'] 
           self.Tmax         = InPut['Tmax'] 
           self.T            = InPut['T'] 
           self.maxiter      = InPut['maxiter']
           self.max_batch    = InPut['max_batch']
           self.mpopt        = InPut['mpopt']
           self.learnWay     = InPut['learnWay']
           self.colFrame     = InPut['colFrame']
           self.bore         = InPut['bore']


  def close(self):
      print('-  LM compeleted.')
      self.atom_name = None
      self.ro        = None



if __name__ == '__main__':
   direcs = {'nm6_5':'/home/feng/siesta/nm6_5',
             'nm6_14':'/home/feng/siesta/nm6_14' }

   lm = LearningMachine(direcs=direcs,
                       ncpu=4,
                       maxiter=200,
                       dt_mlmd=0.1,
                       dt_aimd=1.0,
                       max_batch=20,
                       dft_step=20,
                       batch=50,
                       step=20000,
                       md_step=10,
                       MaxMDstep=20,
                       T=350,
                       Tmax=500,
                       convergence=0.0001,
                       accCriteria=0.943,
                       accMax=0.94,
                       accMin=0.92,
                       accInc=1.00002,
                       resetAcc=2,
                       mdInc=1.1,
                       nn=True,
                       mpopt=1,
                       messages=1,
                       learnWay=3,
                       colFrame=10,
                       rtole=0.6,
                       EngTole=0.1,
                       dEtole=0.20,
                       d2Etole=0.05,
                       dEstop=0.6,
                       bo_layer=[4,1],
                       bf_layer=[9,3],
                       be_layer=[9,2],
                       EnergyFunction=3,
                       MessageFunction=1,
                       FreeAtoms=None)
   lm.run()
   lm.close()


