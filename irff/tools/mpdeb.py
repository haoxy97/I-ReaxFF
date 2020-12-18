#!/usr/bin/env python
from __future__ import print_function
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from os import system, getcwd, chdir,listdir,environ
from irff.mpnn import MPNN
from irff.reax import logger
# from train_reaxff import cnn
import numpy as np

environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logger = logger(flog='debug.log')


def gradb(direcs,v='bo5',bd='H-H',
          nn=True,bo_layer=[9,2],massages=1,
          debd=False,deba=True,deang=True,
          dft='siesta',batch=50):
    ''' variables like: bo1_C-H, boc5_C rosi_C-H boc1
    '''
    v_ = v
    v  = v+'_'+bd  
    print('-  grading ... ...')
    ffield = 'ffield.json' if nn else 'ffield'
    
    rn = MPNN(libfile=ffield,direcs=direcs,
              dft=dft,
              nn=nn,
              bo_layer=bo_layer,
              massages=massages,
              batch_size=batch,
              pkl=True)
    rn.initialize()
    rn.session(learning_rate=3.0e-4,method='AdamOptimizer') 

    if nn:
       bdlit = ['bop','bop_si','bop_pi','bop_pp',
                'F',
                'bosi','bopi','bopp',
                'powb','expb','sieng','EBD']
    else:
       bdlit = ['bop','bop_si','bop_pi','bop_pp',
                'f_1','f_2','f_3','f_4','f_5',
                'bosi','bopi','bopp',
                'powb','expb','sieng','EBD']

    if debd:
       bonds = rn.bonds
       for b in bonds:
           v  = v_ +'_'+ b 

           grad = rn.get_gradient(rn.Loss,rn.p[v]) 
           text_ = '-  the gradient of Loss/%s is ' %v
           logger.info( text_+str(grad))

           if grad is None:
              continue
           if not np.isnan(grad):
              continue

           if rn.nbd[b]>0:
              grad = rn.get_gradient(rn.__dict__['EBD'][b],rn.p[v]) 
              logger.info('-  the gradient of %s/%s is: %s' %('EBD'+'_'+b,v,str(grad))) 
              if not grad is None:
                 if np.isnan(grad):
                    for l in bdlit:
                        grad = rn.get_gradient(rn.__dict__[l][bd],rn.p[v]) 
                        logger.info('-  the gradient of %s/%s is: %s' %(l+'_'+b,v,str(grad))) 
    v  = v_ +'_'+ bd 
    if deba:
       sl = ['EL','EOV','EUN']
       alist = {'EL':['Delta_lp','Delta_e','explp'],
                'EOV':['Delta_lpcorr','Delta_lp','nlp','so','otrm1','otrm2'],
                'EUN':['expeu1','expeu3','Delta_lpcorr','Delta_lp','Delta_e','explp','nlp']}
       for sp in rn.spec:
           for l in sl:
               if sp in rn.__dict__[l]:
                  grad = rn.get_gradient(rn.__dict__[l][sp],rn.p[v]) 
                  logger.info('-  the gradient of %s/%s is: %s' %(l+'_'+sp,v,str(grad))) 

                  if not grad is None:
                     if np.isnan(grad):
                        for al in alist[l]:
                            grad = rn.get_gradient(rn.__dict__[al][sp],rn.p[v]) 
                            logger.info('-  the gradient of %s/%s is: %s' %(al+'_'+sp,v,str(grad))) 

    if deang:
       al = ['EANG','EPEN','ETC']
       for ang in rn.angs:
           # v  = 'val1'+'_'+ang  
           if rn.nang[ang]>0:
              for l in al:
                  grad = rn.get_gradient(rn.__dict__[l][ang],rn.p[v]) 
                  logger.info('-  the gradient of %s/%s is: %s' %(l+'_'+ang,v,str(grad))) 

    tl = ['ETOR','Efcon']
    for tor in rn.tors:
        # v  = 'tor2' # +'_'+tor  
        if rn.ntor[tor]>0:
           for l in tl:
               grad = rn.get_gradient(rn.__dict__[l][tor],rn.p[v]) 
               logger.info('-  the gradient of %s/%s is: %s' %(l+'_'+tor,v,str(grad))) 
    rn.sess.close()


if __name__ == '__main__':
   direcs = {'C2H4N2O2-0':'C2H4N2O2-0.traj',
          'C2H4N2O2-1':'C2H4N2O2-1.traj',
          'C2H4N2O2-2':'C2H4N2O2-2.traj',
          'tatbmd1':'tatbmd1.traj',
          'tatbs1':'tatbs1.traj',
          'tatbs2':'tatbs2.traj',
#         'tatblm0':'tatblm0.traj',
          'O2H4-0':'O2H4-0.traj',     ####### CHO
          'C1H6O1':'C1H6O1.traj',
          'C2H4-0':'C2H4-0.traj',
          'C2H6-0':'C2H6-0.traj',
          'C2H2-0':'C2H2-0.traj',
          'C4H8-0':'C4H8-0.traj',
          'c3h10':'C3H10.traj',
          'h2o':'H2O.traj',
          'c4h8stre':'C4H8.traj',
          'c2h4stre':'C2H4.traj',
          'C2H10O2':'C2H10O2.traj',
          'md1':'md1.traj',
          'md2':'md2.traj',
          'md3':'md3.traj',
          'C2H8O2':'C2H8O2.traj',
          'C2H18O7':'C2H18O7.traj',
          'O7H20C2':'O7H20C2.traj',
          'C2H30O130':'C2H30O130.traj',
          'C2H30O13-0':'C2H30O13-0.traj',
          'C2H30O13-1':'C2H30O13-1.traj',
          'C2H30O13-2':'C2H30O13-2.traj',
          'C2H30O13-3':'C2H30O13-3.traj',
          }
   gradb(direcs,v='bo1',bd='C-C',
          nn=True,bo_layer=[9,2],massages=1,
          debd=False,deba=True,deang=True,
          dft='siesta',batch=50)


