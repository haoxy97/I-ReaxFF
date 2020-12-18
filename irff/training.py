#!/usr/bin/env python
# -*- coding: utf-8 -*-
from os import system, getcwd, chdir,listdir
from os.path import isfile # exists
import argh
import argparse
import numpy as np
from .reax import ReaxFF
from .mpnn import MPNN
from .initCheck import Init_Check
# from .dingtalk import send_msg
import json as js


# direcs v={'ethw':'/home/gfeng/siesta/train/case1',
#           'ethw1':'/home/gfeng/siesta/train/case1/run_1'}
# batch = 50


def train(direcs=None,step=5000,batch=None,convergence=0.97,lossConvergence=1000.0,
          nanv={'pen3':-2.0,'pen4':-2.0,'tor2':-5.0,'tor4':-5.0,
                'ovun3':-2.0,'ovun4':-2.0,
                'lp1':-3.0,'vdw1':-3.0},
          spec=[],
          writelib=1000,nn=True,vdwnn=False,
          bo_layer=[3,1],
          bf_layer=[9,2],
          be_layer=[3,1],
          EnergyFunction=3,MessageFunction=2,mpopt=3,
          bore={'others':0.45},
          bom={'others':1.20},
          weight={'others':10.0},
          bo_penalty=10000.0,
          ffield = 'ffield.json'):
    ''' training the force field '''
    rn = ReaxFF(libfile=ffield,
                direcs=direcs, 
                dft='siesta',
                spec=spec,
                optword='nocoul',
                opt=None,
                nn=nn,
                pkl=False,
                batch_size=batch,
                losFunc='n2',
                bo_penalty=10000.0,
                nanv=nanv,
                weight=weight,
                bore=bore,
                convergence=convergence,
                lossConvergence=lossConvergence) # Loss Functon can be n2,abs,mse,huber
 
    loss,accu,accMax,i,zpe =rn.run(learning_rate=1.0e-4,
                      step=step,
                      print_step=10,
                      writelib=writelib) 

    libstep = int(i - i%writelib)

    if i==libstep:
       libstep = libstep - writelib
    if libstep<=0:
       ffd = 'ffield.json'
    else:
       ffd = 'ffield_' + str(libstep) +'.json'

    if loss==0.0 and accu==0.0:
       # send_msg('-  Warning: the loss is NaN, parameters from %s changed auomatically ...' %ffd)
       with open(ffd,'r') as fj:
            j = js.load(fj)
            ic = Init_Check(nanv=nanv)
            j['p'] = ic.auto(j['p'])
            ic.close()
       with open('ffield.json','w') as fj:
            js.dump(j,fj,sort_keys=True,indent=2)

    p   = rn.p_
    rn.close()
    return loss,accu,accMax,p,zpe,i


def train_mpnn(direcs=None,step=5000,batch=None,convergence=0.97,lossConvergence=1000.0,
          nanv={'pen3':-2.0,'pen4':-2.0,'tor2':-5.0,'tor4':-5.0,
                'ovun3':-2.0,'ovun4':-2.0,
                'lp1':-3.0,'vdw1':-3.0},
          spec=[],
          writelib=1000,nn=True,vdwnn=False,
          bo_layer=[3,1],
          bf_layer=[9,2],
          be_layer=[3,1],
          EnergyFunction=3,MessageFunction=2,mpopt=3,
          bore={'others':0.45},
          bom={'others':1.0},
          weight={'others':10.0},
          bo_penalty=10000.0,
          ffield = 'ffield.json'):
    ''' train the massage passing model '''
    opt_ = None
    if mpopt==1:
       mpopt_=[True,True,True,True]
       messages=1
    elif mpopt==2:
       mpopt_=[False,True,True]
       messages=2
       opt_=[]
    elif mpopt==3:
       mpopt_=[True,True,True,True]
       messages=2

    rn = MPNN(libfile=ffield,
                direcs=direcs, 
                dft='siesta',
                spec=spec,
                optword='nocoul',
                opt=opt_,
                nn=nn,vdwnn=vdwnn,
                bo_layer=bo_layer,
                bf_layer=bf_layer,
                be_layer=be_layer,
                EnergyFunction=EnergyFunction,
                MessageFunction=MessageFunction,
                messages=messages,
                mpopt=mpopt_,
                pkl=False,
                batch_size=batch,
                losFunc='n2',
                bo_penalty=bo_penalty,
                nanv=nanv,
                bom=bom,bore=bore,
                weight=weight,
                convergence=convergence,
                lossConvergence=lossConvergence) # Loss Functon can be n2,abs,mse,huber

    loss,accu,accMax,i,zpe =rn.run(learning_rate=1.0e-4,
                               step=step,
                               print_step=10,
                               writelib=writelib) 

    libstep = int(i - i%writelib)

    if i==libstep:
       libstep = libstep - writelib
    if libstep<=0:
       ffd = 'ffield.json'
    else:
       ffd = 'ffield_' + str(libstep) + '.json'

    p   = rn.p_
    if loss==0.0 and accu==0.0:
       # send_msg('-  Warning: the loss is NaN, parameters from %s changed auomatically ...' %ffd)
       # with open(ffd,'r') as fj:
       #      j = js.load(fj)
       #      ic = Init_Check(nanv=nanv)
       #      j['p'] = ic.auto(j['p'])
       #      ic.close()
       # with open('ffield.json','w') as fj:
       #      js.dump(j,fj,sort_keys=True,indent=2)
       return loss,accu,accMax,p,zpe,i

    rn.close()
    return loss,accu,accMax,p,zpe,i


def sa(direcs=None,step=5000,batch=None,total_step=2):
    rn = ReaxFF(libfile='ffield',
              direcs=direcs, 
              dft='siesta',
              spec=['C','H','O','N'],
              ro_scale=0.1,
              optword='nocoul',
              atomic=True,
              cons=None,
              opt=None,
              pkl=False,
              batch_size=batch,
              losFunc='n2',
              bo_penalty=100.0) # Loss Functon can be n2,abs,mse,huber


    rn.sa(total_step=total_step,
           step=step,astep=1000,zstep=500,
           print_step=10,writelib=1000) 

    p   = rn.p_
    zpe = rn.zpe_
    rn.close()
    return loss,accu,p,zpe


if __name__ == '__main__':
   ''' use commond like ./bp.py <t> to run it
       z:   optimize zpe 
       t:   train the whole net
   '''
   parser = argparse.ArgumentParser()
   argh.add_commands(parser, [z,sa,train])
   argh.dispatch(parser)

