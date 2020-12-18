#!/usr/bin/env python
# coding: utf-8
from irff.mpnn import MPNN


def train_():
    # this trajectory files can visulized by ase, i.e. use commond 
    # can see the configurations "ase gui C2H2-0.traj" 
    direcs = {'C2H2-0':'../data/C2H2-0.traj',
              'C2H30O130':'../data/C2H30O130.traj',
              'C2H40':'../data/C2H40.traj',
              'nm1-5':'../data/C1N1O2H3-0.traj',
              'nmb':'../data/nmb.traj',
              'h2osw':'../data/h2osw.traj',
              'nm2l':'../data/nm2l.traj',
              'C4N4O8H12-3-0':'../data/nml3-0.traj',
              'C4N4O8H12-5-0':'../data/nml5-0.traj',
              'C4N4O8H12-5-1':'../data/nml5-1.traj', 
              'C4N4O8H12-6-0':'../data/nml6-0.traj',
              'C4N4O8H12-6-1':'../data/nml6-1.traj', 
              'C4N4O8H12-7-1':'../data/nml7-1.traj',
              'C4N4O8H12-8-0':'../data/nml8-0.traj',
              'C4N4O8H12-9-0':'../data/nml9-0.traj',
              'C4N4O8H12-11':'../data/nml11.traj',
              'C4N4O8H12-10-0':'../data/nml10-0.traj',
              'C4N4O8H12-10-1':'../data/nml10-1.traj',
              'C4N4O8H12-12-1':'../data/nml12-1.traj',
              'C4N4O8H12-13-0':'../data/nml13-0.traj',
              'C4N4O8H12-140':'../data/nml14-0.traj',
              'C4N4O8H12-141':'../data/nml14-1.traj',
              'nm1-CN':'../data/nm1-CN.traj',
              'c2h6-0':'../data/c2h6-1-0.traj',
              'c2h6-1':'../data/c2h6-1-1.traj',
              'c2h6-2':'../data/c2h6-1-2.traj',
              'c2h6-3':'../data/c2h6-1-3.traj',
              'c2h6-4':'../data/c2h6-1-4.traj',
              'c2h4-0':'../data/c2h4-1-0.traj',
              'c2h4-1':'../data/c2h4-1-1.traj',
              'c2h4-2':'../data/c2h4-1-2.traj',
              'c2h4-3':'../data/c2h4-1-3.traj',
              'c2h4-CC':'../data/c2h4-CC.traj',
              'c2h2-CC':'../data/c2h2-CC.traj',
              'c2h6-CC':'../data/c2h6-CC.traj',
              'c2h2-1-0':'../data/c2h2-1-0.traj',
              'c2h2-1-1':'../data/c2h2-1-1.traj',
              'c2h2-1-2':'../data/c2h2-1-2.traj',
              'c2h2-1-3':'../data/c2h2-1-3.traj',
              }


    mn = MPNN(libfile='ffield.json',
              direcs=direcs, 
              dft='siesta',
              weight={'c2h6-CC':50.0,'others':2.0},
              messages=1,
              bo_layer=[4,1],
              bf_layer=[12,3],
              be_layer=[9,2],
              EnergyFunction=3,
              MessageFunction=1,
              pkl=False,
              batch_size=50,
              losFunc='n2',
              convergence=0.98)

    mn.run(learning_rate=1.0e-4,step=100,print_step=10,writelib=100) 
    mn.close()

if __name__ == '__main__':
   train_()



