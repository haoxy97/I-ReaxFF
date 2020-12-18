import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from ase import Atoms
from ase.io import read,write
from irff.irff_np import IRFF_NP
from ase.io.trajectory import Trajectory


def MessagePassing(atoms,color={'C':'dimgray','H':'silver','O':'crimson','N':'dodgerblue'}, 
                   size={'C':320,'H':90,'O':180,'N':320},
                   bondColor='darkgoldenrod',boxColor='steelblue',bondWidth=2,
                   elev=45,azim=45,Axis=True,Box=True,t=0,text='edge',labelnode=False):
    ''' avilable colors: ghostwhite whitesmoke olive '''
    positions  = atoms.get_positions()
    sym        = atoms.get_chemical_symbols()

    ir = IRFF_NP(atoms=atoms,
                 libfile='ffield.json',
                 rcut=None,
                 nn=True)
    ir.calculate_Delta(atoms)

    # plot scatter points
    fig, ax = plt.subplots() 
    
#     ax.set_xlim([0,5])
#     ax.set_ylim([0,5])
    x_,y_,z_ = [],[],[]
#     circ=plt.Circle((1.5, 0.5),radius=0.2,color=(58/255,94/255,148/255))
#     ax.add_patch(circ)
    for i,atom in enumerate(atoms):
        x_.append(atom.x)
        y_.append(atom.y)
        z_.append(atom.z)
        plt.scatter(atom.y, atom.z, c=color[sym[i]],
                   marker='o',s=size[sym[i]],label=sym[i],
                   alpha=1)
        if text=='node':
           plt.text(atom.y-0.15,atom.z,r'$%3.2f$' %ir.Delta[i],
                    fontsize=20)
        else:
           if labelnode:
              plt.text(atom.y-0.11,atom.z,r'$%s%d$' %(ir.atom_name[i],i),
                       fontsize=20)
    xmin,ymin = np.min(atoms.positions[:,1]),np.min(atoms.positions[:,2])
    yl,yh=ax.get_ylim()
    yr = yh - yl
    xl,xh=ax.get_xlim()
    xr = xh - xl

    for i in range(ir.natom-1):
        for j in range(i+1,ir.natom):
            if ir.bo0[i][j]>0.05:
               x = [atoms.positions[i][0],atoms.positions[j][0]]
               y = [atoms.positions[i][1],atoms.positions[j][1]]
               z = [atoms.positions[i][2],atoms.positions[j][2]]
               # plt.plot(y,z,c=bondColor,linewidth=5,alpha=0.8)
               line = mlines.Line2D(y,z,lw=bondWidth*ir.bop[i][j],ls='-',alpha=1,color=bondColor)
               line.set_zorder(0)
               ax.add_line(line)
               if text=='edge':
                  # if ir.atom_name[i]=='C' and ir.atom_name[j]=='C':
                  #    plt.text(0.5*(y[0]+y[1])-0.05*xr,0.5*(z[0]+z[1]),
                  #             r'$BO=%3.2f$' %ir.H[t][i][j],
                  #             fontsize=16)
                  #    eb = ir.ebond[i][j]
                  #    si = ir.Hsi[t][i][j]
                  #    pi = ir.Hpi[t][i][j]
                  #    pp = ir.Hpp[t][i][j]
                  #    plt.text(0.5*(y[0]+y[1])-0.05*xr,0.5*(z[0]+z[1])+0.3*yr,
                  #             r'$BO^{\sigma}=%3.2f$' %si,fontsize=16)
                  #    plt.text(0.5*(y[0]+y[1])-0.05*xr,0.5*(z[0]+z[1])+0.2*yr,
                  #             r'$BO^{\pi}=%3.2f$' %pi,fontsize=16)
                  #    plt.text(0.5*(y[0]+y[1])-0.05*xr,0.5*(z[0]+z[1])+0.1*yr,
                  #             r'$BO^{\pi\pi}=%3.2f$' %pp,fontsize=16)
                  #    print('f_Ebond(%5.3f,%5.3f,%5.3f)=%5.3f' %(si,pi,pp,eb))
                  #    Di = ir.Delta[i]-ir.bo0[i][j]
                  #    Dj = ir.Delta[j]-ir.bo0[i][j]
                  #    if t>0:
                  #       plt.text(xmin+0.45,0.5*(z[0]+z[1])-0.1*yr,
                  #         r'$f_{Message}^{t=1}(%3.2f,%3.2f,%3.2f)=%3.2f$' %(Di,Dj,ir.bo0[i][j],ir.F[-1][i][j]),
                  #         fontsize=16)
                  #    print('f_BO(%5.3f,%5.3f,%5.3f)=%5.3f' %(Di,Dj,ir.bo0[i][j],ir.F[-1][i][j]))
                  # else:
                  plt.text(0.5*(y[0]+y[1])-0.15*xr,0.5*(z[0]+z[1]),
                           r'$(%3.2f,%3.2f,%3.2f)$' %(ir.Hsi[t][i][j],ir.Hpi[t][i][j],
                           ir.Hpp[t][i][j]),fontsize=16)

    ymin_ =min( ymin,yl+0.1*yr)
    plt.text(xmin,ymin_,r'$BO^{t=%d}$' %t,fontsize=16)
    plt.savefig('MessagePassing.svg',transparent=True)
    # plt.show()

# atoms = read('c2h6.gen')
# MessagePassing(atoms,color={'C':'grey','H':'steelblue','O':'crimson','N':'dodgerblue'}, 
#                size={'C':5000,'H':800,'O':5000,'N':5000},
#                bondColor='olive',boxColor='steelblue',bondWidth=10,
#                elev=0,azim=0,Axis=False,Box=False,text='edge',t=0)


