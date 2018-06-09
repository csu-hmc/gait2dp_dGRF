#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 23 10:46:43 2017

@author: huawei

This program runs various tests on the gait2dpi model.
"""
import yaml
import numpy as np
from gait2dp_dyn import evaluate_autolev_rhs as autolev_rhs
from simulate import map_values_to_autolev_symbols
import matplotlib.pyplot as plt
import time
from numpy import pi, sin, cos
import scipy.sparse as sps

class gait2dpi_test(object):
    def __init__(self):
               
        # Some model related variables
        self.ndof = 9
        self.nstates = 2*self.ndof
		self.nmom = 6
        self.ncontrols = 9
        self.jointnames = ['Hip', 'Knee', 'Ankle', 'LHip', 'LKnee', 'LAnkle']
        self.njoints = np.size(self.jointnames)
        self.dofnames = ['trunk x','trunk y','trunk angle', 'Rhip angle','Rknee angle',
                    'Rankle angle', 'Lhip angle','Lknee angle','Lankle angle']
        
        # create a state where model is upright and in free fall
        self.xff = np.zeros(self.nstates)
        self.xdff = np.zeros(self.nstates)
		self.mom = np.zeros(self.nmom)
        vs0 = np.zeros(2)
        
        with open('data/example_constants.yml', 'r') as f:
            constants_dict = yaml.load(f)
        
        self.constants_dict = map_values_to_autolev_symbols(constants_dict)
        
        QQ, dQQ_dp, dQQ_dpd, dQQ_dpdd, dQQdmom, GRF, dGRFdq, dGRFdqd, sticks, tmp = self.gait2dpi_py(self.xff, self.xdff, vs0, self.mom, self.constants_dict)
        
        offset = sticks[4, 1]   # vertical corrdinate of right heel
        self.xff[1] = -offset     # raise the model up, so that foot is exactly touching the ground
        self.xdff[10] = -9.81      # state derivatives for free fall

    def gait2dpi_py(self, xs, xsd, vs, mom, constants_dict):
        
        x = xs[:9]
        xd = xs[9:]
        xdd = xsd[9:]
        
        QQ, dQQ_dp, dQQ_dpd, dQQ_dpdd, dQQ_dmom, GRF, dGRFdq, dGRFdqd, sticks, tmp = \
                                    autolev_rhs(x, xd, xdd, vs, mom, constants_dict)
    
        return QQ, dQQ_dp, dQQ_dpd, dQQ_dpdd, dQQ_dmom, GRF, dGRFdq, dGRFdqd, sticks, tmp
    
    
    def gait2dpi_u(self, xs, xsd, vs, mom, constants_dict):
        
        x = xs[:9]
        xd = xs[9:]
        xdd = xsd[9:]
        
        QQ, dQQ_dp, dQQ_dpd, dQQ_dpdd, dQQ_dmom, GRF, dGRFdq, dGRFdqd, sticks, tmp = \
                                    autolev_rhs(x, xd, xdd, vs, mom, constants_dict)
                                    
                                    
        f = np.zeros(self.nstates)
        
        f[:9] = xsd[:9] - xs[9:]
        f[9:] = QQ
         
        dfdx = np.zeros((self.nstates, self.nstates))
        
        dfdx[:9,9:] = -np.eye(9)
        dfdx[9:,:9] = np.reshape(dQQ_dp, (9,9))
        dfdx[9:,9:] = np.reshape(dQQ_dpd, (9,9))
        
        dfdxd = np.zeros((self.nstates, self.nstates))
        
        dfdxd[:9,:9] = np.eye(9)
        dfdxd[9:,9:] = np.reshape(dQQ_dpdd, (9,9))
        
        dfdu = np.zeros((self.nstates, self.nmom))
        
        dfdu[9:,:] = np.reshape(dQQ_dmom, (9,6))

        return f, dfdx, dfdxd, dfdmom

    def do_test(self, command):
        #========do the stickfigure test
        if command == 'stick':
            print('Stick figure test...')
            
            # make an typical walking pose
            q = [0, 0.96, 0.1, 0.1, -0.2, 0, -0.4, -0.6, -0.1]
            for k in range(9):
                q += [0]
            q = np.array(q)
            qd = np.zeros(self.nstates)
            vs0 = np.zeros(2)
			mom = np.zeros(self.nmom)
            
            QQ, dQQ_dp, dQQ_dpd, dQQ_dpdd, dQQ_dmom, GRF, dGRFdq, dGRFdqd, sticks_walk, tmp = self.gait2dpi_py(q, qd, vs0, mom, self.constants_dict)
            
            plt.figure(1)
            
            Rx = sticks_walk[2*np.array([0,1,2,3,4,5,3])+0]
            Ry = sticks_walk[2*np.array([0,1,2,3,4,5,3])+1]
            Lx = sticks_walk[2*np.array([1,6,7,8,9,7])+0]
            Ly = sticks_walk[2*np.array([1,6,7,8,9,7])+1]
            
            plt.plot(Rx, Ry, 'r-')
            plt.plot(Lx, Ly, 'b-')
            plt.axis('equal')
    
    
        #===================== do the speed test
        if command == 'speed':
            print('Speed test...')
            
            vs0 = np.zeros(2)
			mom = np.zeros(self.nmom)
            Neval = 10000
            xr = np.random.random(self.nstates)
            xdr = np.random.random(self.nstates)
            start_time = time.time()
            for m in range(Neval):
                xr = np.random.random(self.nstates)
                xdr = np.random.random(self.nstates)
                QQ, dQQ_dp, dQQ_dpd, dQQ_dpdd, dQQ_dmom, GRF, dGRFdq, dGRFdqd, sticks_walk, tmp = self.gait2dpi_py(xr, xdr, vs0, mom, self.constants_dict)
                
            print("Computation time for implicit dynamics with Jacobians (gait2dpmex): %s ms" % ((time.time() - start_time)*1000/10000))
                    
        #==================== do the sparsity test
        if command == 'sparsity':
            print('Sparsity test...')
            
            Neval = 10000
            nz = np.zeros((Neval,3))
            
            print('Doing %s function evaluations...', Neval)
            
            for m in range(Neval):
                xr = np.random.random(self.nstates)
                xdr = np.random.random(self.nstates)
                vsr = np.random.random(2)
                mom = np.random.random(self.nmom)
                
                f, dfdx, dfdxd, dfdu = self.gait2dpi_u(xr, xdr, vsr, mom, self.constants_dict)
                
                nz[m,0] = np.count_nonzero(dfdx)
                nz[m,1] = np.count_nonzero(dfdxd)
                nz[m,2] = np.count_nonzero(dfdu)
                
            plt.figure(2)
            
            plt.subplot(311)
            plt.plot(nz[:,0], label='count_nonzero(dQQ_dp)')
            plt.subplot(312)
            plt.plot(nz[:,1], label='count_nonzero(dQQ_dpd)')
            plt.subplot(313)
            plt.plot(nz[:,2], label='count_nonzero(dQQ_dpdd)')
            plt.legend()
        
        if command == 'grf':
            print('GRF test....')
            
            x = np.zeros(18)  # put model in upright position
            x[6] = pi/2  # flex the left hip, to move left foot away from ground
            x[5] = 0.3  # dorsiflex the right ankle so only heel is ground
            vs = np.zeros(2)
            xd = np.zeros(18)
            mom = np.zeros(self.nmom)
            # for vertical GRF test, use sinusoidal vertical motion
            
            cycle = np.linspace(0, 360, 361)*pi/180
            y = 0.9538 + 0.015*sin(cycle)
            freq = np.array([1, 2, 5, 10])
            omega = 2*pi*freq
            
            Fy = np.zeros((np.size(y), np.size(omega)))
            
            for i in range(np.size(omega)):
                vy = 0.02*omega[i]*cos(cycle)
                for j in range(np.size(y)):
                    x[1] = y[j]
                    x[10] = vy[j]
                    QQ, dQQ_dp, dQQ_dpd, dQQ_dpdd, dQQ_dmom, GRF, dGRFdq, dGRFdqd, sticks_walk, tmp = self.gait2dpi_py(x, xd, vs, mom, self.constants_dict)
                    
                    Fy[j,i] = grf[1]
            
            plt.figure(3)
            
            plt.subplot(211)
            plt.plot(y, Fy)
            # generate horizontal force_velocity curves
            
            x[10] = 0   # reset vertical velocity to zero
            vx = np.linspace(-0.2, 0.2, 401)
            nvx = np.size(vx)
            y = np.linspace(0.945, 0.961, 3)
            ny = np.size(y)
            Fx = np.zeros((nvx, ny))
            
            for i in range(ny):
                x[1] = y[i]
                for j in range(nvx):
                    x[9] = vx[j]                   
                    QQ, dQQ_dp, dQQ_dpd, dQQ_dpdd, dQQ_dmom, GRF, dGRFdq, dGRFdqd, sticks_walk, tmp = self.gait2dpi_py(x, xd, vs, mom, self.constants_dict)
        
                    Fx[j, i] = grf[0]
                    
            plt.subplot(212)
            plt.plot(vx, Fx)
            
            
        # ========================= do the free fall dynamics test
        if command == 'freefall':
            print('Freefall dynamics test...')
            
            x = self.xff
            xd = self.xdff
            u = np.zeros(9)
            vs = np.zeros(2)
            # no generalized forces should be needed for this motion state
            
            f, dfdx, dfdxdot, dfdu = self.gait2dpi_u(x, xd, vs, u, self.constants_dict)
            
            print('All plotted point should be zeros in freefall test...')
            plt.figure(4)
            plt.plot(f, '.')
        
        # ======================= do the derivatives test
        if command == 'derivatives':
            print('Checking derivatives...')
            
            # generate random motion state x, xd and controls u
            qmin = [-2, 0.8, -1, -0.1, -1, -1, -0.1, -1, -1]
            qmax = [2, 1.2, 1, 1, 0, 1, 1, 0, 1]
            qdmin = np.ndarray.tolist(-1*np.ones(9))
            qdmax = np.ndarray.tolist(1*np.ones(9))
            qddmin = np.ndarray.tolist(-1*np.ones(9))
            qddmax = np.ndarray.tolist(1*np.ones(9))
        
            xmin = np.array(qmin + qdmin)
            xmax = np.array(qmax + qdmax)
            xdmin = np.array(qdmin + qddmin)
            xdmax = np.array(qdmax + qddmax)
        
            umin = -100*np.ones(self.nmom)
            umax = 100*np.ones(self.nmom)
            
            x = xmin + (xmax-xmin)*np.random.random(self.nstates)
            xd = xdmin + (xdmax-xdmin)*np.random.random(self.nstates)
            u = umin + (umax-umin)*np.random.random(self.nmom)
            vs = np.random.random(2)
            
            f, dfdx, dfdxd, dfdu = self.gait2dpi_u(x, xd, vs, u, self.constants_dict)
            
            plt.figure(figsize=(20,10))
            
            plt.subplot(311)
            plt.spy(sps.csr_matrix(dfdx))
            plt.subplot(312)
            plt.spy(sps.csr_matrix(dfdxd))
            plt.subplot(313)
            plt.spy(sps.csr_matrix(dfdu))
            
            dfdx_num = np.zeros((self.nstates, self.nstates))
            dfdxd_num = np.zeros((self.nstates, self.nstates))
            dfdu_num = np.zeros((self.nstates, self.ncontrols))
            
            for j in range(self.nstates):
                # estimate df/dx[j]
                h = 1e-7  # finite difference for x
                saved = x[j]
                x[j] = x[j] +h
                fh, dfdx, dfdxdot, dfdu = self.gait2dpi_u(x, xd, vs, u, self.constants_dict)
                dfdx_num[:,j] = (fh-f)/h
                x[j] = saved
                 
                # compute df/dxd[j]
                saved = xd[j]
                if j > self.ndof:
                    h = 1
                xd[j] = xd[j]+h
                fh, dfdx, dfdxdot, dfdu = self.gait2dpi_u(x, xd, vs, u, self.constants_dict)
                dfdxd_num[:,j] = (fh-f)/h
                xd[j] = saved
                  
            # compute df/du
            for j in range(self.nmom):
                h = 1e-7
                saved = u[j]
                u[j] = u[j] + h
                fh, dfdx, dfdxdot, dfdu = self.gait2dpi_u(x, xd, vs, u, self.constants_dict)
                dfdu_num[:,j] = (fh-f)/h
                u[j] = saved
                 
            diff_dfdx = np.max(np.max(dfdx-dfdx_num))
            diff_dfdxd = np.max(np.max(dfdxd-dfdxd_num))
            diff_dfdu = np.max(np.max(dfdu-dfdu_num))
            
            
            print('Checking df/dx...the maximum difference is %s', diff_dfdx)
            print('Checking df/dxd...the maximum difference is %s', diff_dfdxd)
            print('Checking df/du...the maximum difference is %s', diff_dfdu)            
        
