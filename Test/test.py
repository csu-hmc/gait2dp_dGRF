#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 23 10:46:43 2017

@author: huawei

This program runs various tests on the gait2dpi model.
"""
import yaml
import numpy as np
from gait2dpi_dGRF import evaluate_autolev_rhs as autolev_rhs
from simulate import map_values_to_autolev_symbols
import matplotlib.pyplot as plt
import time
from numpy import pi, sin, cos
import scipy.sparse as sps

class gait2dpi_dGRF_test(object):
    def __init__(self):
               
        # Some model related variables
        self.ndof = 9
        self.nstates = 2*self.ndof
        self.ncontrols = 9
        self.jointnames = ['Hip', 'Knee', 'Ankle', 'LHip', 'LKnee', 'LAnkle']
        self.njoints = np.size(self.jointnames)
        self.dofnames = ['trunk x','trunk y','trunk angle', 'Rhip angle','Rknee angle',
                    'Rankle angle', 'Lhip angle','Lknee angle','Lankle angle']
        
        # create a state where model is upright and in free fall
        self.xff = np.zeros(self.nstates)
        self.xdff = np.zeros(self.nstates)
        
        with open('data/example_constants.yml', 'r') as f:
            constants_dict = yaml.load(f)
        
        self.constants_dict = map_values_to_autolev_symbols(constants_dict)
        
#        QQ, dQQ_dp, dQQ_dpd, dQQ_dpdd, GRF, dGRFdq, dGRFdqd, sticks = self.gait2dpi_py(self.xff, self.xdff, vs0, self.constants_dict)
        
#        offset = sticks[10]    # vertical corrdinate of right heel
        self.xff[1] = 1.0  # raise the model up, so that foot is exactly touching the ground
        self.xdff[10] = -9.81  # state derivatives for free fall

    def gait2dpi_py(self, xs, xsd, vs, constants_dict):
        
        x = xs[:9]
        xd = xs[9:]
        xdd = xsd[9:]
        
        QQ, dQQ_dp, dQQ_dpd, dQQ_dpdd, GRF, dGRFdq, dGRFdqd, sticks = \
                                    autolev_rhs(x, xd, xdd, vs, constants_dict)
    
        return QQ, dQQ_dp, dQQ_dpd, dQQ_dpdd, GRF, dGRFdq, dGRFdqd, sticks
    
    
    def gait2dpi_u(self, xs, xsd, vs, constants_dict):
        
        x = xs[:9]
        xd = xs[9:]
        xdd = xsd[9:]
        
        nGRF = 6
        
        QQ, dQQ_dp, dQQ_dpd, dQQ_dpdd, GRF, dGRFdq, dGRFdqd, sticks = \
                                    autolev_rhs(x, xd, xdd, vs, constants_dict)
                                    
                                    
        f = np.zeros(self.nstates)
        
        f[:9] = xsd[:9] - xs[9:]
        f[9:] = -QQ
         
        dfdx = np.zeros((self.nstates, self.nstates))
        
        dfdx[:9,9:] = -np.eye(9)
        dfdx[9:,:9] = -np.reshape(dQQ_dp, (9,9))
        dfdx[9:,9:] = -np.reshape(dQQ_dpd, (9,9))
        
        dfdxd = np.zeros((self.nstates, self.nstates))
        
        dfdxd[:9,:9] = np.eye(9)
        dfdxd[9:,9:] = -np.reshape(dQQ_dpdd, (9,9))

        
        dGRF_dx = np.zeros((nGRF, self.nstates))
        
        dGRF_dx[:, :9] = np.reshape(dGRFdq, (nGRF,9))
        dGRF_dx[:, 9:] = np.reshape(dGRFdqd, (nGRF,9))
             
        return f, dfdx, dfdxd, GRF, dGRF_dx

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
            
            QQ, dQQ_dp, dQQ_dpd, dQQ_dpdd, GRF, dGRFdq, dGRFdqd, sticks_walk = self.gait2dpi_py(q, qd, vs0, self.constants_dict)
            
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
            Neval = 10000
            xr = np.random.random(self.nstates)
            xdr = np.random.random(self.nstates)
            start_time = time.time()
            for m in range(Neval):
                xr = np.random.random(self.nstates)
                xdr = np.random.random(self.nstates)
                QQ, dQQ_dp, dQQ_dpd, dQQ_dpdd, GRF, dGRFdq, dGRFdqd, sticks_walk = self.gait2dpi_py(xr, xdr, vs0, self.constants_dict)
                
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
                
                f, dfdx, dfdxd, GRF, dGRF_dx = self.gait2dpi_u(xr, xdr, vsr, self.constants_dict)
                
                nz[m,0] = np.count_nonzero(dfdx)
                nz[m,1] = np.count_nonzero(dfdxd)
                nz[m,2] = np.count_nonzero(dGRF_dx)
                
            plt.figure(2)
            
            plt.subplot(311)
            plt.plot(nz[:,0], label='count_nonzero(dQQ_dx)')
            plt.subplot(312)
            plt.plot(nz[:,1], label='count_nonzero(dQQ_dxd)')
            plt.subplot(313)
            plt.plot(nz[:,2], label='count_nonzero(dGRF_dx)')
            plt.legend()
        
        if command == 'grf':
            print('GRF test....')
            
            x = np.zeros(18)  # put model in upright position
            x[6] = pi/2  # flex the left hip, to move left foot away from ground
            x[5] = 0.3  # dorsiflex the right ankle so only heel is ground
            vs = np.zeros(2)
            xd = np.zeros(18)
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
                    QQ, dQQ_dp, dQQ_dpd, dQQ_dpdd, GRF, dGRFdq, dGRFdqd, sticks_walk = self.gait2dpi_py(x, xd, vs, self.constants_dict)
                    
                    Fy[j,i] = GRF[1]
            
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
                    QQ, dQQ_dp, dQQ_dpd, dQQ_dpdd, GRF, dGRFdq, dGRFdqd, sticks_walk = self.gait2dpi_py(x, xd, vs, self.constants_dict)
        
                    Fx[j, i] = GRF[0]
                    
            plt.subplot(212)
            plt.plot(vx, Fx)
            
            
        # ========================= do the free fall dynamics test
        if command == 'freefall':
            print('Freefall dynamics test...')
            
            x = self.xff
            x[1] = 0.954
            xd = self.xdff
            vs = np.zeros(2)
            # no generalized forces should be needed for this motion state
            
            f, dfdx, dfdxdot, GRF, dGRF_dx = self.gait2dpi_u(x, xd, vs, self.constants_dict)
            
            print('All plotted point should be zeros in freefall test...')
            plt.figure(4)
            plt.plot(f, '.')
        
        # ======================= do the derivatives test
        if command == 'derivatives':
            print('Checking derivatives...')
            
            # generate random motion state x, xd and controls u
#            qmin = [-2, 0.9, -1, -0.1, -1, -1, -0.1, -1, -1]
#            qmax = [2, 1.2, 1, 1, 0, 1, 1, 0, 1]
            qmin = [0, 0.96, 0.1, 0.1, -0.2, 0, -0.4, -0.6, -0.1]
            qmax = [0, 0.96, 0.1, 0.1, -0.2, 0, -0.4, -0.6, -0.1]
            
            qdmin = np.ndarray.tolist(-1*np.zeros(9))
            qdmax = np.ndarray.tolist(1*np.zeros(9))
            qddmin = np.ndarray.tolist(-1*np.zeros(9))
            qddmax = np.ndarray.tolist(1*np.zeros(9))
        
            xmin = np.array(qmin + qdmin)
            xmax = np.array(qmax + qdmax)
            xdmin = np.array(qdmin + qddmin)
            xdmax = np.array(qdmax + qddmax)
        
            x = xmin + (xmax-xmin)*np.random.random(self.nstates)
            xd = xdmin + (xdmax-xdmin)*np.random.random(self.nstates)
            vs = 0.0*np.random.random(2)
            x[1] = 1.5
            
            f, dfdx, dfdxd, GRF, dGRF_dx = self.gait2dpi_u(x, xd, vs, self.constants_dict)
            
#            plt.figure(figsize=(20,10))
#            
#            plt.subplot(311)
#            plt.spy(sps.csr_matrix(dfdx))
#            plt.subplot(312)
#            plt.spy(sps.csr_matrix(dfdxd))
#            plt.subplot(313)
#            plt.spy(sps.csr_matrix(dGRF_dx))
            
            dfdx_num = np.zeros((self.nstates, self.nstates))
            dfdxd_num = np.zeros((self.nstates, self.nstates))
            dGRFdx_num = np.zeros((6, self.nstates))
            
            for j in range(self.nstates):
                # estimate df/dx[j]
                h = 1e-7  # finite difference for x
                saved = x[j]
                x[j] = x[j] +h
                fh, dfdx, dfdxdot, GRFh, dGRF_dx = self.gait2dpi_u(x, xd, vs, self.constants_dict)
                dfdx_num[:,j] = (fh-f)/h
                dGRFdx_num[:,j] = (GRFh - GRF)/h
                x[j] = saved
                
                # compute df/dxd[j]
                saved = xd[j]
                if j > self.ndof:
                    h = 1e-7
                xd[j] = xd[j]+h
                fh, dfdx, dfdxdot, GRF, dGRF_dx = self.gait2dpi_u(x, xd, vs, self.constants_dict)
                dfdxd_num[:,j] = (fh-f)/h
                xd[j] = saved
                 
            diff_dfdx = dfdx-dfdx_num
            diff_dfdxd = dfdxd-dfdxd_num
            diff_dGRFdx = dGRF_dx - dGRFdx_num
            
            Rdiff_dfdx = np.zeros((18, 18))
            Rdiff_dfdxd = np.zeros((18, 18))
            Rdiff_dGRFdx = np.zeros((6, 18))
            
            for i in range(18):
                for j in range(18):
                    if np.abs(dfdx[i, j]) > 1e-5:
                        Rdiff_dfdx[i, j] = diff_dfdx[i, j]/dfdx[i, j]
                    if np.abs(dfdxd[i, j]) > 1e-5:
                        Rdiff_dfdxd[i, j] = diff_dfdxd[i, j]/dfdxd[i, j]
                for k in range(6):
                    if np.abs(dGRF_dx[k, i]) > 1e-5:
                        Rdiff_dGRFdx[k, i] = diff_dGRFdx[k, i]/dGRF_dx[k, i]
                        
            index1 = np.argwhere(np.abs(diff_dfdx).max() == np.abs(diff_dfdx))
            index2 = np.argwhere(np.abs(diff_dfdxd).max() == np.abs(diff_dfdxd))
            index3 = np.argwhere(np.abs(diff_dGRFdx).max() == np.abs(diff_dGRFdx))
            
            index4 = np.argwhere(np.abs(Rdiff_dfdx).max() == np.abs(Rdiff_dfdx))
            index5 = np.argwhere(np.abs(Rdiff_dfdxd).max() == np.abs(Rdiff_dfdxd))
            index6 = np.argwhere(np.abs(Rdiff_dGRFdx).max() == np.abs(Rdiff_dGRFdx))
            
            print('The maximum error in dfdx is %s  at row %s col %s  relative error is %s'
                  % (np.max(np.abs(diff_dfdx)), index1[0][0], index1[0][1], Rdiff_dfdx[index1[0][0], index1[0][1]]))
            print('The maximum error in df/dxd is %s  at row %s col %s  relative error is %s'
                  % (np.max(np.abs(diff_dfdxd)), index2[0][0], index2[0][1], Rdiff_dfdxd[index2[0][0], index2[0][1]]))
            print('The maximum error in dGRF/dx is %s  at row %s col %s  relative error is %s\n'
                  % (np.max(np.abs(diff_dGRFdx)), index3[0][0], index3[0][1], Rdiff_dGRFdx[index3[0][0], index3[0][1]]))  

            print('The maximum relative error in df/dx is %s  at row %s col %s  error is %s'
                  % (np.max(np.abs(Rdiff_dfdx)), index4[0][0], index4[0][1], diff_dfdx[index4[0][0], index4[0][1]]))
            print('The maximum relative error in df/dxd is %s  at row %s col %s  error is %s'
                  % (np.max(np.abs(Rdiff_dfdxd)), index5[0][0], index5[0][1], diff_dfdxd[index5[0][0], index5[0][1]]))
            print('The maximum relative error in dGRF/dx is %s  at row %s col %s  error is %s'
                  % (np.max(np.abs(Rdiff_dGRFdx)), index6[0][0], index6[0][1], diff_dGRFdx[index6[0][0], index6[0][1]]))
            
        #return f, diff_dfdx, dfdx, dfdx_num, diff_dGRFdx, dGRF_dx, dGRFdx_num             
