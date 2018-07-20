#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 25 12:01:37 2017

@author: huawei
"""

from test import gait2dpi_dGRF_test
gait = gait2dpi_dGRF_test()
gait.do_test('derivatives')

#f, diff_dfdx, dfdx, dfdx_num, diff_dGRFdx, dGRF_dx, dGRFdx_num = gait.do_test('derivatives')

#'stick' : Stick figure test
#'speed' : Speed test of the excutation
#'sparsity' : Sparsity test of the derivatives
#'grf' : Ground Reaction Force test
#'freefall' : Freefall dynamics test
#'derivatives' : Check derivatives of jacobian
