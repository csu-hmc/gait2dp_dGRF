#!/usr/bin/env python

"""This is an example of how to use the gait2dpi model."""

import yaml
import numpy as np
from gait2dpi import evaluate_autolev_rhs as autolev_rhs

# local imports
from simulate import map_values_to_autolev_symbols

totaltime = 0

coordinate_values = np.random.random(9)
speed_values = np.random.random(9)
acceleration_values = np.random.random(9)
velocity_surface = np.random.random(2)
    
with open('data/example_constants.yml', 'r') as f:
    constants_dict = yaml.load(f)
    
constants_dict = map_values_to_autolev_symbols(constants_dict)

    
QQ, Jac_dQQ_dp, Jac_dQQ_dpd,Jac_dQQ_dpdd, grf, animation_data = \
    autolev_rhs(coordinate_values, speed_values, acceleration_values,
                velocity_surface, constants_dict)
