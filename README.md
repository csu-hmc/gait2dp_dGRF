# gait2dpi
Implict torque driven gait 2D model taking account of belt velocity perturbation

Introduction
------------

Gait2DPI is one dirvatives of Gait2D model. Gait2D is a dynamic model that 
contains the essential elements for simulating human gait in the sagittal 
plane. The model has been used in earlier forms by Ackermann and van den 
Bogert (2010) and by Gerritsen et al. (1998). The main features of the 
model are:

- Seven body segments
- Fast execution

Model dynamics and outputs are twice differentiable with respect to all inputs,
which is important for certain numerical methods for simulation and optimal
control. The model is intended for education and basic research. The model can
be used for studies such as Ackermann and van den Bogert (2010) and Geyer and
Herr (2010). More information about Gait2D model can be found in gait2d_reference.pdf

Gait2DPI is an implict model, which added walking surface velocity perturbation
on the base of Gait2D. This model can be used for finding optimal controller using direct collocation 
in perturbed standing and walking.

Changes
-------
The only change comparing to Gait2D model is the contact model. In original Gait2D model, walking surface
is assumed in static. In this gait2dpi model, x direction friction in contact model is calcuated based on the 
relative velocity between foot and walking surface.

- Caution:

    Velocity of walking surface here is only the perturbation, not include average walking speed.
    
    This model does has derivative of GRF with respect of model states

Speed
-----

The cythonized Autolev C code takes about 40 micro seconds per rhs eval under Core i5 
vPro processer.

Build
-----

To manually build the Autolev model and make use of it in Python::

   $ cd (Store Path...)/gait2dpi
   
   $ python setup.py build_ext --inplace

Usage
-----

See ``examples/run.py``.
