import numpy as np
cimport numpy as np

cdef extern from "gait2dpi_dGRF.h":

    cdef enum:
        NDOF = 9
        NMOM = 6
        NSTICK = 10

    ctypedef struct param_struct:
        double TrunkMass
        double TrunkInertia
        double TrunkCMy
        double ThighMass
        double ThighInertia
        double ThighCMy
        double ThighLen
        double ShankMass
        double ShankInertia
        double ShankCMy
        double ShankLen
        double FootMass
        double FootInertia
        double FootCMx
        double FootCMy
        double ContactY
        double ContactHeelX
        double ContactToeX
        double ContactStiff
        double ContactDamp
        double ContactV0
        double ContactFric
        double gravity

    void gait2dpi_dGRF(param_struct* par,
                   double q[NDOF],
                   double qd[NDOF],
                   double qdd[NDOF],
                   double Vsurfaces[2],
                   double QQ[NDOF],
                   double dQQdq[NDOF*NDOF],
                   double dQQdqd[NDOF*NDOF],
                   double dQQdqdd[NDOF*NDOF],
                   double GRF[6],
                   double dGRFdq[6*NDOF],
                   double dGRFdqd[6*NDOF],
                   double stick[NSTICK*2])


def evaluate_autolev_rhs(np.ndarray[np.double_t, ndim=1, mode='c'] generalized_coordinates,
                         np.ndarray[np.double_t, ndim=1, mode='c'] generalized_speeds,
                         np.ndarray[np.double_t, ndim=1, mode='c'] generalized_accelerations,
                         np.ndarray[np.double_t, ndim=1, mode='c'] velocity_surface,
                         constants):
    """This function takes the current values of the coordinates, speeds,
    ,accelerations, beltvelocity, and joints' moments and returns dynamic
	euqations residule, derrivatives i.e. dQQdp, dQQdpd, dQQdpdd, GRF, dGRFdq, dGRFdqd, and stick coordinates.

    [QQ, dQQdp, dQQdpd, dQQdpdd, GRF, dGRFdq, dGRFdqd, stick, tmp] = f(q, q', q'',vel_surf, mom)

    Parameters
    ----------
    generalized_coordinates : ndarray of floats, shape(9,)
        Trunk translation (x, y) and the joint angles.
    generalized_speeds : ndarray of floats, shape(9,)
        Trunk translation rate and the joint rates.
    generalized_accelerations : ndarray of floats, shape(9,)
        Trunk translation acceleration and the joint acceleration.
    velocity_surface : ndarray of floats, shape(2,)
        The velocity of surface perturbation, contains left and right
    constants : dictionary
        A dictionary of floats with keys that match the Autolev constants'
        names.

    Returns
    -------
    specified_quantities : ndarray, shape(9,)
        Required trunk horizontal and vertical force and the joint torques.
    dQQ_dq: ndarray, shape(9*9)
        The derivative of QQ to q.
    dQQ_dqd: ndarray, shape(9*9)
        The derivative of QQ to qd.
    dQQ_dqdd: ndarray, shape(9*9)
        The derivative of QQ to qdd.
    ground_reaction_forces : ndarray, shape(4,)
        The right and left ground reaction forces.
    dGRFdq : ndarray, shape(4*9)
	The derivative of GRF to q.
    dGRFdqd : ndarray, shape(4*9)
	The derivative of GRF to qd.
    stick_figure_coordinates : ndarray, shape(10*2)
        The x and y coordinates of the important points.
	tmp : ndarray, shape(16,)
		Heel/Toe velocities and forces. Right->Left

    Notes
    -----

    Generalized Coordinates

    q1: x hip translation wrt ground
    q2: y hip translation wrt ground
    q3: trunk z rotation wrt ground
    q4: right thigh z rotation wrt trunk
    q5: right shank z rotation wrt right thigh
    q6: right foot z rotation wrt right shank
    q7: left thigh z rotation wrt trunk
    q8: left shank z rotation wrt left thigh
    q9: left foot z rotation wrt left shank

    Surface velocity
      surface velocity under right foot
 
    Specified outputs

    t1: x force applied to trunk mass center
    t2: y force applied to trunk mass center
    t3: torque between ground and trunk
    t4: residuletorque between right thigh and trunk
    t5: residuletorque between right thigh and right shank
    t6: residuletorque between right foot and right shank
    t7: residuletorque between left thigh and trunk
    t8: residuletorque between left thigh and left shank
    t9: residuletorque between left foot and left shank

    GRFs

    grf1: right horizontal
    grf2: right vertical
    grf3: left horizontal
    grf4: left vertical

    Sticks:

    stick0: trunkCM position in x corrodinate
    stick1: trunkCM position in y corrodinate
    stick2: hip position in x corrodinate
    stick3: hip position in y corrodinate
    stick4: Rknee position in x corrodinate
    stick5: Rknee position in y corrodinate
    stick6: Rankle position in x corrodinate
    stick7: Rankle position in y corrodinate
    stick8: Rheel position in x corrodinate
    stick9: Rheel position in y corrodinate
    stick10: Rtoe position in x corrodinate
    stick11: Rtoe position in y corrodinate
    stick12: Lknee position in x corrodinate
    stick13: Lknee position in y corrodinate
    stick14: Lankle position in x corrodinate
    stick15: Lankle position in y corrodinate
    stick16: Lheel position in x corrodinate
    stick17: Lheel position in y corrodinate
    stick18: Ltoe position in x corrodinate
    stick19: Ltoe position in y corrodinate

    """

    cdef param_struct p = param_struct(
        TrunkMass=constants['TrunkMass'],
        TrunkInertia=constants['TrunkInertia'],
        TrunkCMy=constants['TrunkCMy'],
        ThighMass=constants['ThighMass'],
        ThighInertia=constants['ThighInertia'],
        ThighCMy=constants['ThighCMy'],
        ThighLen=constants['ThighLen'],
        ShankMass=constants['ShankMass'],
        ShankInertia=constants['ShankInertia'],
        ShankCMy=constants['ShankCMy'],
        ShankLen=constants['ShankLen'],
        FootMass=constants['FootMass'],
        FootInertia=constants['FootInertia'],
        FootCMx=constants['FootCMx'],
        FootCMy=constants['FootCMy'],
        ContactY=constants['ContactY'],
        ContactHeelX=constants['ContactHeelX'],
        ContactToeX=constants['ContactToeX'],
        ContactStiff=constants['ContactStiff'],
        ContactDamp=constants['ContactDamp'],
        ContactV0=constants['ContactV0'],
        ContactFric=constants['ContactFric'],
		gravity=constants['gravity'])

    # TODO: Should allow the option to pass these in, instead of creating a
    # new array on each call to this function. It would be faster.
    cdef np.ndarray[np.double_t, ndim=1, mode='c'] specified_quantities = np.zeros(9)
    cdef np.ndarray[np.double_t, ndim=1, mode='c'] jac_dQQ_dq = np.zeros(9*9)
    cdef np.ndarray[np.double_t, ndim=1, mode='c'] jac_dQQ_dqd = np.zeros(9*9)
    cdef np.ndarray[np.double_t, ndim=1, mode='c'] jac_dQQ_dqdd = np.zeros(9*9)
    cdef np.ndarray[np.double_t, ndim=1, mode='c'] ground_reaction_forces = np.zeros(6)
    cdef np.ndarray[np.double_t, ndim=1, mode='c'] jac_dGRF_dq = np.zeros(6*9)
    cdef np.ndarray[np.double_t, ndim=1, mode='c'] jac_dGRF_dqd = np.zeros(6*9)
    cdef np.ndarray[np.double_t, ndim=1, mode='c'] stick_figure_coordinates = np.zeros(2*10)

    gait2dpi_dGRF(&p,
              <double*> generalized_coordinates.data,
              <double*> generalized_speeds.data,
              <double*> generalized_accelerations.data,
              <double*> velocity_surface.data,
              <double*> specified_quantities.data,
              <double*> jac_dQQ_dq.data,
              <double*> jac_dQQ_dqd.data,
              <double*> jac_dQQ_dqdd.data,
              <double*> ground_reaction_forces.data,
              <double*> jac_dGRF_dq.data,
              <double*> jac_dGRF_dqd.data,
              <double*> stick_figure_coordinates.data)

    return specified_quantities, jac_dQQ_dq, jac_dQQ_dqd, jac_dQQ_dqdd, ground_reaction_forces, jac_dGRF_dq, jac_dGRF_dqd, stick_figure_coordinates
