// gait2dp_dyn.h
// This file defines the data structure that holds model parameters.
// It is needed to pass model parameters to the Autolev generated C code
// in gait2dp_dyn.c

#define NDOF 9		/* number of kinematic degrees of freedom */
#define NMOM 6		/* number of joint moments */
#define NSTICK 10	/* number of stick figure points */

typedef struct {
    double gravity;     // local gravitational acceleration

// Body segment parameters
	double TrunkMass, TrunkInertia, TrunkCMy;
	double ThighMass, ThighInertia, ThighCMy, ThighLen;
	double ShankMass, ShankInertia, ShankCMy, ShankLen;
	double FootMass, FootInertia, FootCMx, FootCMy;

// Parameters for the ground contact model
	double ContactY;
	double ContactHeelX, ContactToeX;
	double ContactStiff, ContactDamp, ContactY0, ContactV0, ContactFric;

} param_struct;

// function prototype for the Q(q,qd,qdd) function
void gait2dp_dyn(param_struct* par, double q[NDOF], double qd[NDOF], double qdd[NDOF],
		double Vsurface[2], double mom[NMOM], double QQ[NDOF], double dQQdq[NDOF][NDOF],
		double dQQdqd[NDOF][NDOF], double dQQdqdd[NDOF][NDOF], double dQQdmom[NDOF][NMOM],
		double GRF[4], double dGRFdq[4][NDOF], double dGRFdqd[4][NDOF],
		double stick[NSTICK][2], double tmp[16]);
