% gait2dp.al
%
% This is an input file for Autolev to generate dynamic equations of motion for the
% 7-segment, 9-dof model for 2D gait simulations.
% This version has surface horizontal velocity as a possible perturbation input
%
% Author: Ton van den Bogert
% Last revised: 06/01/2017

PAUSE 0
AUTOZ ON
OVERWRITE ALL

% segment reference frames:
% origin is at center of mass of segment, X anterior while neutral standing

% Define model parameters as variables, so they can be modified from Matlab
Variables par__TrunkMass, par__TrunkInertia, par__TrunkCMy
Variables par__ThighMass, par__ThighInertia, par__ThighCMy, par__ThighLen
Variables par__ShankMass, par__ShankInertia, par__ShankCMy, par__ShankLen
Variables par__FootMass, par__FootInertia, par__FootCMx, par__FootCMy
Variables par__ContactY, par__ContactHeelX, par__ContactToeX
Variables par__ContactStiff, par__ContactDamp, par__ContactY0, par__ContactV0, par__ContactFric
Variables par__gravity					% Local gravitational acceleration

% The surface velocity is a model input, can be time varying
Variables Vsurface

%--------------------------------------------------------------------
%	Ground reference frame, units
%--------------------------------------------------------------------
Newtonian   Ground
UnitSystem  kg,meter,sec

%---------------------------------------------------
% Generalized coordinates
%---------------------------------------------------
MotionVariables' q{9}''

%----------------------------------
% Define bodies and points
%----------------------------------
Bodies Trunk, RThigh, RShank, RFoot, LThigh, LShank, LFoot
Points Hip, RKnee, LKnee, RAnkle, LAnkle

%-----------------------------------
% Trunk
%-----------------------------------
% define kinematics
P_GroundO_Hip> = Vector(Ground, q1, q2, 0)		% trunk translation: x and y of hip
V_Hip_Ground> = Vector(Ground, q1', q2', 0)		% and velocity
A_Hip_Ground> = Vector(Ground, q1'', q2'', 0)	% and acceleration
Simprot(Ground, Trunk, 3, q3)					% trunk rotation about z axis

% define mass properties and CM
Mass     Trunk = par__TrunkMass
Inertia  Trunk, 0, 0, par__TrunkInertia
P_Hip_TrunkO> = par__TrunkCMy*Trunk2>

% define angular velocity, angular acceleration, velocity, acceleration
W_Trunk_Ground> = q3' * Ground3>					
ALF_Trunk_Ground> = q3'' * Ground3>					
V2pts(Ground, Trunk, Hip, TrunkO)
A2pts(Ground, Trunk, Hip, TrunkO)

%-----------------------------------
% RThigh
%-----------------------------------
% define kinematics (hinge joint, rotation q4 on z axis)
Simprot(Trunk, RThigh, 3, q4)

% define mass properties and CM
Mass     RThigh = par__ThighMass
Inertia  RThigh, 0, 0, par__ThighInertia
P_Hip_RThighO> = par__ThighCMy*RThigh2>

% define angular velocity, angular acceleration, velocity, acceleration
W_RThigh_Ground> = (q4' + q3') * Ground3>	
ALF_RThigh_Ground> = (q4'' + q3'') * Ground3>	
V2pts(Ground, RThigh, Hip, RThighO)			
A2pts(Ground, RThigh, Hip, RThighO)	

%-----------------------------------
% RShank
%-----------------------------------
% define kinematics (hinge joint, rotation q5 on z axis)
Simprot(RThigh, RShank, 3, q5)
P_Hip_RKnee> = -par__ThighLen*RThigh2>		% define the knee position
V2pts(Ground, RThigh, Hip, RKnee)			% its velocity is needed also
A2pts(Ground, RThigh, Hip, RKnee)			% and acceleration

% define mass properties and CM
Mass     RShank = par__ShankMass
Inertia  RShank, 0, 0, par__ShankInertia	
P_RKnee_RShankO> = par__ShankCMy*RShank2>

% define angular velocity, angular acceleration, velocity, acceleration
W_RShank_Ground> = (q5' + q4' + q3') * Ground3>		
ALF_RShank_Ground> = (q5'' + q4'' + q3'') * Ground3>		
V2pts(Ground, RShank, RKnee, RShankO)				
A2pts(Ground, RShank, RKnee, RShankO)	

%-----------------------------------
% RFoot
%-----------------------------------
% define kinematics (hinge joint, rotation q6 on z axis)
Simprot(RShank, RFoot, 3, q6)
P_RKnee_RAnkle> = -par__ShankLen*RShank2>;	% define the ankle position
V2pts(Ground, RShank, RKnee, RAnkle)					% its velocity is needed also
A2pts(Ground, RShank, RKnee, RAnkle)					% and acceleration

% define mass properties and CM
Mass     RFoot = par__FootMass
Inertia  RFoot, 0, 0, par__FootInertia
P_RAnkle_RFootO> = par__FootCMx*RFoot1> + par__FootCMy*RFoot2>

% define angular velocity, angular acceleration, velocity, acceleration
W_RFoot_Ground> = (q6' + q5' + q4' + q3') * Ground3>	% angular velocity
ALF_RFoot_Ground> = (q6'' + q5'' + q4'' + q3'') * Ground3>	% angular velocity
V2pts(Ground, RFoot, RAnkle, RFootO)		
A2pts(Ground, RFoot, RAnkle, RFootO)	

%-----------------------------------
% LThigh
%-----------------------------------
% define kinematics (hinge joint, rotation q7 on z axis)
Simprot(Trunk, LThigh, 3, q7)

% define mass properties and CM
Mass     LThigh = par__ThighMass
Inertia  LThigh, 0, 0, par__ThighInertia
P_Hip_LThighO> = par__ThighCMy*LThigh2>

% define angular velocity, angular acceleration, velocity, acceleration
W_LThigh_Ground> = (q7' + q3') * Ground3>	
ALF_LThigh_Ground> = (q7'' + q3'') * Ground3>	
V2pts(Ground, LThigh, Hip, LThighO);				
A2pts(Ground, LThigh, Hip, LThighO);		

%-------------------
% LShank
%-------------------
% define kinematics (hinge joint, rotation q8 on z axis)
Simprot(LThigh, LShank, 3, q8)
P_Hip_LKnee> = -par__ThighLen*LThigh2>;	% define the knee position
V2pts(Ground, LThigh, Hip, LKnee);						% its velocity is needed also
A2pts(Ground, LThigh, Hip, LKnee);						% and acceleration

% define mass properties and CM
Mass     LShank = par__ShankMass
Inertia  LShank, 0, 0, par__ShankInertia	
P_LKnee_LShankO> = par__ShankCMy*LShank2>

% define angular velocity, angular acceleration, velocity, acceleration
W_LShank_Ground> = (q8' + q7' + q3') * Ground3>		
ALF_LShank_Ground> = (q8'' + q7'' + q3'') * Ground3>		
V2pts(Ground, LShank, LKnee, LShankO);					
A2pts(Ground, LShank, LKnee, LShankO);		

%-----------------------------------
% LFoot
%-----------------------------------
% define kinematics (hinge joint, rotation q6 on z axis)
Simprot(LShank, Lfoot, 3, q9)
P_LKnee_LAnkle> = -par__ShankLen*LShank2>;		% define the ankle position
V2pts(Ground, LShank, LKnee, LAnkle);			% its velocity is needed also
A2pts(Ground, LShank, LKnee, LAnkle);			% and acceleration

% define mass properties and CM
Mass     LFoot = par__FootMass
Inertia  LFoot, 0, 0, par__FootInertia
P_LAnkle_LFootO> = par__FootCMx*LFoot1> + par__FootCMy*LFoot2>

% define angular velocity, angular acceleration, velocity, acceleration
W_LFoot_Ground> = (q9' + q8' + q7' + q3') * Ground3>	% angular velocity
ALF_LFoot_Ground> = (q9'' + q8'' + q7'' + q3'') * Ground3>	% angular velocity
V2pts(Ground, LFoot, LAnkle, LFootO);		
A2pts(Ground, LFoot, LAnkle, LFootO);	

%--------------------------------------------------------------------
% Apply gravity
%--------------------------------------------------------------------
Gravity( -par__gravity*Ground2> )			% gravity acts along -Y axis of Ground

%--------------------------------------------------------------------------
% Ground reaction forces on 4 points on the feet
%--------------------------------------------------------------------------
Points RHeel, RToe, LHeel, LToe
P_RAnkle_RHeel>  := par__ContactHeelX * RFoot1> + par__ContactY * RFoot2>;
P_RAnkle_RToe>   := par__ContactToeX  * RFoot1> + par__ContactY * RFoot2>;
P_LAnkle_LHeel>  := par__ContactHeelX * LFoot1> + par__ContactY * LFoot2>;
P_LAnkle_LToe>   := par__ContactToeX  * LFoot1> + par__ContactY * LFoot2>;
V2PTS(Ground, RFoot, RAnkle, RHeel); 
V2PTS(Ground, RFoot, RAnkle, RToe); 
V2PTS(Ground, LFoot, LAnkle, LHeel); 
V2PTS(Ground, LFoot, LAnkle, LToe); 
contact(RHeel) 
contact(RToe) 
contact(LHeel) 
contact(LToe) 

% Create a 6 x 1 matrix with GR force and moment of each foot, expressed in Ground XY frame
FR> = FRHeel> + FRToe>
MR> = cross(P_GroundO_RHeel>, FRHeel>) + cross(P_GroundO_RToe>, FRToe>);
FL> = FLHeel> + FLToe>
ML> = cross(P_GroundO_LHeel>, FLHeel>) + cross(P_GroundO_LToe>, FLToe>);
grf = [	dot(FR>,Ground1>)	; dot(FR>,Ground2>)	 ; dot(MR>,Ground3>)	;&
		dot(FL>,Ground1>)	; dot(FL>,Ground2>) ; dot(ML>,Ground3>)	];

%--------------------------------------------------------------------------------------------------------------
% Ground reaction forces on trunk CM and all joints (so model will not penetrate the ground even if it falls)
% (these are currently commented out to make the code as fast as possible)
%--------------------------------------------------------------------------------------------------------------
%contact(TrunkO) 
%contact(Hip) 
%contact(RKnee) 
%contact(RAnkle) 
%contact(LKnee) 
%contact(LAnkle) 

%---------------------------------------------------------------------------------------
% Produce output for a 10-point stick figure, a 2 x 10 matrix, containing x and y of 10 points:
% trunk, hip, rknee, rankle, rheel, rtoe, lknee, lankle, lheel, ltoe
%---------------------------------------------------------------------------------------

stick = [	dot(P_GroundO_TrunkO>, Ground1>) , dot(P_GroundO_TrunkO>, Ground2>) ; &
			dot(P_GroundO_Hip>,    Ground1>) , dot(P_GroundO_Hip>,    Ground2>) ; &
			dot(P_GroundO_RKnee>,  Ground1>) , dot(P_GroundO_RKnee>,  Ground2>) ; &
			dot(P_GroundO_RAnkle>, Ground1>) , dot(P_GroundO_RAnkle>, Ground2>) ; &
			dot(P_GroundO_RHeel>,  Ground1>) , dot(P_GroundO_RHeel>,  Ground2>) ; &
			dot(P_GroundO_RToe>,   Ground1>) , dot(P_GroundO_RToe>,   Ground2>) ; &
			dot(P_GroundO_LKnee>,  Ground1>) , dot(P_GroundO_LKnee>,  Ground2>) ; &
			dot(P_GroundO_LAnkle>, Ground1>) , dot(P_GroundO_LAnkle>, Ground2>) ; &
			dot(P_GroundO_LHeel>,  Ground1>) , dot(P_GroundO_LHeel>,  Ground2>) ; &
			dot(P_GroundO_LToe>,   Ground1>) , dot(P_GroundO_LToe>,   Ground2>) ]
%--------------------------------------------------------------------
% Solve the nine generalized forces Q
%--------------------------------------------------------------------
% variable name is QQ to avoid confusion with q (Autolev is not case-sensitive!)
QQ = -(Fr() + FrStar())
encode QQ

%-----------------------------------------------------------------------------------------------------
% Generate the jacobians dQQ/dq, dQQ/dqd, dQQ/dqdd
%-----------------------------------------------------------------------------------------------------
q = [q1, q2, q3, q4, q5, q6, q7, q8, q9]
qd = [q1', q2', q3', q4', q5', q6', q7', q8', q9']
qdd = [q1'', q2'', q3'', q4'', q5'', q6'', q7'', q8'', q9'']
dQQ_dq = D(QQ,q)
dQQ_dqd = D(QQ,qd)
dQQ_dqdd = D(QQ,qdd)
dgrf_dq = D(grf,q)
dgrf_dqd = D(grf,qd)

%------------------------------------
% generate C code
%------------------------------------
encode dQQ_dq, dQQ_dqd, dQQ_dqdd, grf, dgrf_dq, dgrf_dqd, stick
Code Algebraic() gait2dp_raw.c

EXIT
