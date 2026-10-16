# HEAL Task Workspace and Table Scenario

This is a configurable analysis scenario, not a table/object specification found in the HEAL files.

## Assumptions
- Top-down grasp; 16 world-Z yaw samples over 360 degrees.
- Cube side: 0.040 m.
- Square tray opening: 0.160 m; wall height 0.050 m; wall thickness 0.010 m.
- Vertical approach/withdrawal clearance: 0.120 m.
- XY grid spacing: 0.100 m; IK position tolerance 0.003 m; orientation tolerance 0.030 rad.
- The analyzed H/L/W and tray center are held at the user's provisional values; they are not optimized in this diagnosis.
- The cube is represented by a geometric proxy during carry/placement checks; it does not participate in MuJoCo dynamics or finger friction.
- Self-collision checks ignore penetrations smaller than 0.1 mm to avoid classifying pad seam contact as a collision.
- These are kinematic/geometric path checks, not actuator torque or dynamic grasp validation.

## Diagnosed model issue
The previous analyzer used left_pad/right_pad body origins as the grasp center. In this MJCF the physical pad collision boxes are offset from those origins. The corrected TCP is the mean of the four actual pad-box geom centers, and its Jacobian is evaluated at those points.
The prior calibration also reused geom IDs from the bare model in a separately compiled table/tray model. Scene insertion changes geom IDs. The corrected code carries geom/joint names and resolves IDs per compiled model.
Together, the 129.4 mm old TCP offset and stale scene geom indices explain why the previous aggregate 0/48 path result did not describe the intended fingertip TCP in the task model.

## Fixed provisional table
- H/L/W and tray center are unchanged from the requested diagnosis values.
- This does not globally validate every point on the tabletop; only listed diagnostic cases are tested.
- Actual base mesh projection at the world origin: X=[-0.100, 0.100] m, Y=[-0.100, 0.100] m; search keeps at least 0.050 m between the base projection and tabletop edge.
- H = 0.300 m above the model floor/base plane.
- X = [0.150, 0.350] m; L = 0.200 m.
- Y = [-0.450, -0.050] m; W = 0.400 m.
- Tray center = (0.240, -0.160) m in world XY.

## Episode diagnostics
PASS requires IK through approach and withdrawal, robot/table/tray and significant self-collision checks, carried-cube transfer/descent clearance, static tray placement, and release retreat.
- success_pick_yaw_0: target=(0.320, -0.380) m, yaw=0.0 deg; success=True; reason=PASS_TRANSFER_PLACE_RELEASE_RETREAT at complete; pick=PASS_IK_APPROACH_WITHDRAWAL at complete; pick residuals=(0.0011 m, 0.0202 rad); joint margin=2.2694 rad; minimum robot/environment distance=0.0000 m; obstacle pair=('left_coupler/unnamed_geom', 'analysis_table_top'); minimum cube distance=0.0000 m.
- success_pick_yaw_90: target=(0.320, -0.380) m, yaw=90.0 deg; success=True; reason=PASS_TRANSFER_PLACE_RELEASE_RETREAT at complete; pick=PASS_IK_APPROACH_WITHDRAWAL at complete; pick residuals=(0.0011 m, 0.0217 rad); joint margin=1.7846 rad; minimum robot/environment distance=0.0000 m; obstacle pair=('left_driver/unnamed_geom', 'analysis_tray_wall_px'); minimum cube distance=0.0000 m.
- tray_edge_collision_yaw_45: target=(0.190, -0.270) m, yaw=45.0 deg; success=False; reason=TABLE_TRAY_COLLISION at approach_5/5; pick=TABLE_TRAY_COLLISION at approach_5/5; pick residuals=(0.0019 m, 0.0257 rad); joint margin=2.1799 rad; minimum robot/environment distance=-0.0054 m; obstacle pair=('right_follower/unnamed_geom', 'analysis_tray_wall_ny'); minimum cube distance=nan m.
- joint_limit_yaw_202_5: target=(0.190, -0.380) m, yaw=202.5 deg; success=False; reason=IK_JOINT_LIMIT at approach_5/5; pick=IK_JOINT_LIMIT at approach_5/5; pick residuals=(0.0083 m, 0.0469 rad); joint margin=0.0000 rad; minimum robot/environment distance=0.0000 m; obstacle pair=('end_effector/end_effector_collision', 'analysis_table_top'); minimum cube distance=nan m.

## Recheck of historical 48 target/yaw paths
- Replayed 3 historical locations × 16 yaws = 48 corrected-TCP approach/withdrawal tests.
- Outcome counts: 46 pass; failures by reason: {'PASS_IK_APPROACH_WITHDRAWAL': 46, 'IK_JOINT_LIMIT': 2}.
- These 48 are vertical target paths only; they are not counted as full pick-and-place successes. Full transfer/place/release is checked only for the named episode cases above.
- Failure: front_left at yaw 202.5 deg, IK_JOINT_LIMIT during approach_5/5; residual=(0.0056 m, 0.0327 rad), joint margin=0.0000 rad.
- Failure: front_left at yaw 225.0 deg, IK_JOINT_LIMIT during approach_5/5; residual=(0.0055 m, 0.0327 rad), joint margin=0.0000 rad.

## Height sweep
H (m) | any-yaw grid points | all-yaw grid points | any-yaw area (m²) | all-yaw rectangle area (m²)
---:|---:|---:|---:|---:
0.100 | 30 | 10 | 0.300 | 0.030
0.200 | 41 | 10 | 0.410 | 0.080
0.300 | 32 | 9 | 0.320 | 0.060
0.400 | 0 | 0 | 0.000 | 0.000
0.500 | 0 | 0 | 0.000 | 0.000

## Interpretation
Position workspace maps are computed from the six-joint arm and corrected pad-geom TCP. Dexterous cells mean IK converged at grasp and pre-grasp for every tested discrete yaw; intermediate yaws are not certified. Cells with no sampled IK solution are unknown, not proven unreachable. Episode success labels apply only to the explicitly tested target/yaw cases.
A reported 0.0000 m minimum clearance is tangency/support contact within the distance routine's precision; penetrations below -0.00001 m are rejected. Cube/table-floor contact at placement is intentional. The cube proxy is geometric only and does not validate dynamic grasp forces or friction.
