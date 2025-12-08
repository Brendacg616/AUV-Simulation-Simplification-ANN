from numpy import real
import numpy as np
import math

# Motor Number = [Step 1 Step 2 Step 3 Step 4 Step 5 Step 6 Step 7 Step 8]
ANGLE_MOTOR1  = np.array([3, 12, 11, 4, -3, -12, -11, -4])*math.pi/180
ANGLE_MOTOR2  = np.array([2, -5, -5, -10, -2, 5, 5, 10])*math.pi/180
ANGLE_MOTOR3  = np.array([0, -5, -6, -5, 0, 5, 6, 5])*math.pi/180
ANGLE_MOTOR4  = np.array([-1, -5, -10, -3, 1, 5, 10, 3])*math.pi/180
ANGLE_MOTOR5  = np.array([-3, -5, -2, 2, 3, 5, 2, -2])*math.pi/180
ANGLE_MOTOR6  = np.array([-3, -5, 0, 5, 3, 5, 0, -5])*math.pi/180
ANGLE_MOTOR7  = np.array([-4, -1, 1, 6, 4, 1, -1, -6])*math.pi/180
ANGLE_MOTOR8  = np.array([-5, 2, 5, 9, 5, -2, -5, -9])*math.pi/180

def step(step,Delta_r):
    Def_Angle = real(Delta_r)

    Tail_Centre = 36*Def_Angle

    # Count represent number of step
    Step_Angle = np.zeros(8)

    Step_Angle[0]  = (ANGLE_MOTOR1[step])+ Def_Angle
    Step_Angle[1]  = (ANGLE_MOTOR2[step])+ 2 * Def_Angle
    Step_Angle[2]  = (ANGLE_MOTOR3[step])+ 3 * Def_Angle
    Step_Angle[3]  = (ANGLE_MOTOR4[step])+ 4 * Def_Angle
    Step_Angle[4]  = (ANGLE_MOTOR5[step])+ 5 * Def_Angle
    Step_Angle[5]  = (ANGLE_MOTOR6[step])+ 6 * Def_Angle
    Step_Angle[6]  = (ANGLE_MOTOR7[step])+ 7 * Def_Angle
    Step_Angle[7]  = (ANGLE_MOTOR8[step])+ 8 * Def_Angle

    
    return Step_Angle,Tail_Centre

def smooth_motor_trajectory(step_angle_current, step_angle_previous, tail_rate, time):
    slope = (step_angle_current - step_angle_previous) / tail_rate
    step_angle_smooth = slope * time + step_angle_previous
    return step_angle_smooth