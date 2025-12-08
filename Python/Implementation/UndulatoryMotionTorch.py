import torch
import math

device = torch.device("cpu")
class UndulatoryMotionTorch:
    def __init__(self, device='cpu'):
        self.device = device
        self.ANGLE_MOTOR1  = torch.tensor([3, 12, 11, 4, -3, -12, -11, -4],  dtype=torch.float32, device=self.device)*math.pi/180
        self.ANGLE_MOTOR2  = torch.tensor([2, -5, -5, -10, -2, 5, 5, 10],    dtype=torch.float32, device=self.device)*math.pi/180
        self.ANGLE_MOTOR3  = torch.tensor([0, -5, -6, -5, 0, 5, 6, 5],       dtype=torch.float32, device=self.device)*math.pi/180
        self.ANGLE_MOTOR4  = torch.tensor([-1, -5, -10, -3, 1, 5, 10, 3],    dtype=torch.float32, device=self.device)*math.pi/180
        self.ANGLE_MOTOR5  = torch.tensor([-3, -5, -2, 2, 3, 5, 2, -2],      dtype=torch.float32, device=self.device)*math.pi/180
        self.ANGLE_MOTOR6  = torch.tensor([-3, -5, 0, 5, 3, 5, 0, -5],       dtype=torch.float32, device=self.device)*math.pi/180
        self.ANGLE_MOTOR7  = torch.tensor([-4, -1, 1, 6, 4, 1, -1, -6],      dtype=torch.float32, device=self.device)*math.pi/180
        self.ANGLE_MOTOR8  = torch.tensor([-5, 2, 5, 9, 5, -2, -5, -9],      dtype=torch.float32, device=self.device)*math.pi/180

    def step(self, step, Delta_r):
        Def_Angle = Delta_r

        Tail_Centre = 36*Def_Angle

        # Count represent number of step
        Step_Angle = torch.zeros(8, dtype=torch.float32, device=device)

        Step_Angle[0]  = (self.ANGLE_MOTOR1[step])+ Def_Angle
        Step_Angle[1]  = (self.ANGLE_MOTOR2[step])+ 2 * Def_Angle
        Step_Angle[2]  = (self.ANGLE_MOTOR3[step])+ 3 * Def_Angle
        Step_Angle[3]  = (self.ANGLE_MOTOR4[step])+ 4 * Def_Angle
        Step_Angle[4]  = (self.ANGLE_MOTOR5[step])+ 5 * Def_Angle
        Step_Angle[5]  = (self.ANGLE_MOTOR6[step])+ 6 * Def_Angle
        Step_Angle[6]  = (self.ANGLE_MOTOR7[step])+ 7 * Def_Angle
        Step_Angle[7]  = (self.ANGLE_MOTOR8[step])+ 8 * Def_Angle

        return Step_Angle,Tail_Centre

    def smooth_motor_trajectory(self,step_angle_current, step_angle_previous, tail_rate, time):
        slope = (step_angle_current - step_angle_previous) / tail_rate
        step_angle_smooth = slope * time + step_angle_previous
        return step_angle_smooth