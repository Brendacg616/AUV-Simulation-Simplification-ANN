import os, sys
pathname = os.path.dirname(sys.argv[0])        
print('path =', pathname)
# General imports
import numpy as np
import torch
import matplotlib.pyplot as plt

# Custom module imports
import UndulatoryMotionTorch
import AdditionalFncs

#Import Models
import TailKinematicsNN
import ThrustNN
from MotorModels.MotorModels import MotorModel, rk4_step_func
import Control.MotorControlNN as MotorControlNN
import RigidBody.RigidBodyNN as RigidBodyNN

# Underdulatory motion instance
undulatory_motion = UndulatoryMotionTorch.UndulatoryMotionTorch()

# Initialize and load NN models
# Create dictionaries to hold 8 motor models
motor_models = {}
for i in range(8):
    motor_models[i] = MotorModel()
    # motor_models[i].load_state_dict(torch.load(f"./MotorModels/model_retrained_motor_{i}.pth"))
    motor_models[i].load_state_dict(torch.load(pathname + "/MotorModels/node_ABmats_v2.pth"))
    #motor_models[i] = torch.compile(motor_models[i])
    motor_models[i].eval()
# Load motor scalers
motor_u_scaler = AdditionalFncs.torchScaler(pathname + "/MotorModels/Scalers/u_scaler.pkl")
motor_x_scaler = AdditionalFncs.torchScaler(pathname + "/MotorModels/Scalers/x_scaler.pkl")

# Control model
motor_control_model = MotorControlNN.MotorControlModelv2()
# motor_control_model.load_state_dict(torch.load(pathname + "/Control/motorControlModel.pth"))
motor_control_model.load_state_dict(torch.load(pathname + "/Control/control_model_v2_retrained2.pth"))
# motor_control_model = torch.compile(motor_control_model)
motor_control_model.eval()
motor_control_input_scaler = AdditionalFncs.torchScaler(pathname + "/Control/input_scaler.pkl")
motor_control_output_scaler = AdditionalFncs.torchScaler(pathname + "/Control/output_scaler.pkl")

# Tail Kinematics Model
tail_kinematics_model = TailKinematicsNN.TailKinematicsRNN(num_layers=1)
tail_kinematics_model.load_state_dict(torch.load(pathname + "/TailKinematics/kinematicsMoments_radians.pt"))
#tail_kinematics_model = torch.compile(tail_kinematics_model)
tail_kinematics_model.eval()

# Thrust Model
thrust_model = ThrustNN.thrustFlexNN(input_size=16, hidden_size=32, output_size=7, hidden_layers=5)
thrust_model.load_state_dict(torch.load(pathname + "/Thrust/simple_data_modelv2_32_neurons_5_layers.pt"))
#thrust_model = torch.compile(thrust_model)
thrust_model.eval()
input_scaler_thrust = AdditionalFncs.torchScaler(pathname + "/Thrust/scaler_thrust_in.pkl")
output_scaler_thrust = AdditionalFncs.torchScaler(pathname + "/Thrust/scaler_thrust_out.pkl")

# Rigid Body Model
rigid_body_model = RigidBodyNN.RigidBodyNN(u_size=8, x_size=12, n_neurons=64)
rigid_body_model.load_state_dict(torch.load(pathname + "/RigidBody/rigid_body_nn_model_clean_input_1.pth"))
#rigid_body_model = torch.compile(rigid_body_model)
rigid_body_model.eval()
rigid_body_x_scaler = AdditionalFncs.torchScaler(pathname + "/RigidBody/x_scaler_clean.pkl")
rigid_body_u_scaler = AdditionalFncs.torchScaler(pathname + "/RigidBody/u_scaler_clean.pkl")


# Main simulation loop
motors_dt = 1e-3
N = 2000 # Number of simulation steps - Simulation time = N*dt
Delta_r = 0.0
tail_frequency = 1.0  # Hz

# Undulatory motion parameters
step_update = (1/motors_dt) * tail_frequency / 8
undulatory_step = 0
step_angle_history = torch.zeros((N, 8), dtype=torch.float32)
tail_centreline_history = torch.zeros((N, 1), dtype=torch.float32)
motor_time = 0.0
previous_step_angle = torch.zeros(8, dtype=torch.float32)
smooth_angle_history = torch.zeros((N, 8), dtype=torch.float32)

#Motor parameters and variables
gear_ratio = 298
max_u_signal = 12.0  # Max control signal
min_u_signal = -12.0 # Min control signal
motor_states = torch.zeros((8, 3), dtype=torch.float32)  # [position, velocity, current] for each motor
motor_states_history = torch.zeros((N, 8, 3), dtype=torch.float32)  # History of motor states

# #Control parameters and variables
errors = torch.zeros((8, 3), dtype=torch.float32) # Accumulated error, Previous error, Current error
motor_u_signals = torch.zeros((8, 1), dtype=torch.float32) # Control signals for each motor
motor_u_signals_history = torch.zeros((N, 8, 1), dtype=torch.float32) # History of control signals

# Tail Kinematics and Thrust variables
tail_kinematics_history = torch.zeros((N, 3), dtype=torch.float32)
tail_data = torch.zeros((9), dtype=torch.float32)  # 8 motors + tail centreline
old_caudal_amplitude = 0.0
# Thrust variables
thrust_input = torch.zeros((1,16), dtype=torch.float32)  # 16 itorchuts for thrust model
thrust_output = torch.zeros((1,7), dtype=torch.float32)  # 7 thrust outputs
thrust_history = torch.zeros((N, 7), dtype=torch.float32)  # 7 thrust outputs

# Rigid Body variables
rigid_body_states = torch.zeros((12), dtype=torch.float32)  # Placeholder for rigid body states
rigid_body_states[8] = -5.0  # Initial depth
rigid_body_u_signals = torch.zeros((8), dtype=torch.float32)  # Placeholder for rigid body control itorchuts
rigid_body_states_history = torch.zeros((N, 1, 12), dtype=torch.float32)  # History of rigid body states
rigid_body_dt = 5e-3

# Get simulation running time
import time
start_time = time.time()
with torch.no_grad():
# Simulation loop
    for i in range(N):
        # Get undulatory step angles
        [step_angle, tail_centreline] = undulatory_motion.step(undulatory_step, Delta_r)
        # Smooth motor target trajectory
        step_angle_smooth = undulatory_motion.smooth_motor_trajectory(step_angle, previous_step_angle, step_update * motors_dt, motor_time) * gear_ratio
        # Get motor angular position errors
        errors[:, 0] += errors[:, 2] # Accumulated error
        errors[:, 1] = errors[:, 2]  # Previous error
        errors[:, 2] = step_angle_smooth - motor_states[:, 1]*gear_ratio  # Position error

        # Scale errors for NN input
        errors = motor_control_input_scaler.transform(errors)
        motor_control_output = motor_control_model(errors)
        u_signals = torch.clip(motor_control_output_scaler.inverse_transform(motor_control_output), min_u_signal, max_u_signal)
        # Store real control signals before scaling
        motor_u_signals_history[i] = u_signals
        # Integrate motor models
        motor_states = motor_x_scaler.transform(motor_states)
        u_signals =  motor_u_scaler.transform(u_signals)
        for j in range(8):
            motor_states[j] = rk4_step_func(motor_models[j], motor_states[j], u_signals[j], motors_dt)

        # Inverse scale motor states
        motor_states = motor_x_scaler.inverse_transform(motor_states)

        # Motor error reset 
        motor_time += motors_dt
        if (i % step_update) == 0:
            motor_time = 0.0
            previous_step_angle = step_angle
            undulatory_step += 1
            errors.zero_()  # Reset errors
            if undulatory_step >= 8:
                undulatory_step = 0
        
        # Replicate multirate simulation
        if (i % 5) == 0:
            # Tail Kinematics and Thrust estimation (8 motors & Tail centreline)
            tail_data[:8] =motor_states[:, 1] 
            tail_data[8] = tail_centreline
            tail_kinematics_output = tail_kinematics_model(tail_data.view(1, -1, 1)).view(-1)

            # Thrust estimation
            thrust_input[0,:12] = rigid_body_states  # Rigid body states
            thrust_input[0,12:14] = tail_kinematics_output[:2]  # Tail kinematics outputs
            thrust_input[0,14] = old_caudal_amplitude  # Motor angular velocities
            thrust_input[0,15] = tail_centreline  # Tail centreline position

            # Scale thrust inputs
            thrust_input  =  input_scaler_thrust.transform(thrust_input)
            thrust_output = thrust_model(thrust_input)
            old_caudal_amplitude = tail_kinematics_output[1]
            # Scale thrust outputs
            thrust_output = output_scaler_thrust.inverse_transform(thrust_output)

            # Rigid Body state update
            rigid_body_u_signals[:7] = thrust_output[0]  # Thrust forces and moments
            rigid_body_u_signals[7] = tail_kinematics_output[0]  # Moments from tail
            # Scale rigid body inputs
            rigid_body_states = rigid_body_x_scaler.transform(rigid_body_states)
            rigid_body_u_signals = rigid_body_u_scaler.transform(rigid_body_u_signals) 
            rigid_body_states = rk4_step_func(rigid_body_model, rigid_body_states, rigid_body_u_signals, rigid_body_dt)
            # Inverse scale rigid body states
            rigid_body_states = rigid_body_x_scaler.inverse_transform(rigid_body_states)

        # Store data
        step_angle_history[i] = step_angle
        tail_centreline_history[i, 0] = tail_centreline
        smooth_angle_history[i] = step_angle_smooth
        motor_states_history[i] = motor_states
        tail_kinematics_history[i] = tail_kinematics_output
        thrust_history[i] = thrust_output
        rigid_body_states_history[i] = rigid_body_states.reshape(1, -1)

end_time = time.time()
print(f"Simulation Time for {N} steps: {end_time - start_time} seconds")

# Save simulation data in csv files
np.savetxt(pathname +"/OutputData/step_angle_history.csv", step_angle_history.numpy(), delimiter=",")
np.savetxt(pathname +"/OutputData/smooth_angle_history.csv", smooth_angle_history.numpy(), delimiter=",")
for j in range(8):
    np.savetxt(pathname +f"/OutputData/motor_u_signals_history_motor_{j+1}.csv", motor_u_signals_history[:, j].numpy(), delimiter=",")
    np.savetxt(pathname +f"/OutputData/motor_states_history_motor_{j+1}.csv", motor_states_history[:, j].numpy().reshape(N, -1), delimiter=",")
np.savetxt(pathname +"/OutputData/tail_kinematics_history.csv", tail_kinematics_history.numpy(), delimiter=",")
np.savetxt(pathname +"/OutputData/thrust_history.csv", thrust_history.numpy(), delimiter=",")
np.savetxt(pathname +"/OutputData/rigid_body_states_history.csv", rigid_body_states_history.numpy().reshape(N, -1), delimiter=",")


# Plot target motor positions results
time = np.arange(0, N*motors_dt, motors_dt)
plt.figure()
for j in range(8):
    plt.subplot(8, 1, j+1)
    #plt.plot(time, step_angle_history[:, j], label=f'Motor {j+1}')
    plt.plot(time, smooth_angle_history[:, j], label=f'Smooth Motor {j+1}', linestyle='--')
    plt.plot(time, motor_states_history[:, j, 1]*298, label=f'Motor {j+1}')
    plt.xlabel('Time (s)')
    plt.ylabel('Step Angle (rad)')
    plt.legend()

# Plot control signals results
plt.figure()
for j in range(8):
    plt.subplot(8, 1, j+1)
    plt.plot(time, motor_u_signals_history[:, j], label=f'Motor {j+1}')
    plt.xlabel('Time (s)')
    plt.ylabel('Control Signal (N)')
    plt.legend()


# Plot tail kinematics results
plt.figure()
plt.subplot(3,1,1)
plt.plot(time, tail_kinematics_history[:, 0], label='Tail Moment')
plt.xlabel('Time (s)')
plt.ylabel('Tail Moment (N*m)')
plt.legend()
plt.subplot(3,1,2)
plt.plot(time, tail_kinematics_history[:, 1], label='Caudal Amplitude')
plt.xlabel('Time (s)')
plt.ylabel('Tail Amplitude (m)')
plt.legend()
plt.subplot(3,1,3)
plt.plot(time, tail_kinematics_history[:, 2], label='Caudal Angle')
plt.xlabel('Time (s)')
plt.ylabel('Tail Angle (rad)')
plt.legend()


# Plot thrust results
plt.figure()
for j in range(7):
    plt.subplot(4, 2, j+1)
    plt.plot(time, thrust_history[:, j], label=f'Thrust {j+1}')
    plt.xlabel('Time (s)')
    plt.ylabel(f'Thrust {j+1} Value')
    plt.legend()

# Plot rigid body states results
plt.figure()
for j in range(12):
    plt.subplot(3, 4, j+1)
    plt.plot(time, rigid_body_states_history[:, 0, j], label=f'State {j+1}')
    plt.xlabel('Time (s)')
    plt.ylabel(f'State {j+1} Value')
    plt.legend()

# # Show all plots
plt.show()