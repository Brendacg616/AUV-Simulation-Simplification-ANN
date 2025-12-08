import os, sys
pathname = os.path.dirname(sys.argv[0])      

# General imports
import numpy as np
import torch
import matplotlib.pyplot as plt
import joblib

# Custom module imports
import UndulatoryMotion

#Import Models
import TailKinematicsNN
import ThrustNN
from MotorModels.MotorModels import MotorModel, rk4_step_func
import Control.MotorControlNN as MotorControlNN
import RigidBody.RigidBodyNN as RigidBodyNN

# Initialize variables
u_signals = np.zeros((1, 1000, 1), dtype=np.float32)  # Placeholder for control signals
x_motor = np.zeros((1, 1000, 3), dtype=np.float32)     # Placeholder for motor states

# Initialize and load NN models
# Create dictionaries to hold 8 motor models
motor_models = {}
for i in range(8):
    motor_models[i] = MotorModel()
    # motor_models[i].load_state_dict(torch.load(f"./MotorModels/model_retrained_motor_{i}.pth"))
    motor_models[i].load_state_dict(torch.load(pathname + "/MotorModels/node_ABmats_v2.pth"))
# Load motor scalers
motor_u_scaler = joblib.load(pathname + "/MotorModels/Scalers/u_scaler.pkl")
motor_x_scaler = joblib.load(pathname + "/MotorModels/Scalers/x_scaler.pkl")
# Control model
motor_control_model = MotorControlNN.MotorControlModelv2()
motor_control_model.load_state_dict(torch.load(pathname + "/Control/motorControlModel.pth"))
motor_control_input_scaler = joblib.load(pathname + "/Control/input_scaler.pkl")
motor_control_output_scaler = joblib.load(pathname + "/Control/output_scaler.pkl")

# Tail Kinematics Model
tail_kinematics_model = TailKinematicsNN.TailKinematicsRNN(num_layers=1)
tail_kinematics_model.load_state_dict(torch.load(pathname + "/TailKinematics/kinematicsMoments_radians.pt"))

# Thrust Model
thrust_model = ThrustNN.thrustFlexNN(input_size=16, hidden_size=32, output_size=7, hidden_layers=5)
thrust_model.load_state_dict(torch.load(pathname + "/Thrust/simple_data_modelv2_32_neurons_5_layers.pt"))
input_scaler_thrust = joblib.load(pathname + "/Thrust/scaler_thrust_in.pkl")
output_scaler_thrust = joblib.load(pathname + "/Thrust/scaler_thrust_out.pkl")

# Rigid Body Model
rigid_body_model = RigidBodyNN.RigidBodyNN(u_size=8, x_size=12, n_neurons=64)
rigid_body_model.load_state_dict(torch.load(pathname + "/RigidBody/rigid_body_nn_model_clean_input.pth"))
rigid_body_x_scaler = joblib.load(pathname + "/RigidBody/x_scaler_clean.pkl")
rigid_body_u_scaler = joblib.load(pathname + "/RigidBody/u_scaler_clean.pkl")


# Main simulation loop
motors_dt = 1e-3
N = 100 # Number of simulation steps - Simulation time = N*dt
Delta_r = 0.0
tail_frequency = 1.0  # Hz

# Undulatory motion parameters
step_update = (1/motors_dt) * tail_frequency / 8
undulatory_step = 0
step_angle_history = np.zeros((N, 8), dtype=np.float32)
tail_centreline_history = np.zeros((N, 1), dtype=np.float32)
motor_time = 0.0
previous_step_angle = np.zeros(8, dtype=np.float32)
smooth_angle_history = np.zeros((N, 8), dtype=np.float32)

#Motor parameters and variables
gear_ratio = 298
max_u_signal = 12.0  # Max control signal
min_u_signal = -12.0 # Min control signal
motor_states = np.zeros((8, 3), dtype=np.float32)  # [position, velocity, current] for each motor
motor_states_history = np.zeros((N, 8, 3), dtype=np.float32)  # History of motor states

#Control parameters and variables
errors = np.zeros((8, 3), dtype=np.float32) # Accumulated error, Previous error, Current error
u_signals = np.zeros((8, 1), dtype=np.float32) # Control signals for each motor
u_signals_history = np.zeros((N, 8, 1), dtype=np.float32) # History of control signals

# Tail Kinematics and Thrust variables
tail_kinematics_history = np.zeros((N, 3), dtype=np.float32)
tail_data = np.zeros((9), dtype=np.float32)  # 8 motors + tail centreline
old_caudal_amplitude = 0.0
# Thrust variables
thrust_input = np.zeros((1,16), dtype=np.float32)  # 16 inputs for thrust model
thrust_data = np.zeros((1,7), dtype=np.float32)  # 7 thrust outputs
thrust_history = np.zeros((N, 7), dtype=np.float32)  # 7 thrust outputs

# Rigid Body variables
rigid_body_states = np.zeros((12), dtype=np.float32)  # Placeholder for rigid body states
rigid_body_u_signals = np.zeros((8), dtype=np.float32)  # Placeholder for rigid body control inputs
rigid_body_states_history = np.zeros((N, 1, 12), dtype=np.float32)  # History of rigid body states
rigid_body_dt = 5e-3
#Get simulation running time
import time
start_time = time.time()
# Simulation loop
for i in range(N):
    # Get undulatory step angles
    [step_angle, tail_centreline] = UndulatoryMotion.step(undulatory_step, Delta_r)
    # Smooth motor target trajectory
    step_angle_smooth = UndulatoryMotion.smooth_motor_trajectory(step_angle, previous_step_angle, step_update * motors_dt, motor_time) * gear_ratio
    # Get motor angular position errors
    errors[:, 0] += errors[:, 2] # Accumulated error
    errors[:, 1] = errors[:, 2]  # Previous error
    errors[:, 2] = step_angle_smooth - motor_states[:, 1]*298  # Position error

    # Scale errors for NN input
    scaled_errors = motor_control_input_scaler.transform(errors)
    motor_control_output = motor_control_model(torch.from_numpy(scaled_errors).float())
    u_signals = np.clip(motor_control_output_scaler.inverse_transform(motor_control_output.detach().numpy()), min_u_signal, max_u_signal)

    # Integrate motor models
    motor_states_scaled = torch.from_numpy(motor_x_scaler.transform(motor_states))
    u_signals_scaled = torch.from_numpy(motor_u_scaler.transform(u_signals))
    for j in range(8):
        motor_states[j] = rk4_step_func(motor_models[j], motor_states_scaled[j], u_signals_scaled[j], motors_dt).detach().numpy()

    # Inverse scale motor states
    motor_states = motor_x_scaler.inverse_transform(motor_states)

    # Motor error reset 
    motor_time += motors_dt
    if (i % step_update) == 0:
        motor_time = 0.0
        previous_step_angle = step_angle
        undulatory_step += 1
        errors = np.zeros((8, 3))
        if undulatory_step >= 8:
            undulatory_step = 0
    
    # Replicate multirate simulation
    if (i % 5) == 0:
        # Tail Kinematics and Thrust estimation (8 motors & Tail centreline)
        tail_data[:8] =motor_states[:, 1] 
        tail_data[8] = tail_centreline
        tail_data_tensor = torch.from_numpy(tail_data).float().view(1, -1, 1)  
        tail_kinematics_output = tail_kinematics_model(tail_data_tensor).view(-1).detach().numpy()

        # Thrust estimation
        thrust_input[0,:12] = rigid_body_states  # Rigid body states
        thrust_input[0,12:14] = tail_kinematics_output[:2]  # Tail kinematics outputs
        thrust_input[0,14] = old_caudal_amplitude  # Motor angular velocities
        thrust_input[0,15] = tail_centreline  # Tail centreline position

        # Scale thrust inputs
        thrust_data_tensor = torch.from_numpy(input_scaler_thrust.transform(thrust_input)).float()
        thrust_data = thrust_model(thrust_data_tensor).detach().numpy()
        old_caudal_amplitude = tail_kinematics_output[1]
        # Scale thrust outputs
        thrust_data = output_scaler_thrust.inverse_transform(thrust_data)

        # Rigid Body state update
        rigid_body_u_signals[:7] = thrust_data[0]  # Thrust forces and moments
        rigid_body_u_signals[7] = tail_kinematics_output[0]  # Moments from tail
        # Scale rigid body inputs
        rigid_body_states_scaled = torch.from_numpy(rigid_body_x_scaler.transform(rigid_body_states.reshape(1, -1))).float()
        rigid_body_u_scaled = torch.from_numpy(rigid_body_u_scaler.transform(rigid_body_u_signals.reshape(1, -1))).float()
        rigid_body_states = rk4_step_func(rigid_body_model, rigid_body_states_scaled, rigid_body_u_scaled, rigid_body_dt).detach().numpy()
        # Inverse scale rigid body states
        rigid_body_states = rigid_body_x_scaler.inverse_transform(rigid_body_states)

    # Store data
    step_angle_history[i] = step_angle
    tail_centreline_history[i, 0] = tail_centreline
    smooth_angle_history[i] = step_angle_smooth
    u_signals_history[i] = u_signals.reshape(8, 1)
    motor_states_history[i] = motor_states
    tail_kinematics_history[i] = tail_kinematics_output
    thrust_history[i] = thrust_data
    rigid_body_states_history[i] = rigid_body_states.reshape(1, -1)

end_time = time.time()
print(f"Simulation Time for {N} steps: {end_time - start_time} seconds")

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
    plt.plot(time, u_signals_history[:, j], label=f'Motor {j+1}')
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
    plt.subplot(7, 1, j+1)
    plt.plot(time, thrust_history[:, j], label=f'Thrust {j+1}')
    plt.xlabel('Time (s)')
    plt.ylabel(f'Thrust {j+1} Value')
    plt.legend()

# Plot rigid body states results
plt.figure()
for j in range(12):
    plt.subplot(12, 1, j+1)
    plt.plot(time, rigid_body_states_history[:, 0, j], label=f'State {j+1}')
    plt.xlabel('Time (s)')
    plt.ylabel(f'State {j+1} Value')
    plt.legend()

# Show all plots
plt.show()