from cgitb import reset
import multiprocessing
import numpy as np
import os
import time
import gym
import subprocess
from gym import spaces
# import pyautogui
import threading
import torch
import pickle
import pandas as pd
from io import StringIO
from sklearn import base
from socketserver import TCPServer
from IPython.display import display
from multiprocessing import Manager
from scipy.spatial.transform import Rotation as R

from queue import Queue 

# imports from other python files
# from eval_LSTM_for_Shubhrika import run_surrogate_model



# CSV/txt file path
file_name = "C:/TetherNet/Shubhrika/ICRA2025/RL_DATA/RL_DATA_set_2_18000_combined.csv"

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, Dataset
import torch.optim as optim

import joblib
import time
import pickle
from matplotlib import pyplot as plt

############### Surrogate model ######################
class LSTMModel(nn.Module):
    def __init__(self, static_input_size, hidden_size, output_size, max_seq_len, num_layers=1):
        super(LSTMModel, self).__init__()
        self.static_fc = nn.Linear(static_input_size, hidden_size)
        self.lstm = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, batch_first=True, num_layers=num_layers)
        self.fc = nn.Linear(hidden_size, output_size)
        self.max_seq_len = max_seq_len
        self.hidden_size = hidden_size
        self.num_layers = num_layers

    def forward(self, static_features):
        batch_size = static_features.size(0)
        h0 = torch.tanh(self.static_fc(static_features)).unsqueeze(0).repeat(self.num_layers, 1, 1)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=static_features.device)
        
        dummy_input = torch.zeros(batch_size, self.max_seq_len, self.hidden_size, device=static_features.device)
        lstm_out, _ = self.lstm(dummy_input, (h0, c0))
        
        out = self.fc(lstm_out)
        return out

def denormalize_ARR(outARR_pred, min_val, max_val):
    max_seq_len = 70
    batch_size = outARR_pred.shape[0]
    outARR_pred = outARR_pred.cpu().numpy()

    arr_denorm = np.zeros((batch_size, max_seq_len, outARR_pred.shape[2]))

    for i in range(outARR_pred.shape[2]):
        arr_denorm[:, :, i] = outARR_pred[:,:,i] * (max_val[i] - min_val[i]) + min_val[i]

    return arr_denorm

def run_surrogate_model(model_inputs):
    '''
    model_inputs should be a (n_samples, 15) array
    '''
    # model_inputs[0,2] = -model_inputs[0,2]
    # model_inputs_old = model_inputs.copy()
    # model_inputs[0, 3:6] = model_inputs_old[]
    model_id = 4
    model_folder = 'surrogate_models_ICRA2025'

    model_inputs = np.array(model_inputs).reshape(1, 15)

    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            # print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # model_savename = f"LSTM_ICRA2025_{model_id}.pth"
    # model = torch.load('LSTM_ICRA2025_scripted.pth').to(device)
    
    model = LSTMModel(static_input_size=15, hidden_size=512, 
                    output_size=12, max_seq_len=70, num_layers=1).to(device)

    # model = torch.load(f'./{model_folder}/LSTM_ICRA2025_{model_id}.pth').to(device)
    state_dict_path = f"./{model_folder}/LSTM_ICRA2025_state_dict_{model_id}.pth"
    model.load_state_dict(torch.load(state_dict_path, weights_only=True))

    #scripted_model = torch.jit.script(model)  # Convert to TorchScript
    #scripted_model.save("LSTM_ICRA2025_scripted.pth")  # Save model



    ub = np.array([ 9, 9, -40, 10, 10, 30,       # position and rotation
                180, 180, 180,                # initial orientation
                180, 180, 180, 180, 55, 12])      # thrust angles (phi1~4, theta), thrust force
    lb = np.array([-9, -9, -60, 1, 1, 5, 
                    -180, -180, -180,
                    -180, -180, -180, -180, 35, 5])
    with open(f'output_scaler.pkl', 'rb') as f:
        output_scaler = pickle.load(f)
    out_min, out_max = output_scaler

    ub_mx = np.tile(ub, (model_inputs.shape[0],1))
    lb_mx = np.tile(lb, (model_inputs.shape[0],1))

    inARR_norm = (model_inputs - lb_mx) / (ub_mx-lb_mx)
    input_batch = torch.tensor(inARR_norm, dtype=torch.float).to(device)
    model.eval()
    with torch.no_grad():
        outputs = model(input_batch)

    traj_denorm = denormalize_ARR(outputs, out_min, out_max)

    return traj_denorm


############### Surrogate model end ####################


# RL environment
class Tethernet_Env(gym.Env):
    def __init__(self, port, env_id, num_envs):
        """
        Initialize the environment with data passed directly.
        """
        time.sleep(5)
        self.port = port
        self.num_envs = num_envs
        self.env_id = env_id
        self.bounded_target_states_est = None
        self.iteration = 0

        # Move tensors to GPU if available
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        row_number = self.iteration * self.num_envs + self.env_id
        
        self.target_ground_truth, self.target_angular_velocity = self.read_data(file_name, row_number)




        # Env state
        # print(f"Env ID: {self.env_id} is initializing with Row Number: {row_number}")
        # print(f"Ground Truth: {self.target_ground_truth}")
        # print(f"Estimated Pose: {self.target_states_est}")
        # print(f"Angular Velocity: {self.target_angular_velocity}")

        # Actions
        self.action_ub = [135, 135, 135, 135, 55, 12]
        self.action_lb = [-45, -45, -45, -45, 35, 5]
        # self.action_st = [1.0, 1.0, 1.0, 1.0, 1.0, 0.1]
        self.action_st = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1]

        # Target position, Orientations (at t = 0, 5, 10, 15)    
        self.pose_lb = [-9, -9, -60, -180, -180, -180, -180, -180, -180, -180, -180, -180, -180, -180, -180]     # -Z
        self.pose_ub = [9, 9, -40, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180]                    #- Z

        # Action space and observation space
        # action_space_num = np.divide(np.subtract(self.action_ub, self.action_lb), self.action_st)
        # action_space_n = action_space_num.astype(int)
        # self.action_space = spaces.MultiDiscrete(action_space_n)
        self.action_space = spaces.Box(low=0, high=1, shape=(len(self.action_st),))

        self.observation_space = spaces.Box(low=np.array(self.pose_lb), high=np.array(self.pose_ub), dtype=np.float32)

        self.max_steps = 1
        self.states = None
        self.step_no = 0
        self.episode_steps = 0

        # Weight parameters
        self.alpha1 = 1.5
        self.alpha2 = 1.0
        self.alpha3 = 1.0

        # Initialize state
        # print(f"Initial Coordinates: Step 1 {self.target_ground_truth}")
        # print(f"Process {self.port} working on provided data")

        # Bound the  states
        #self.bounded_target_states_est = self.bound_states(self.target_ground_truth)   # not needed for GT since it is already within bounds
        # print(f'Bounded states : {self.bounded_target_states_est}')

        # Normalize estimated states
        self.states = self.normalize_states(self.target_ground_truth)
        # print(f'Current state: {self.states}')


    def read_data(self, file_name, row_number):
       
       with open(file_name, mode='r', encoding='utf-8-sig') as file:  # The 'utf-8-sig' encoding automatically removes the BOM
        lines = file.readlines()

        if 'x' in lines[0].lower():  # Check if the first line contains the header (e.g., 'x', 'y', etc.)
            lines = lines[1:]

        if row_number >= len(lines):
            raise IndexError(f"Row {row_number} exceeds the number of rows in the file.")

        # Read and split the values from the CSV row
        line = lines[row_number].strip()
        values = list(map(float, line.split(',')))

        # Split the data into ground truth pose, angular velocity, and estimated pose
        ground_truth_pose = values[:15]  # Target ground truth pose (position and orientation)
        angular_velocity = values[15:18]  # Angular velocity (x, y, z)
        estimated_pose = values[18:]  # Estimated pose (position and orientation


        # Convert estimated state Z to -Z
        estimated_pose[2] = - estimated_pose[2]      ##### Z to - Z
        ground_truth_pose[2] = -ground_truth_pose[2]

        return ground_truth_pose, angular_velocity

    # Bound state with Pose limits
    def bound_states(self, states):
        bounded_states = []
        for s, lb, ub in zip(states, self.pose_lb, self.pose_ub):
            bounded_value = max(lb, min(s, ub))
            bounded_states.append(bounded_value)
        return bounded_states


    def normalize_states(self, states):
        # print(f"States: {states}")
        # print(f"Lower bounds: {self.pose_lb}")
        # print(f"Upper bounds: {self.pose_ub}")

        if not all(isinstance(s, (int, float)) for s in states):
            raise TypeError(f"Expected states to be numeric, got {states}")

        if not hasattr(states, '__iter__'):
            raise TypeError(f"Expected states to be iterable, got {type(states)}")
        if len(states) != len(self.pose_lb) or len(states) != len(self.pose_ub):
            raise ValueError("Length of states does not match length of bounds")

        normalized_states = []
        for s, lb, ub in zip(states, self.pose_lb, self.pose_ub):
            normalized_value = (s - lb) / (ub - lb)
            normalized_value = max(0, min(1, normalized_value))
            normalized_states.append(normalized_value)

        return normalized_states



    def calculate_reward(self, full_traj, target_ground_truth_pose, action6):

        # Variables to store values
        v_net_list = []
        direction_vector_list = []
        dot_product_list = []
        beta_arr_list = []
        cos_beta_list = []


        full_traj = full_traj.squeeze()
        # Center of Target Debris
        CT = np.array(target_ground_truth_pose[:3])
        # CT[2] = -CT[2]
        
        
        total_time_traj = full_traj.shape[0]


        # Starting at 15th sec since thrusters start at 15th sec
        xMU = full_traj[:, 0:4]
        yMU = full_traj[:, 4:8]
        zMU = full_traj[:, 8:12]

        xN = np.mean(xMU, axis=1)
        yN = np.mean(yMU, axis=1)
        zN = np.mean(zMU, axis=1)
        Cnet = np.vstack((xN, yN, zN))   # (3, 70)
        Cnet = np.transpose(Cnet)

        t_open = np.where((Cnet[:, 2] - CT[2]) < 5.5)[0]
        if t_open.size == 0:
            t_open = total_time_traj-1  # modified (-1) for indexing
        else:
            t_open = t_open[0]

        t_close = np.where((CT[2]- Cnet[:, 2]) > 5.5)[0]
        if t_close.size == 0:
            t_close = total_time_traj-1  # modified (-1) for indexing
        else:
            t_close = t_close[0]

        velocities = np.diff(full_traj, axis=0) # (69, 12)
        v_net_x = np.mean(velocities[:, :4], axis=1)
        v_net_y = np.mean(velocities[:, 4:8], axis=1)
        v_net_z = np.mean(velocities[:, 8:12], axis=1)
        v_net = np.vstack((v_net_x, v_net_y, v_net_z))
        v_net = np.transpose(v_net)
        
        CT_arr = np.tile(CT, (Cnet.shape[0],1))
        direction_vector = CT_arr - Cnet   # (70, 3)
        d = np.linalg.norm(direction_vector, axis=1)  # (70, )
        beta_arr = np.zeros(t_open)

        for i in range(t_open):
            dot_product = np.dot(v_net[i, :], direction_vector[i, :])
            v_net_magnitude = np.linalg.norm(v_net[i, :])
            direction_vector_magnitude = np.linalg.norm(direction_vector[i, :])
            cos_beta = dot_product / (v_net_magnitude * direction_vector_magnitude)
            beta_arr[i] = np.degrees(np.arccos(np.clip(cos_beta, -1.0, 1.0)))

            dot_product_list.append(dot_product)
            v_net_list.append(v_net_magnitude)
            direction_vector_list.append(direction_vector_magnitude)
            cos_beta_list.append(cos_beta)
            beta_arr_list.append(beta_arr)


        Rcenter = -self.alpha1 * np.sum((beta_arr/180) / t_open)
        k = 0.0121 / 8.9
        Fthrust = action6
        Mfuel = k * Fthrust * t_close/10  # fuel consump rate by EACH MU

        Mmax = 0.5  # fuel carried by EACH MU
        Rfuel = -self.alpha2*4*Mfuel / (4*Mmax)





        open_traj_flatten = full_traj[:t_open, :].flatten()
        # print('open ', open_traj_flatten)
        Anet = self.calculate_net_area(open_traj_flatten)

        Amax = 1.2 * 22**2
        Rarea = self.alpha3 * abs(Anet / Amax)

        epsilon = 40
        if t_open == total_time_traj:
            Ropen = 0
        else:
            Ropen = min(1, 1 - (beta_arr[-1] - epsilon) / (180 - epsilon))  # FL DEBUG
        # print('Ropen :' , Ropen)

        r_phase_MU = np.transpose(np.array([xMU[t_open, :], yMU[t_open, :], zMU[t_open, :]]))
        r_D = np.tile(CT, (4,1))
        r_rel = r_phase_MU - r_D

        Rphase = 0
        if r_rel[0, 0]<0 and r_rel[0, 1]<0:  # MU1
            Rphase = Rphase + 0.25
        if r_rel[1, 0]>0 and r_rel[1, 1]<0:  # MU2
            Rphase = Rphase + 0.25
        if r_rel[2, 0]<0 and r_rel[2, 1]>0:  # MU3
            Rphase = Rphase + 0.25
        if r_rel[3, 0]>0 and r_rel[3, 1]>0:  # MU4
            Rphase = Rphase + 0.25

        total_reward = Rcenter + Rfuel + Rarea + Ropen + Rphase
        ################# END OF NEW REWARD ##################

        # print('total reward:', total_reward)
        
        with open('./tethernet_Experiment_icra/Split_reward/reward_split_log%.3d.txt' % self.port, 'a') as f:
        # Write a newline character to the file
            act_rw =  np.asarray([Rcenter, Rfuel, Rarea, Ropen, Rphase, total_reward, 
                                  CT[0], CT[1], CT[2], target_ground_truth_pose[3], target_ground_truth_pose[4], target_ground_truth_pose[5]])
            line_act_rw = " ".join(act_rw.astype(str)) + "\n"
            f.write(line_act_rw)
            # time.sleep(2)


        # Save Anet, Mfuel, beta_arr (final)
        with open('./tethernet_Experiment_icra/Split_reward/area_final_beta%.3d.txt' % self.port, 'a') as f:
        # Write a newline character to the file
                act_rw =  np.asarray([Anet, Mfuel, beta_arr[-1]] )
                line_act_rw = " ".join(act_rw.astype(str)) + "\n"
                f.write(line_act_rw)
                # time.sleep(2)

        
        with open('./tethernet_Experiment_icra/Split_reward/reward_data_%.3d.txt' % self.port, 'a') as f:
            for i in range(t_open):
                act_rw = np.asarray([v_net_list[i], direction_vector_list[i], cos_beta_list[i]])
                line_act_rw = " ".join(act_rw.astype(str)) + "\n"
                f.write(line_act_rw)
 

        

        return total_reward

    def calculate_net_area(self, full_traj):

        # Extract the positions of the four MUs
        MU_positions = [(full_traj[i], full_traj[i + 4], full_traj[i + 8]) for i in range(4)]

        # Position of MUs
        # print(f'Mu1 0 {MU_positions[0]}')
        # print(f'Mu2 1 {MU_positions[1]}')
        # print(f'Mu3 2 {MU_positions[2]}')
        # print(f'Mu4 3 {MU_positions[3]}')
        
        # Triangle MU1-MU2-MU3
        a12 = np.linalg.norm(np.array(MU_positions[1]) - np.array(MU_positions[0]))
        b23 = np.linalg.norm(np.array(MU_positions[2]) - np.array(MU_positions[1]))
        c31 = np.linalg.norm(np.array(MU_positions[0]) - np.array(MU_positions[2]))
        s1 = (a12 + b23 + c31) / 2
        # print(f's1 {s1}')
        Area1 = np.sqrt(s1 * (s1 - a12) * (s1 - b23) * (s1 - c31))

        ########### debug################
        Area1 = np.sqrt(max(0, s1 * (s1 - a12) * (s1 - b23) * (s1 - c31)))
        ############################################

        # print(f'Area 1: {Area1}')
        
        # Triangle MU1-MU3-MU4
        a13 = np.linalg.norm(np.array(MU_positions[2]) - np.array(MU_positions[0]))
        b34 = np.linalg.norm(np.array(MU_positions[3]) - np.array(MU_positions[2]))
        c41 = np.linalg.norm(np.array(MU_positions[3]) - np.array(MU_positions[0]))
        s2 = (a13 + b34 + c41) / 2
        # print(f's2 {s2}')
        # print(a13, b34, c41)
        a =  (s2-a13 )
        b = (s2 - b34)
        c = (s2 -c41)

        # print(a,b,c)

        # z = s2 * a * b * c
        # Area2 = np.sqrt(z)

        ######## debug#################

        Area2 = np.sqrt(max(0, s2 * (s2 - a13) * (s2 - b34) * (s2 - c41)))

        ######### debug################
        
        # print(f'Area 2: {Area2}')
        
        # Total net area
        Anet = Area1 + Area2
        
        return Anet

    def scale_action(self, action):
            
            scaled_action = np.array(self.action_lb) + (action * (np.array(self.action_ub) - np.array(self.action_lb)))
            return scaled_action

    def step(self, action):
        self.step_no += 1
        self.episode_steps += 1

        new_state = self.states

        # action1 = self.action_lb[0] + action[0] * self.action_st[0]
        # action2 = self.action_lb[1] + action[1] * self.action_st[1]
        # action3 = self.action_lb[2] + action[2] * self.action_st[2]
        # action4 = self.action_lb[3] + action[3] * self.action_st[3]
        # action5 = self.action_lb[4] + action[4] * self.action_st[4]
        # action6 = self.action_lb[5] + action[5] * self.action_st[5]

        # conitnuous actions
        action_scaled= self.scale_action(action)
        action_resize = np.round(np.array(action_scaled) / np.array(self.action_st)) * np.array(self.action_st)
        action1, action2, action3, action4, action5, action6 = action_resize
        surrogate_input_array = [action1, action2, action3, action4, action5, action6]
        # print(f'Actions: {surrogate_input_array}')

        input_dir = "./Inputs"
        if not os.path.exists(input_dir):
            os.makedirs(input_dir)

        with open(os.path.join(input_dir, "Input_param%.3d.txt" % self.port), "w") as f:
            f.write(" ".join(str(x) for x in surrogate_input_array))


        # print(f'Current Ground truth: {self.target_ground_truth}')

        # print(f"Surrgate model input : {surrogate_input_array}")


       # convert GT to list
        # target_ground_list = self.target_ground_truth.split(',')

        # target_ground_truth_pose =  [float(value.strip()) for value in target_ground_list]

        target_ground_list = list(self.target_ground_truth)
        angular_vel = list(self.target_angular_velocity)  # 3
        actions_list = list(surrogate_input_array)    # 6

        # Extract position (X, Y, Z)
        target_location = target_ground_list[:3]  # (3)
        # target_location[2] = -target_location[2]

        inital_orientation  = target_ground_list[3:6]  # 3

        surrogate_input = target_location + angular_vel + inital_orientation + actions_list

        
        ################## Call Surrogate Model ########################
        # Input to Surrogate - (X,Y,Z, WX,WY,WZ, action1, action2, action3, action4, action5, action6)  #########
        #full_traj = self.run_surrogate_model(target_location, self.target_angular_velocity, surrogate_input_array)   

        full_traj = run_surrogate_model(surrogate_input)  

        #print('Full traj ', full_traj)


        # Calculate reward
        reward = self.calculate_reward(full_traj, self.target_ground_truth, action6)
         
        # check if RL needs to be stop
        if self.step_no>=1:
            done = True
        else:
            done = False

        info = {"reward": reward, "done": done, "episode_steps": self.episode_steps}

        # print("Thread %.3d Episode: " % self.port, self.step_no - 1)
        # print("Actions: ", surrogate_input_array)
        # print("Current reward: ", reward)
        # print("Episode steps:", self.episode_steps)

        if done:
            self.episode_steps = 0

        with open('./tethernet_Experiment_icra/action_reward_log%.3d.txt' % self.port, 'a') as f:
            act_rw = np.append(target_ground_list, surrogate_input_array + [reward])
            line_act_rw = " ".join(act_rw.astype(str)) + "\n"
            f.write(line_act_rw)

        pickle_folder = "Pickle_folder"
        pickle_file = f"./tethernet_Experiment_icra/{pickle_folder}/simoutput_log{self.port}.pkl"
        pickle_newline = np.concatenate((target_ground_list, surrogate_input_array,[reward]))

        if os.path.exists(pickle_file):
            with open(pickle_file, 'rb') as f:
                pickle_old = pickle.load(f)
            pickle_new = np.vstack((pickle_old, pickle_newline))
            with open(pickle_file, 'wb') as f:
                pickle.dump(pickle_new, f)
        else:
            pickle_new = pickle_newline
            with open(pickle_file, 'wb') as f:
                pickle.dump(pickle_new, f)

        # print("new_state before conversion:", new_state)
        if not isinstance(new_state, np.ndarray):
            try:
                new_state = np.array(new_state, dtype=np.float32)
                # print("new_state after conversion to ndarray:", new_state)
            except Exception as e:
                # print(f"Error converting new_state to ndarray: {e}")
                pass

        if np.isnan(new_state).any():
            raise ValueError("New state contains NaN values")

        return new_state, reward, done, info

    def reset(self):

        # print(f"Environment {self.port} is being reset.")

        self.step_no = 0
        self.episode_steps = 0

        row_number = self.iteration * self.num_envs + self.env_id


        self.target_ground_truth, self.target_angular_velocity = self.read_data(file_name, row_number)

        
        states = self.normalize_states(self.target_ground_truth)
    
        
        
        self.states = states 

        self.iteration += 1

        # print(f' Reset state is:  {self.states} ')
        return self.states