import os
import numpy as np
import time
import torch
import json
import glob
import torch.multiprocessing as mp
import matplotlib.pyplot as plt
from multiprocessing import Lock, Manager
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback, CallbackList, StopTrainingOnMaxEpisodes
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.utils import set_random_seed
# from tether_net_vision_parallel_final_new_15state import Tethernet_Env  # Custom environment
# from tether_net_vision_parallel_final_new_15state_continuous import Tethernet_Env  # Continuous actions
from tether_net_vision_parallel_final_new_15state_continuous_random import Tethernet_Env  # Continuous actions
# from tether_net_vision_parallel_final_new_15state_continuous_ground_truth import Tethernet_Env  # Continuous actions
# from tether_net_vision_parallel_final_new_15state_continuous_ground_truth_random import Tethernet_Env  # Continuous actions


# Ensure CUDA is available and being used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def delete_old_files(folder_path, format):
    # Use glob to find all .txt files in the folder
    old_files = glob.glob(os.path.join(folder_path, f'*.{format}'))
    # Loop through the list and delete each file
    for file in old_files:
        try:
            os.remove(file)
            print(f"Deleted: {file}")
        except Exception as e:
            print(f"Error deleting {file}: {e}")
    print(f"All .{format} files have been deleted.")
    


class SaveBestModelCallback(BaseCallback):
    def __init__(self, check_freq=1, log_dir="log_tether_net_icra", n_best_episodes=10, verbose=1):
        super(SaveBestModelCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.n_best_episodes = n_best_episodes
        self.best_mean_reward = -np.inf
        self.episode_rewards = []
        self.episode_counter = 0
        self.current_episode_reward = 0
        with open('RL_itercount.json', 'w') as file:
            json.dump(self.episode_counter + 1, file)

    def _safe_model_save(self):
        model_params = self.model.get_parameters()
        model_path = os.path.join(self.log_dir, 'best_model')
        np.savez(model_path, **model_params)

    def _on_step(self) -> bool:
        if 'obs' not in self.locals:
            print("Keys available in self.locals:", self.locals.keys())
            print("Skipping step as 'obs' is not in locals.")
            return True

        obs_tensor = self.model.policy.obs_to_tensor(self.locals["obs"])[0]
        if torch.isnan(obs_tensor).any():
            print("NaN detected in model output")

        actions, values, log_probs = self.model.policy.forward(obs_tensor)
        if torch.isnan(actions).any() or torch.isnan(values).any() or torch.isnan(log_probs).any():
            print("NaN detected in model output")


class PlottingCallback(BaseCallback):
    def __init__(self, plot_freq, save_path='./plots_icra/', verbose=0):
        super(PlottingCallback, self).__init__(verbose)
        self.plot_freq = plot_freq
        self.save_path = save_path
        self.rewards = []
        self.timesteps = []

    def _on_step(self):
        # Append the latest reward
        self.rewards.append(np.mean(self.locals['rewards']))
        self.timesteps.append(self.num_timesteps)

        # Check if it's time to plot
        if self.num_timesteps % self.plot_freq == 0:
            self._plot_rewards()

        return True

    def _plot_rewards(self):
        plt.figure(figsize=(10, 5))
        plt.plot(self.timesteps, self.rewards, label='Reward')
        plt.xlabel('Timesteps')
        plt.ylabel('Rewards')
        plt.title('Reward vs Timesteps')
        plt.legend()
        plt.grid(True)

        # Ensure the directory exists
        os.makedirs(self.save_path, exist_ok=True)
        plt.savefig(os.path.join(self.save_path, f'rewards_plot_{self.num_timesteps}.png'))
        plt.close()


class InfoCallback(BaseCallback):
    def __init__(self, verbose=1):
        super(InfoCallback, self).__init__(verbose)

    def _on_step(self):
        # Print total timesteps every 100 steps
        if self.num_timesteps % 100 == 0:
            print(f"Total timesteps executed: {self.num_timesteps}")
        return True

def make_env(env_id, seed, num_envs):

    def _init():
  
        env = Tethernet_Env(port=8888 + env_id, env_id= env_id, num_envs=num_envs)
        set_random_seed(seed)
        return env
    return _init

# RL training 
def train_rl_model(env, total_timesteps, callback):
    # model_path = './TetherNet_Unreal_icra/rl_model_unreal_icra_201216_steps.zip'  #check point 1, txt line12576, new lr0.001
    # model_path = './TetherNet_Unreal_icra/rl_model_unreal_icra_402432_steps.zip'  #check point 1, txt line25152, new lr0.01
    model_path = './TetherNet_Unreal_icra/rl_model_unreal_icra_515072_steps.zip'  #check point 1, accidentally stopped
    if os.path.exists(model_path):
        model = PPO.load(model_path, env=env)
        print("Loaded model from checkpoint")
        model.learning_rate = 0.01
    else:
        model = PPO("MlpPolicy", env, learning_rate=0.0001, batch_size=32 * 16, n_steps=32, verbose=1,
                    tensorboard_log="./tethernet_Experiment_icra/Thesis_vision/", device=device)

    model.learn(total_timesteps=total_timesteps, reset_num_timesteps=False, callback=callback)
    model.save("PPO_model_icra_1000ep")

    rewards = callback.callbacks[2].episode_rewards
    plt.plot(rewards)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Rewards over Episode')
    plt.show()

if __name__ == '__main__':
    data_clean = 0
    if data_clean == 1:
        input("!!!!!TRAINING LOG WILL BE DELETED!!!!!")
        folder_path = "tethernet_Experiment_icra"
        delete_old_files(folder_path, 'txt')
        folder_path = "./tethernet_Experiment_icra/Split_reward"
        delete_old_files(folder_path, 'txt')
        folder_path = "./tethernet_Experiment_icra/Pickle_folder"
        delete_old_files(folder_path, 'pkl')


    mp.set_start_method('spawn', force=True)

    seed = 0
    num_envs = 16
    total_timesteps = 200960
    iteration  = 0


    envs = [make_env(i, seed, num_envs) for i in range(num_envs)]
    env = SubprocVecEnv(envs)
    env = VecMonitor(env)

    # Callbacks
    checkpoint_callback = CheckpointCallback(save_freq=32, save_path='./TetherNet_Unreal_icra/', name_prefix='rl_model_unreal_icra')
    callback_max_episodes = StopTrainingOnMaxEpisodes(max_episodes=80000, verbose=1)
    # save_best_model_callback = SaveBestModelCallback()
    plot_rewards_callback = PlottingCallback(plot_freq=100, save_path='./reward_plots_icra/')
    info_callback = InfoCallback()
    # callback = CallbackList([checkpoint_callback, callback_max_episodes, save_best_model_callback, plot_rewards_callback, info_callback])
    callback = CallbackList([checkpoint_callback, callback_max_episodes, plot_rewards_callback, info_callback])


    # Train RL model
    train_rl_model(env, total_timesteps=total_timesteps, callback=callback)


    print("All processes terminated")
    print("Training completed")
