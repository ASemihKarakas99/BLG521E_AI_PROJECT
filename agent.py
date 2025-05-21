import torch
import torch.nn as nn
import gymnasium as gym
import argparse
import itertools
import random
import os
import yaml
from datetime import datetime, timedelta

from DQN.dqn import DQN
from buffers.experience_replay import ReplayBuffer
from buffers.prioritized_experience_replay import PrioritizedReplayBuffer
from buffers.attentive_experience_replay import AttentiveReplayBuffer

# For printing date and time
DATE_FORMAT = "%m-%d %H:%M:%S"

# Directory for saving run info
RUNS_DIR = "training_results"
os.makedirs(RUNS_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Agent:
    def __init__(self, hyperparameter_set, buffer_type='uniform'):
        with open('hyperparameters.yml', 'r') as file:
            all_hyperparameter_sets = yaml.safe_load(file)
            hyperparameters = all_hyperparameter_sets[hyperparameter_set]
            # print(hyperparameters)

        self.hyperparameter_set = hyperparameter_set
        self.buffer_type = buffer_type  # 'uniform' or 'prioritized'
        # Hyperparameters (adjustable)
        self.env_id                    = hyperparameters['env_id']
        self.learning_rate_a           = hyperparameters['learning_rate_a']        # learning rate (alpha)
        self.discount_factor_g         = hyperparameters['discount_factor_g']      # discount rate (gamma)
        self.network_sync_rate         = hyperparameters['network_sync_rate']      # number of steps the agent takes before syncing the policy and target network
        self.replay_memory_size        = hyperparameters['replay_memory_size']     # size of replay memory
        self.mini_batch_size           = hyperparameters['mini_batch_size']        # size of the training data set sampled from the replay memory
        self.epsilon_init              = hyperparameters['epsilon_init']           # 1 = 100% random actions
        self.epsilon_decay             = hyperparameters['epsilon_decay']          # epsilon decay rate
        self.epsilon_min               = hyperparameters['epsilon_min']            # minimum epsilon value
        self.stop_on_reward            = hyperparameters['stop_on_reward']         # stop training after reaching this number of rewards
        self.fc1_nodes                 = hyperparameters['fc1_nodes']
        self.env_make_params           = hyperparameters.get('env_make_params',{}) # Get optional environment-specific parameters, default to empty dict
        self.early_stop_avg_window     = hyperparameters.get('early_stop_avg_window')   # Moving average window size
        self.es_reward_threshold       = hyperparameters.get('early_stop_reward_threshold')  
        self.num_episodes              = hyperparameters.get('num_episodes')


        # Neural Network
        self.loss_fn = nn.MSELoss()          # NN Loss function. MSE=Mean Squared Error can be swapped to something else.
        self.optimizer = None                # NN Optimizer. Initialize later.

        # Path to Run info
        self.RUN_DIR = os.path.join(RUNS_DIR, self.hyperparameter_set, self.buffer_type)
        if os.path.exists(self.RUN_DIR):
            raise FileExistsError(f"Run directory already exists: {self.RUN_DIR}")
        else:
            os.makedirs(self.RUN_DIR)


        self.LOG_FILE   = os.path.join(self.RUN_DIR, 'training.log')
        self.MODEL_FILE = os.path.join(self.RUN_DIR, 'model.pt')
        self.REWARD_FILE = os.path.join(self.RUN_DIR, 'rewards.csv')
        self.GRAPH_FILE = os.path.join(self.RUN_DIR, f'{self.hyperparameter_set}.png')


    def run(self, num_episodes=None, render=False): 
        if num_episodes is None:
            num_episodes = self.num_episodes
            
        start_time = datetime.now()
        # last_graph_update_time = start_time

        log_message = f"{start_time.strftime(DATE_FORMAT)}: Training starting..."
        print(log_message)
        with open(self.LOG_FILE, 'w') as file:
            file.write(log_message + '\n')

        # Create instance of the environment.
        # Use "**self.env_make_params" to pass in environment-specific parameters from hyperparameters.yml.
        env = gym.make(self.env_id, render_mode='human' if render else None, **self.env_make_params)

        # Number of possible actions
        num_actions = env.action_space.n

        # Get observation space size
        num_states = env.observation_space.shape[0] # Expecting type: Box(low, high, (shape0,), float64)

        # List to keep track of rewards collected per episode.
        rewards_per_episode = []

        # Create policy and target network. Number of nodes in the hidden layer can be adjusted.
        policy_dqn = DQN(num_states, num_actions, self.fc1_nodes).to(device)


        epsilon = self.epsilon_init

        # Initialize replay memory
        # memory = ReplayMemory(self.replay_memory_size)

        if self.buffer_type == 'uniform':
            memory = ReplayBuffer(num_states, 1, self.replay_memory_size)
        elif self.buffer_type == 'prioritized':
            memory = PrioritizedReplayBuffer(num_states, 1, self.replay_memory_size)
        elif self.buffer_type == 'attentive':
            memory = AttentiveReplayBuffer(num_states, 1, self.replay_memory_size)

        # Create the target network and make it identical to the policy network
        target_dqn = DQN(num_states, num_actions, self.fc1_nodes).to(device)
        target_dqn.load_state_dict(policy_dqn.state_dict())

        # Policy network optimizer. "Adam" optimizer can be swapped to something else.
        self.optimizer = torch.optim.Adam(policy_dqn.parameters(), lr=self.learning_rate_a)

        # List to keep track of epsilon decay
        epsilon_history = []

        # Track number of steps taken. Used for syncing policy => target network.
        step_count=0

        # Track best reward
        best_reward = -9999999

        print(f"Early stopping if average reward over last {self.early_stop_avg_window} episodes exceeds {self.es_reward_threshold}")
        # Train INDEFINITELY, manually stop the run when you are satisfied (or unsatisfied) with the results
        # for episode in itertools.count(): # For infinite training
        for episode in range(num_episodes):

            state, _ = env.reset()  # Initialize environment. Reset returns (state,info).
            state = torch.tensor(state, dtype=torch.float, device=device) # Convert state to tensor directly on device

            terminated = False      # True when agent reaches goal or fails
            episode_reward = 0.0    # Used to accumulate rewards per episode

            # Perform actions until episode terminates or reaches max rewards
            # (on some envs, it is possible for the agent to train to a point where it NEVER terminates, so stop on reward is necessary)
            while(not terminated and episode_reward < self.stop_on_reward):

                # Select action based on epsilon-greedy
                if random.random() < epsilon:
                    # select random action
                    action = env.action_space.sample()
                    action = torch.tensor(action, dtype=torch.int64, device=device)
                else:
                    # select best action
                    with torch.no_grad():
                        # state.unsqueeze(dim=0): Pytorch expects a batch layer, so add batch dimension i.e. tensor([1, 2, 3]) unsqueezes to tensor([[1, 2, 3]])
                        # policy_dqn returns tensor([[1], [2], [3]]), so squeeze it to tensor([1, 2, 3]).
                        # argmax finds the index of the largest element.
                        action = policy_dqn(state.unsqueeze(dim=0)).squeeze().argmax()

                # Execute action. Truncated and info is not used.
                new_state,reward,terminated,truncated,info = env.step(action.item())

                # Accumulate rewards
                episode_reward += reward

                # Convert new state and reward to tensors on device
                new_state = torch.tensor(new_state, dtype=torch.float, device=device)
                reward = torch.tensor(reward, dtype=torch.float, device=device)

                # Save experience into memory
                # memory.append((state, action, new_state, reward, terminated))

                memory.add((state.cpu().numpy(), action.cpu().numpy(), reward.cpu().numpy(),
                            new_state.cpu().numpy(), int(terminated)))


                # Increment step counter
                step_count+=1

                # Move to the next state
                state = new_state

            # Keep track of the rewards collected per episode.
            rewards_per_episode.append(episode_reward)

            # Check early stopping condition
            if len(rewards_per_episode) >= self.early_stop_avg_window:
                recent_avg = sum(rewards_per_episode[-self.early_stop_avg_window:]) / self.early_stop_avg_window
                if recent_avg >= self.es_reward_threshold:
                    log_message = f"{datetime.now().strftime(DATE_FORMAT)}: Early stopping triggered at episode {episode} with average reward {recent_avg:.2f}"
                    print(log_message)
                    with open(self.LOG_FILE, 'a') as file:
                        file.write(log_message + '\n')
                    break


            # Save model when new best reward is obtained.
            if episode_reward > best_reward:
                log_message = f"{datetime.now().strftime(DATE_FORMAT)}: New best reward {episode_reward:0.1f} ({(episode_reward-best_reward)/best_reward*100:+.1f}%) at episode {episode}, saving model..."
                print(log_message)
                with open(self.LOG_FILE, 'a') as file:
                    file.write(log_message + '\n')

                torch.save(policy_dqn.state_dict(), self.MODEL_FILE)
                best_reward = episode_reward

            # Update graph every x seconds
            # current_time = datetime.now()
            # if current_time - last_graph_update_time > timedelta(seconds=10):
            #     self.save_graph(rewards_per_episode, epsilon_history)
            #     last_graph_update_time = current_time

            # If enough experience has been collected
            # if len(memory)>self.mini_batch_size:
            #     mini_batch = memory.sample(self.mini_batch_size)
            #     self.optimize(mini_batch, policy_dqn, target_dqn)


            if memory.real_size > self.mini_batch_size:
                if self.buffer_type == 'uniform':
                    batch = memory.sample(self.mini_batch_size)
                    self.optimize_uniform(batch, policy_dqn, target_dqn)
                elif self.buffer_type == 'prioritized':
                    batch, is_weights, tree_idxs = memory.sample(self.mini_batch_size)
                    self.optimize_prioritized(batch, is_weights, tree_idxs, policy_dqn, target_dqn, memory)
                elif self.buffer_type == 'attentive':
                    batch = memory.sample(self.mini_batch_size, current_state=state.cpu().numpy())
                    self.optimize_attentive(batch, policy_dqn, target_dqn)



                # Decay epsilon
                epsilon = max(epsilon * self.epsilon_decay, self.epsilon_min)
                epsilon_history.append(epsilon)

                # Copy policy network to target network after a certain number of steps
                if step_count > self.network_sync_rate:
                    target_dqn.load_state_dict(policy_dqn.state_dict())
                    step_count=0


        with open(self.REWARD_FILE, 'w') as f:
            f.write("episode,reward\n")
            for idx, reward in enumerate(rewards_per_episode):
                f.write(f"{idx},{reward}\n")

    # Optimize policy network
    def optimize_uniform(self, mini_batch, policy_dqn, target_dqn):
        states, actions, rewards, next_states, terminations = mini_batch  # All tensors with batch dimension

        # Double-check dimensions
        assert states.ndim == 2
        assert actions.ndim == 1
        assert rewards.ndim == 1
        assert next_states.ndim == 2
        assert terminations.ndim == 1

        with torch.no_grad():
            target_q = rewards + (1 - terminations) * self.discount_factor_g * \
                    target_dqn(next_states).max(dim=1)[0]  # [B]

        # Get current Q values
        current_q = policy_dqn(states).gather(1, actions.unsqueeze(1)).squeeze(1)  # [B]

        loss = self.loss_fn(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


    def optimize_attentive(self, batch, policy_dqn, target_dqn):
        states, actions, rewards, next_states, terminations = batch

        with torch.no_grad():
            target_q = rewards + (1 - terminations) * self.discount_factor_g * target_dqn(next_states).max(dim=1)[0]

        current_q = policy_dqn(states).gather(dim=1, index=actions.long()).squeeze()
        loss = self.loss_fn(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()



    def optimize_prioritized(self, batch, is_weights, tree_idxs, policy_dqn, target_dqn, memory):
        states, actions, rewards, next_states, dones = batch

        with torch.no_grad():
            target_q = rewards + (1 - dones) * self.discount_factor_g * target_dqn(next_states).max(dim=1)[0]

        current_q = policy_dqn(states).gather(dim=1, index=actions.long().unsqueeze(1)).squeeze()

        # DO NOT detach td_errors here – keep them in graph for gradient computation
        td_errors = target_q - current_q

        # Importance-sampling weighted loss
        loss = (is_weights.squeeze() * (td_errors ** 2)).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Detach td_errors when updating priorities (no gradient needed)
        memory.update_priorities(tree_idxs, td_errors.abs().detach().cpu())



if __name__ == "__main__":

    '''Hyperparameter Setting'''
    parser = argparse.ArgumentParser()
    parser.add_argument("--hyperparameter_set", type=str, default="lunar_lander") # lunar_lander, cartpole
    parser.add_argument("--buffer_type", type=str, default="uniform") # uniform, prioritized, attentive
    args = parser.parse_args()

    agent = Agent(args.hyperparameter_set, args.buffer_type)
    agent.run(render=False) # Can be changed to True to visualize the agent
