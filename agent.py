import torch
import gymnasium as gym

import itertools
import random
import os
import yaml
from datetime import datetime, timedelta

from DQN.dqn import DQN
from buffers.experience_replay import ReplayBuffer
from buffers.prioritized_experience_replay import PrioritizedReplayBuffer

# For printing date and time
DATE_FORMAT = "%m-%d %H:%M:%S"

# Directory for saving run info
RUNS_DIR = "runs"
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
        self.env_id             = hyperparameters['env_id']
        self.learning_rate_a    = hyperparameters['learning_rate_a']        # learning rate (alpha)
        self.discount_factor_g  = hyperparameters['discount_factor_g']      # discount rate (gamma)
        self.network_sync_rate  = hyperparameters['network_sync_rate']      # number of steps the agent takes before syncing the policy and target network
        self.replay_memory_size = hyperparameters['replay_memory_size']     # size of replay memory
        self.mini_batch_size    = hyperparameters['mini_batch_size']        # size of the training data set sampled from the replay memory
        self.epsilon_init       = hyperparameters['epsilon_init']           # 1 = 100% random actions
        self.epsilon_decay      = hyperparameters['epsilon_decay']          # epsilon decay rate
        self.epsilon_min        = hyperparameters['epsilon_min']            # minimum epsilon value
        self.stop_on_reward     = hyperparameters['stop_on_reward']         # stop training after reaching this number of rewards
        self.fc1_nodes          = hyperparameters['fc1_nodes']
        self.env_make_params    = hyperparameters.get('env_make_params',{}) # Get optional environment-specific parameters, default to empty dict


        # Neural Network
        self.loss_fn = nn.MSELoss()          # NN Loss function. MSE=Mean Squared Error can be swapped to something else.
        self.optimizer = None                # NN Optimizer. Initialize later.

        # Path to Run info
        self.LOG_FILE   = os.path.join(RUNS_DIR, f'{self.hyperparameter_set}.log')
        self.MODEL_FILE = os.path.join(RUNS_DIR, f'{self.hyperparameter_set}.pt')
        self.GRAPH_FILE = os.path.join(RUNS_DIR, f'{self.hyperparameter_set}.png')


    def run(self, num_episodes=10000, render=False):
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


        # Train INDEFINITELY, manually stop the run when you are satisfied (or unsatisfied) with the results
        for episode in itertools.count():

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


            if memory.real_size > self.mini_batch_size:
                states, actions, rewards, next_states, dones = memory.sample(self.mini_batch_size)
                self.optimize((states, actions, next_states, rewards, dones), policy_dqn, target_dqn)


                # Decay epsilon
                epsilon = max(epsilon * self.epsilon_decay, self.epsilon_min)
                epsilon_history.append(epsilon)

                # Copy policy network to target network after a certain number of steps
                if step_count > self.network_sync_rate:
                    target_dqn.load_state_dict(policy_dqn.state_dict())
                    step_count=0


    # Optimize policy network
    def optimize_uniform(self, mini_batch, policy_dqn, target_dqn):

        # Transpose the list of experiences and separate each element
        # states, actions, new_states, rewards, terminations = zip(*mini_batch)
        states, actions, new_states, rewards, terminations = mini_batch


        # Stack tensors to create batch tensors
        # tensor([[1,2,3]])
        states = torch.stack(states)

        actions = torch.stack(actions)

        new_states = torch.stack(new_states)

        rewards = torch.stack(rewards)
        terminations = torch.tensor(terminations).float().to(device)

        with torch.no_grad():
            # if self.enable_double_dqn:
            #     best_actions_from_policy = policy_dqn(new_states).argmax(dim=1)

            #     target_q = rewards + (1-terminations) * self.discount_factor_g * \
            #                     target_dqn(new_states).gather(dim=1, index=best_actions_from_policy.unsqueeze(dim=1)).squeeze()
            # else:
            # Calculate target Q values (expected returns)
            target_q = rewards + (1-terminations) * self.discount_factor_g * target_dqn(new_states).max(dim=1)[0]
            '''
                target_dqn(new_states)  ==> tensor([[1,2,3],[4,5,6]])
                    .max(dim=1)         ==> torch.return_types.max(values=tensor([3,6]), indices=tensor([3, 0, 0, 1]))
                        [0]             ==> tensor([3,6])
            '''

        # Calcuate Q values from current policy
        current_q = policy_dqn(states).gather(dim=1, index=actions.unsqueeze(dim=1)).squeeze()
        '''
            policy_dqn(states)  ==> tensor([[1,2,3],[4,5,6]])
                actions.unsqueeze(dim=1)
                .gather(1, actions.unsqueeze(dim=1))  ==>
                    .squeeze()                    ==>
        '''

        # Compute loss
        loss = self.loss_fn(current_q, target_q)

        # Optimize the model (backpropagation)
        self.optimizer.zero_grad()  # Clear gradients
        loss.backward()             # Compute gradients
        self.optimizer.step()       # Update network parameters i.e. weights and biases



    def optimize_prioritized(self, batch, is_weights, tree_idxs, policy_dqn, target_dqn, memory):
        states, actions, rewards, next_states, dones = batch

        with torch.no_grad():
            target_q = rewards + (1 - dones) * self.discount_factor_g * target_dqn(next_states).max(dim=1)[0]

        current_q = policy_dqn(states).gather(dim=1, index=actions.long()).squeeze()
        td_errors = (target_q - current_q).detach()
        loss = (is_weights.squeeze() * (td_errors ** 2)).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        memory.update_priorities(tree_idxs, td_errors.abs().cpu())


if __name__ == "__main__":
    pass
    # agent = Agent("lunar_lander")
    # agent.run(render=True)  
