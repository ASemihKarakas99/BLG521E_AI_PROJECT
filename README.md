# BLG521E AI Project


# Project Overview

This project focuses on implementing and comparing various experience replay buffer strategies in reinforcement learning, specifically within the context of Deep Q-Networks (DQN). The primary goal is to enhance the learning efficiency and performance of agents in different environments by utilizing advanced buffer techniques. The project includes implementations for standard experience replay, prioritized experience replay (PER), and attentive experience replay (AER), with a significant contribution to the development of Balanced Attentive Experience Replay (BAER).

## Balanced Attentive Experience Replay (BAER) Improvements

BAER is an innovative approach designed to improve the sampling efficiency of experience replay buffers. Traditional experience replay methods often sample experiences uniformly, which can lead to inefficient learning due to the inclusion of redundant or less informative experiences. BAER addresses this by incorporating an attention mechanism that prioritizes experiences based on their relevance to the current state of the agent.

Key improvements introduced by BAER include:

- **Diversity and Relevance Balance**: BAER maintains a balance between sampling diverse experiences and those that are most relevant to the agent's current state. This is achieved by dynamically adjusting the sampling strategy to include both attentive and diverse samples, enhancing the learning process.

- **Adaptive Sampling Strategy**: The buffer uses a lambda factor to control the proportion of attentive versus random samples, allowing for flexibility in how experiences are prioritized. This adaptability ensures that the buffer can cater to different learning phases and environments.

- **Integration with DQN**: BAER is seamlessly integrated with the DQN architecture, utilizing the network's embeddings to compute similarities between experiences. This integration allows for more informed sampling decisions, leveraging the network's understanding of the environment.

- **Improved Learning Performance**: By focusing on more informative experiences, BAER has shown to improve the convergence speed and stability of learning in complex environments like Lunar Lander and CartPole.

Overall, BAER represents a significant advancement in experience replay strategies, offering a more nuanced and effective approach to experience sampling in reinforcement learning.

## Project Components

- **Buffers**: Implementation of various buffer types including uniform, prioritized, and attentive buffers, with BAER as a key enhancement.
- **Agent**: A flexible agent class capable of operating in different environments and utilizing different buffer strategies.
- **Environments**: Support for multiple environments such as CartPole and Lunar Lander, with plans to include additional environments like FlappyBird and Atari games.
- **DQN Architecture**: A robust DQN implementation that aligns with state-of-the-art practices, supporting the integration of advanced buffer strategies.
- **Plotting and Analysis**: Tools for visualizing and analyzing the performance of different buffer strategies across various environments.

This project serves as a comprehensive exploration of experience replay techniques, providing valuable insights and tools for researchers and practitioners in the field of reinforcement learning.


## How to Run

To run the project and experiment with different experience replay strategies, follow these steps:

 **Run Experiments**:
   You can run experiments for different environments and buffer types using the command line interface. Here are some examples:

   - **CartPole with Vanilla Experience Replay**:
     ```bash
     python agent.py --hyperparameter_set cartpole --buffer_type uniform
     ```

   - **Lunar Lander with Prioritized Experience Replay**:
     ```bash
     python agent.py --hyperparameter_set lunar_lander --buffer_type prioritized
     ```
   - **Lunar Lander with Balanced Attentive Experience Replay**:
     ```bash
     python agent_baer.py --hyperparameter_set lunar_lander --buffer_type aer
     ```
   - **Lunar Lander with Balanced Attentive Experience Replay**:
     ```bash
     python agent_baer.py --hyperparameter_set lunar_lander --buffer_type baer
     ```

   - **Flappy Bird with Experience Replay**:
     ```bash
     python agent_baer.py --hyperparameter_set flappybird --buffer_type uniform
     ```

 **Visualize Results**:
   After running the experiments, you can visualize the results using the plotting scripts. For example:
   ```bash
   python plotting.py
   ```

## TODO

- Buffers will be updated according to [this repo](https://github.com/Howuhh/prioritized_experience_replay/blob/main/memory/buffer.py)

- Agent class will be more generic:
  - Adding Step size / Infinite loop option [DONE]
  - Choosing Buffer type options [DONE]
  - Choosing Gym envs options [DONE]

- Attentive Experience Replay Class will be added [DONE]

- Contribution to Attentive Experience Replay Class (BAER) will be added

- Additional environments will be added:
  - CartPole [DONE]
  - Lunar Lander [DONE]
  - FlappyBird
  - Various Atari Games

- DQN Archirecture will be adjsuted so that it aligns with paper implementation.

- Tests, Results, and plotting will be added. [DONE]

- Optional: Implementing DDQN style enhanced architectures will be implemented.

## Test Steps:

- Run for Cartpole:
    - Vannila ER [DONE]
    - PER [DONE]
    - AER [DONE]
    - BAER [DONE]

- Run for Lunar Landers :
    - Vannila ER [DONE]
    - PER [DONE]
    - AER [DONE]
    - BAER [DONE]

- Run for FlappyBird:
    - Vannila ER [DONE]
    - PER [DONE]
    - AER [DONE]
    - BAER [DONE]


## References

### Repositories
- [pytorch-dqn](https://github.com/hungtuchen/pytorch-dqn/blob/master/utils/replay_buffer.py)
- [elite_buffer_vtrace](https://github.com/parralplex/elite_buffer_vtrace/blob/main/agent/learner_d/learner.py)
- [dqn_pytorch](https://github.com/johnnycode8/dqn_pytorch/tree/main)
- [Deep-Q-Learning-Network](https://github.com/AleksandarHaber/Deep-Q-Learning-Network-from-Scratch-in-Python-TensorFlow-and-OpenAI-Gym/tree/main)
- [prioritized_experience_replay](https://github.com/Howuhh/prioritized_experience_replay/blob/main/memory/buffer.py)

### YouTube Resources
- [Reinforcement Learning Playlist](https://www.youtube.com/playlist?list=PLO89phzZmnHjYXlCNR_y2qF0gr9x8YpC8)
- [DQN Tutorial by Qasim Wani](https://www.youtube.com/watch?v=cQtGJDS-Rfs)
- [RL Fundamentals](https://www.youtube.com/playlist?list=PL58zEckBH8fCMIVzQCRSZVPUp3ZAVagWi)
- [Stanford RL Lecture](https://www.youtube.com/watch?v=b_wvosA70f8&list=PLoROMvodv4rN4wG6Nk6sNpTEbuOSosZdX&index=5)
- [Steve Brunton RL Tutorial](https://www.youtube.com/watch?v=wDVteayWWvU)
- [DeepMind RL Course](https://www.youtube.com/playlist?list=PLqYmG7hTraZDM-OYHWgPebj2MfCFzFObQ)

### Reading Materials
- [Stanford CS234 Lecture Notes](https://web.stanford.edu/class/cs234/slides/lecture4post.pdf)
- [Reinforcement Learning: An Introduction](https://www.amazon.com/Reinforcement-Learning-Introduction-Adaptive-Computation/dp/0262039249) by Sutton & Barto
