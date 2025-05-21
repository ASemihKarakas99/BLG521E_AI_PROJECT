import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_comparison_cartpole():
    # Base directory containing results
    base_dir = 'training_results/cartpole'
    
    # Buffer types to compare
    buffer_types = ['uniform', 'prioritized', 'attentive']
    
    # Setup plot
    plt.figure(figsize=(10, 6))
    
    # Plot each buffer type
    for buffer_type in buffer_types:
        # Read rewards CSV
        rewards_file = os.path.join(base_dir, buffer_type, 'rewards.csv')
        if os.path.exists(rewards_file):
            df = pd.read_csv(rewards_file)
            
            # Limit to first 6000 episodes
            df = df.iloc[:6000]
            
            # Calculate rolling average
            rolling_avg = df['reward'].rolling(window=100, min_periods=1).mean()
            
            # Plot rolling average
            plt.plot(df['episode'], rolling_avg, label=buffer_type.capitalize())
    
    plt.title('Comparison of Buffer Types on CartPole')
    plt.xlabel('Episode')
    plt.ylabel('Average Reward (100 episode window)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save plot
    plt.savefig('plots/buffer_comparison_cartpole.png')
    plt.close()


def plot_comparison_cartpole_smoothed():
    base_dir = 'training_results/cartpole'
    buffer_types = ['uniform', 'prioritized', 'attentive']
    plt.figure(figsize=(10, 6))

    bin_size = 100  # Episodes per bin

    for buffer_type in buffer_types:
        rewards_file = os.path.join(base_dir, buffer_type, 'rewards.csv')
        if os.path.exists(rewards_file):
            df = pd.read_csv(rewards_file)
            df = df.iloc[:6000]

            # Add a column to identify bins
            df['bin'] = df['episode'] // bin_size

            # Group by bin and calculate mean and std
            grouped = df.groupby('bin')['reward']
            mean_rewards = grouped.mean()
            std_rewards = grouped.std()

            # X axis = bin center
            x = (mean_rewards.index + 0.5) * bin_size

            # Plot with shaded standard deviation
            plt.plot(x, mean_rewards, label=buffer_type.capitalize())
            plt.fill_between(x, mean_rewards - std_rewards, mean_rewards + std_rewards, alpha=0.2)

    plt.title('Comparison of Buffer Types on CartPole')
    plt.xlabel('Episode')
    plt.ylabel('Reward (mean ± std over bins of 100 episodes)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    os.makedirs('plots', exist_ok=True)
    plt.savefig('plots/buffer_comparison_cartpole_smooth.png')
    plt.close()




if __name__ == "__main__":
    plot_comparison_cartpole()
    plot_comparison_cartpole_smoothed()