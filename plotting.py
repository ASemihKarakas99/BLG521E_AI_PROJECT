import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_comparison_cartpole():
    # Base directory containing results
    base_dir = 'training_results/cartpole'
    
    # Buffer types to compare
    buffer_types = ['uniform', 'prioritized', 'attentive', 'baer']
    
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
    os.makedirs('plots/cartpole', exist_ok=True)
    plt.savefig('plots/cartpole/buffer_comparison_cartpole.png')
    plt.close()


def plot_comparison_cartpole_smoothed():
    base_dir = 'training_results/cartpole'
    buffer_types = ['uniform', 'prioritized', 'attentive', 'baer']
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

    # os.makedirs('plots/cartpole', exist_ok=True)
    plt.savefig('plots/cartpole/buffer_comparison_cartpole_smooth.png')
    plt.close()



import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_comparison_lunar():
    base_dir = 'training_results/lunar_lander'
    buffer_types = ['uniform', 'prioritized', 'attentive']
    window = 50  # Rolling window size

    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    for buffer_type in buffer_types:
        rewards_file = os.path.join(base_dir, buffer_type, 'rewards.csv')
        if os.path.exists(rewards_file):
            df = pd.read_csv(rewards_file)
            df = df.iloc[:3000]

            # Clean data
            df['reward'].replace([np.inf, -np.inf], np.nan, inplace=True)
            df['reward'].fillna(method='ffill', inplace=True)
            df['reward'] = df['reward'].clip(lower=-500, upper=500)

            # Rolling stats
            rolling_mean = df['reward'].rolling(window=window, min_periods=10).mean()
            rolling_std = df['reward'].rolling(window=window, min_periods=10).std()
            rolling_max = df['reward'].rolling(window=window, min_periods=10).max()

            # Top plot: Rolling average and confidence band
            axes[0].plot(df['episode'], rolling_mean, label=buffer_type.capitalize())
            axes[0].fill_between(df['episode'],
                                 rolling_mean - rolling_std,
                                 rolling_mean + rolling_std,
                                 alpha=0.2)

            # Bottom plot: Rolling max (dashed)
            axes[1].plot(df['episode'], rolling_max, label=buffer_type.capitalize(), linestyle='--', alpha=0.7)

    # Top axis formatting
    axes[0].set_title('Rolling Average Reward (with Std Dev) - Lunar Lander')
    axes[0].set_ylabel(f'Average Reward (window={window})')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Bottom axis formatting
    axes[1].set_title('Rolling Max Reward - Lunar Lander')
    axes[1].set_xlabel('Episode')
    axes[1].set_ylabel(f'Max Reward (window={window})')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs('plots/lunar_lander', exist_ok=True)
    plt.savefig('plots/lunar_lander/buffer_comparison_lunar_split.png')
    plt.close()

    base_dir = 'training_results/lunar_lander'
    buffer_types = ['uniform', 'prioritized', 'attentive']
    window = 50  # Reduced rolling window for faster learning trends

    plt.figure(figsize=(10, 6))

    for buffer_type in buffer_types:
        rewards_file = os.path.join(base_dir, buffer_type, 'rewards.csv')
        if os.path.exists(rewards_file):
            df = pd.read_csv(rewards_file)
            df = df.iloc[:3000]  # Trim to 3000 episodes

            # Replace infs and fill NaNs
            df['reward'].replace([np.inf, -np.inf], np.nan, inplace=True)
            df['reward'].fillna(method='ffill', inplace=True)

            # Clip outliers (allow more negative rewards)
            df['reward'] = df['reward'].clip(lower=-500, upper=500)

            # Rolling stats
            rolling_mean = df['reward'].rolling(window=window, min_periods=10).mean()
            rolling_std = df['reward'].rolling(window=window, min_periods=10).std()
            rolling_max = df['reward'].rolling(window=window, min_periods=10).max()

            # Plot mean and confidence band
            plt.plot(df['episode'], rolling_mean, label=buffer_type.capitalize())
            plt.fill_between(df['episode'],
                             rolling_mean - rolling_std,
                             rolling_mean + rolling_std,
                             alpha=0.2)

            # Plot max reward as dashed line
            plt.plot(df['episode'], rolling_max, linestyle='--', alpha=0.5)

    plt.title('Comparison of Buffer Types on Lunar Lander')
    plt.xlabel('Episode')
    plt.ylabel(f'Average Reward ({window}-episode window)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    os.makedirs('plots/lunar_lander', exist_ok=True)
    plt.savefig('plots/lunar_lander/buffer_comparison_lunar.png')
    plt.close()



def plot_comparison_lunar_smoothed():
    # Base directory containing results
    base_dir = 'training_results/lunar_lander'
    buffer_types = ['uniform', 'prioritized', 'attentive']

    plt.figure(figsize=(10, 6))

    for buffer_type in buffer_types:
        rewards_file = os.path.join(base_dir, buffer_type, 'rewards.csv')
        if os.path.exists(rewards_file):
            df = pd.read_csv(rewards_file)
            df = df.iloc[:3000]

            # Optional: Filter out extremely negative spikes
            df = df[df['reward'] >= -200]

            # Use rolling average and std (smoothing)
            window = 100
            rolling_mean = df['reward'].rolling(window=window, min_periods=1).mean()
            rolling_std = df['reward'].rolling(window=window, min_periods=1).std()

            # Plot the rolling mean
            plt.plot(df['episode'], rolling_mean, label=buffer_type.capitalize())

            # Plot confidence interval (mean ± std)
            plt.fill_between(df['episode'],
                             rolling_mean - rolling_std,
                             rolling_mean + rolling_std,
                             alpha=0.2)

    plt.title('Comparison of Buffer Types on Lunar Lander')
    plt.xlabel('Episode')
    plt.ylabel('Average Reward (100 episode window)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    os.makedirs('plots/lunar_lander', exist_ok=True)
    plt.savefig('plots/lunar_lander/buffer_comparison_lunar_smooth.png')
    plt.close()


if __name__ == "__main__":
    plot_comparison_cartpole()
    plot_comparison_cartpole_smoothed()
    # plot_comparison_lunar()
    # plot_comparison_lunar_smoothed()