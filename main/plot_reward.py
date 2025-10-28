"""
Script for visualizing reward distribution
Visualizes reward distribution for MDD and NC patients across folds using histograms, KDE, and relationships with window size/step ratio.
"""

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

# Visualization style settings
# Use 'seaborn-v0_8-whitegrid' or default 'whitegrid' in newer versions
try:
    plt.style.use('seaborn-v0_8-whitegrid')  # Latest version
    print("Using seaborn-v0_8-whitegrid style")
except:
    try:
        plt.style.use('seaborn')  # Default seaborn style
        print("Using default seaborn style")
    except:
        print("No seaborn style available, using default style")

plt.rcParams['font.size'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['axes.grid'] = True  # Enable grid display

# Create image save directory
os.makedirs('images', exist_ok=True)

# Global constants
N_FOLDS = 5
GROUP_NAMES = {
    'mdd': 'MDD',
    'nc': 'NC'
}
GROUP_COLORS = {
    'mdd': 'red',
    'nc': 'blue'
}


def load_reward_data():
    """
    Load reward data for MDD and NC patients for each fold.
    Returns:
        reward_mdd: Fold-wise reward data for MDD patients
        reward_nc: Fold-wise reward data for NC patients
    """
    try:
        with open('fold_reward_mdd.pkl', 'rb') as f:
            reward_mdd = pickle.load(f)
        with open('fold_reward_nc.pkl', 'rb') as f:
            reward_nc = pickle.load(f)
        
        print(f"Reward data loading completed")
        return reward_mdd, reward_nc
    except (FileNotFoundError, EOFError) as e:
        print(f"Failed to load reward data: {e}")
        # Return empty data
        return defaultdict(dict), defaultdict(dict)


def create_reward_dataframe(reward_mdd, reward_nc):
    """
    Convert reward data for MDD and NC patients into a dataframe.
    Returns:
        df: Dataframe containing reward data
    """
    data = []
    
    # Add data for each fold
    for fold_idx in range(1, N_FOLDS+1):
        # Add MDD patient data
        if fold_idx in reward_mdd:
            for pid, rewards in reward_mdd[fold_idx].items():
                for window, step, reward in rewards:
                    data.append({
                        'fold': fold_idx,
                        'group': 'mdd',
                        'patient_id': pid,
                        'window_size': window,
                        'step_ratio': step,
                        'reward': reward
                    })
        
        # Add NC patient data
        if fold_idx in reward_nc:
            for pid, rewards in reward_nc[fold_idx].items():
                for window, step, reward in rewards:
                    data.append({
                        'fold': fold_idx,
                        'group': 'nc',
                        'patient_id': pid,
                        'window_size': window,
                        'step_ratio': step,
                        'reward': reward
                    })
    
    # Convert to dataframe
    df = pd.DataFrame(data)
    
    # Print summary statistics
    print(f"Reward dataframe creation completed")
    print(f"Total {len(df)} data points")
    print(f"Data points per group:")
    print(df['group'].value_counts())
    print(f"Data points per fold:")
    print(df['fold'].value_counts())
    
    return df


def plot_reward_histograms(df):
    """
    Visualize reward histograms for MDD and NC patients.
    Combines data from all folds into a single graph.
    """
    print("Creating reward histogram by combining all fold data...")
    
    plt.figure(figsize=(12, 8))
    
    # Plot MDD and NC groups together
    sns.histplot(
        data=df, 
        x='reward', 
        hue='group',
        palette={'mdd': 'red', 'nc': 'blue'},
        element='step',
        stat='density',
        common_norm=False,
        alpha=0.6,
        bins=20,
        multiple='layer'
    )
    
    plt.title('All Folds - Reward Distribution')
    plt.xlabel('Reward')
    plt.ylabel('Density')
    plt.legend(title='Group', labels=['MDD', 'NC'])
    plt.grid(True, alpha=0.3)
    
    # Set x-axis ticks according to current reward range
    # Correct predictions: 0.25 ~ 0.75 range, Wrong predictions: -0.15 ~ -0.45 range
    plt.xticks([-0.45, -0.3, -0.15, 0, 0.25, 0.5, 0.75])
    
    os.makedirs('images/reward', exist_ok=True)
    plt.savefig('images/reward/all_folds_reward_histogram.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Reward histogram save completed: images/all_folds_reward_histogram.png")


def plot_reward_by_window_step(df):
    """
    Visualize reward distribution by window size and step ratio.
    Combines data from all folds into a single graph.
    """
    print("Creating reward by window/step visualization by combining all fold data...")
    
    # Reward distribution by window size
    plt.figure(figsize=(12, 8))
    sns.boxplot(
        data=df,
        x='window_size',
        y='reward',
        hue='group',
        palette={'mdd': 'red', 'nc': 'blue'}
    )
    plt.title('All Folds - Reward by Window Size')
    plt.xlabel('Window Size')
    plt.ylabel('Reward')
    plt.legend(title='Group', labels=['MDD', 'NC'])
    
    # If too many x-axis values, show only some
    window_ticks = sorted(df['window_size'].unique())
    if len(window_ticks) > 10:
        # Select 10 evenly
        indices = np.linspace(0, len(window_ticks)-1, 10, dtype=int)
        window_ticks = [window_ticks[i] for i in indices]
    plt.xticks(window_ticks)
    
    os.makedirs('images/reward', exist_ok=True)
    plt.savefig('images/reward/all_folds_reward_by_window.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Reward distribution by step ratio
    plt.figure(figsize=(12, 8))
    
    # Convert step ratio to categorical (divide into 10 bins)
    df['step_bin'] = pd.cut(df['step_ratio'], 10)
    
    sns.boxplot(
        data=df,
        x='step_bin',
        y='reward',
        hue='group',
        palette={'mdd': 'red', 'nc': 'blue'}
    )
    plt.title('All Folds - Reward by Step Ratio')
    plt.xlabel('Step Ratio')
    plt.ylabel('Reward')
    plt.legend(title='Group', labels=['MDD', 'NC'])
    plt.xticks(rotation=45)
    
    os.makedirs('images/reward', exist_ok=True)
    plt.savefig('images/reward/all_folds_reward_by_step.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Reward by window/step visualization save completed: images/all_folds_reward_by_*.png")


def plot_reward_heatmap(df):
    """
    Generate reward heatmap by window size and step ratio.
    Combines data from all folds into a single heatmap.
    """
    print("Creating reward heatmap by combining all fold data...")
    
    for group in ['mdd', 'nc']:
        group_df = df[df['group'] == group]
        
        # Data preparation for heatmap (binning if too many window and step values)
        # Divide window size into 10 bins
        group_df['window_bin'] = pd.cut(group_df['window_size'], 10)
        # Divide step ratio into 10 bins
        group_df['step_bin'] = pd.cut(group_df['step_ratio'], 10)
        
        # Calculate mean reward for each bin
        pivot = group_df.pivot_table(
            values='reward', 
            index='window_bin', 
            columns='step_bin', 
            aggfunc='mean',
            observed=True  # Parameter to resolve FutureWarning
        )
        
        plt.figure(figsize=(14, 10))
        sns.heatmap(
            pivot, 
            cmap='RdBu_r', 
            center=0,
            annot=False,
            fmt='.2f',
            cbar_kws={'label': 'Mean Reward'}
        )
        
        plt.title(f'All Folds - {GROUP_NAMES[group]} Reward Heatmap')
        plt.xlabel('Step Ratio')
        plt.ylabel('Window Size')
        
        os.makedirs('images/reward', exist_ok=True)
        plt.savefig(f'images/reward/all_folds_{group}_reward_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    print("Reward heatmap save completed: images/all_folds_*_reward_heatmap.png")


def plot_reward_statistics(df):
    """
    Visualize reward statistics comparison between MDD and NC groups.
    Combines data from all folds to generate group-wise statistics.
    """
    print("Creating reward statistics visualization by combining all fold data...")
    
    # 1. Calculate mean reward by group
    group_stats = df.groupby(['group'])['reward'].agg(['mean', 'std', 'count']).reset_index()
    
    # 2. Fill mean reward values by group
    labels = [GROUP_NAMES[g] for g in group_stats['group']]
    means = group_stats['mean'].values
    stds = group_stats['std'].values
    counts = group_stats['count'].values
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(
        labels,
        means,
        yerr=stds,
        color=['red', 'blue'],
        alpha=0.7,
        capsize=10
    )
    
    # Show values
    for i, (mean, count) in enumerate(zip(means, counts)):
        plt.text(i, mean + (0.05 if mean > 0 else -0.05), 
                 f'Mean: {mean:.3f}\nN: {count}', 
                 ha='center', va='center', fontweight='bold')
    
    plt.title('Average Reward by Group (All Folds)')
    plt.ylabel('Mean Reward')
    plt.grid(True, alpha=0.3, axis='y')
    plt.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    
    # Set range
    plt.ylim(-1, 1)
    
    os.makedirs('images/reward', exist_ok=True)
    plt.savefig('images/reward/all_folds_reward_statistics.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Additional visualization: Relationship between window size, step ratio, and mean reward
    plt.figure(figsize=(12, 6))
    
    # Categorize window size range
    df_copy = df.copy()  # Copy to resolve warning message
    df_copy['window_range'] = pd.cut(df_copy['window_size'], 5, labels=['Very Small', 'Small', 'Medium', 'Large', 'Very Large'])
    df_copy['step_range'] = pd.cut(df_copy['step_ratio'], 5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
    
    # Mean reward by window size range
    window_stats = df_copy.groupby(['window_range', 'group'])['reward'].mean().reset_index()
    
    plt.subplot(1, 2, 1)
    sns.barplot(
        data=window_stats,
        x='window_range',
        y='reward',
        hue='group',
        palette={'mdd': 'red', 'nc': 'blue'},
        errorbar=None
    )
    plt.title('Average Reward by Window Size Range')
    plt.xlabel('Window Size Range')
    plt.ylabel('Mean Reward')
    plt.legend(title='Group', labels=['MDD', 'NC'])
    plt.grid(True, alpha=0.3, axis='y')
    plt.xticks(rotation=45)
    
    # Mean reward by step ratio range
    step_stats = df_copy.groupby(['step_range', 'group'])['reward'].mean().reset_index()
    
    plt.subplot(1, 2, 2)
    sns.barplot(
        data=step_stats,
        x='step_range',
        y='reward',
        hue='group',
        palette={'mdd': 'red', 'nc': 'blue'},
        errorbar=None
    )
    plt.title('Average Reward by Step Ratio Range')
    plt.xlabel('Step Ratio Range')
    plt.ylabel('Mean Reward')
    plt.legend(title='Group', labels=['MDD', 'NC'])
    plt.grid(True, alpha=0.3, axis='y')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    os.makedirs('images/reward', exist_ok=True)
    plt.savefig('images/reward/all_folds_reward_by_ranges.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Reward statistics visualization save completed: images/all_folds_reward_statistics.png")
    print("Reward range statistics visualization save completed: images/all_folds_reward_by_ranges.png")


def main():
    """
    Main function for visualizing reward distribution of MDD and NC patients
    """
    print("Starting reward distribution visualization for MDD and NC patients...")
    
    # Load reward data
    reward_mdd, reward_nc = load_reward_data()
    
    # Check if data is empty
    has_data = False
    for fold_idx in range(1, N_FOLDS+1):
        if ((fold_idx in reward_mdd and reward_mdd[fold_idx]) or 
            (fold_idx in reward_nc and reward_nc[fold_idx])):
            has_data = True
            break
    
    if not has_data:
        print("Warning: No reward data to visualize.")
        return
    
    # 데이터프레임 생성
    df = create_reward_dataframe(reward_mdd, reward_nc)
    
    if len(df) == 0:
        print("Warning: No reward data to visualize.")
        return
    
    # 시각화 수행
    plot_reward_histograms(df)
    plot_reward_by_window_step(df)
    plot_reward_heatmap(df)
    plot_reward_statistics(df)
    
    print("Reward distribution visualization completed. Check 'images' folder for results.")


if __name__ == "__main__":
    main()
