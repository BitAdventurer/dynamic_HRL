#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

# Create images directory if it doesn't exist
os.makedirs('images', exist_ok=True)
os.makedirs('images/ablation', exist_ok=True)

# --- Set global plot parameters ---
matplotlib.rcParams['font.size'] = 18
matplotlib.rcParams['axes.titlesize'] = 18
matplotlib.rcParams['axes.labelsize'] = 18
matplotlib.rcParams['legend.fontsize'] = 18
matplotlib.rcParams['xtick.labelsize'] = 18
matplotlib.rcParams['ytick.labelsize'] = 18

# Fix the x-axis formatter to avoid 'st', 'nd', 'rd', 'th' suffixes
def format_ticks(ax):
    """Apply a simple numeric formatter to axis ticks"""
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f"{int(x)}" if x == int(x) else f"{x:.1f}"))

def plot_ablation_performance():
    """
    Plot the performance metrics for different ablation modes.
    This includes accuracy, sensitivity, specificity, F1 score, AUC, and reward.
    """
    print("시각화 이미지가 'images/ablation' 폴더에 저장되었습니다.")
    
    # Read ablation stats data
    stats_path = 'ablation_results/ablation_stats.csv'
    if not os.path.exists(stats_path):
        print(f"Error: {stats_path} not found. Run the ablation script first.")
        return
    
    # Read the stats data
    stats_df = pd.read_csv(stats_path)
    
    # Check if all required columns exist
    required_cols = ['Mode', 'AccMean', 'AccStd', 'F1Mean', 'F1Std', 'AUCMean', 'AUCStd']
    if not all(col in stats_df.columns for col in required_cols):
        print(f"Error: Missing required columns in {stats_path}")
        return
    
    # Check if reward data exists
    has_reward = 'RewardMean' in stats_df.columns and 'RewardStd' in stats_df.columns
    
    # Define mode names and colors for plotting
    mode_names = {
        'full': 'Full HRL',
        'no_macro': 'No Macro',
        'no_micro': 'No Micro',
        'fixed': 'Fixed'
    }
    
    mode_colors = {
        'full': 'green',
        'no_macro': 'blue',
        'no_micro': 'orange',
        'fixed': 'red'
    }
    
    # Get modes present in the dataframe
    modes = stats_df['Mode'].unique()
    
    # Add nice mode labels
    stats_df['ModeLabel'] = stats_df['Mode'].map(lambda x: mode_names.get(x, x))
    
    # Plot performance metrics (Acc, F1, AUC)
    metrics = [
        {'col': 'AccMean', 'std': 'AccStd', 'title': 'Accuracy', 'ylabel': 'Accuracy'},
        {'col': 'F1Mean', 'std': 'F1Std', 'title': 'F1 Score', 'ylabel': 'F1 Score'},
        {'col': 'AUCMean', 'std': 'AUCStd', 'title': 'AUC', 'ylabel': 'AUC'}
    ]
    
    if has_reward:
        metrics.append({
            'col': 'RewardMean', 'std': 'RewardStd', 
            'title': 'Average Reward', 'ylabel': 'Reward'
        })
    
    # 1. Bar chart with all metrics
    plt.figure(figsize=(16, 10))
    
    # Set up the bar positions
    num_metrics = len(metrics)
    num_modes = len(modes)
    bar_width = 0.2
    index = np.arange(num_metrics)
    
    # Create bars for each mode
    for i, mode in enumerate(modes):
        mode_df = stats_df[stats_df['Mode'] == mode]
        if len(mode_df) == 0:
            continue
            
        values = [mode_df[m['col']].values[0] for m in metrics]
        errs = [mode_df[m['std']].values[0] for m in metrics]
        pos = index + (i - num_modes/2 + 0.5) * bar_width
        
        plt.bar(pos, values, bar_width, 
                label=mode_names.get(mode, mode),
                color=mode_colors.get(mode, 'gray'),
                yerr=errs, capsize=5, alpha=0.7)
    
    # Add labels and legend
    plt.xlabel('Metrics')
    plt.ylabel('Score')
    plt.title('Ablation Study Performance Comparison')
    plt.xticks(index, [m['title'] for m in metrics])
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('images/ablation/metrics_comparison.png', dpi=300)
    plt.close()
    
    # 2. Individual plots for each metric
    for metric in metrics:
        plt.figure(figsize=(10, 6))
        
        # Extract data for this metric
        metric_data = []
        for mode in modes:
            mode_df = stats_df[stats_df['Mode'] == mode]
            if len(mode_df) == 0:
                continue
                
            mean_val = mode_df[metric['col']].values[0]
            std_val = mode_df[metric['std']].values[0]
            mode_label = mode_names.get(mode, mode)
            mode_color = mode_colors.get(mode, 'gray')
            
            metric_data.append({
                'mode': mode, 
                'mode_label': mode_label,
                'mean': mean_val, 
                'std': std_val,
                'color': mode_color
            })
        
        # Sort by performance (higher is better)
        metric_data.sort(key=lambda x: x['mean'], reverse=True)
        
        # Create the bar chart
        x_pos = range(len(metric_data))
        plt.bar(x_pos, 
                [d['mean'] for d in metric_data], 
                yerr=[d['std'] for d in metric_data],
                align='center', 
                alpha=0.7, 
                capsize=10,
                color=[d['color'] for d in metric_data])
        
        # Add text labels on top of the bars
        for i, d in enumerate(metric_data):
            plt.text(i, d['mean'] + d['std'] + 0.01, 
                    f"{d['mean']:.3f}", 
                    ha='center', va='bottom', fontsize=14)
        
        plt.xlabel('Model Variant')
        plt.ylabel(metric['ylabel'])
        plt.title(f'Ablation Study: {metric["title"]} Comparison')
        plt.xticks(x_pos, [d['mode_label'] for d in metric_data])
        
        # Format y-axis to avoid decorations
        format_ticks(plt.gca())
        
        plt.tight_layout()
        plt.savefig(f'images/ablation/{metric["col"]}_comparison.png', dpi=300)
        plt.close()
    
    print(f"Ablation plots saved to 'images/ablation' directory!")

def plot_epoch_performance_trends():
    """
    Plot performance metrics per epoch for different ablation modes.
    This creates line graphs showing how metrics change over epochs.
    """
    print("시각화 이미지가 'images/ablation' 폴더에 저장되었습니다.")
    
    # Define mode directories and their display names
    mode_dirs = {
        'full_hrl': 'Full HRL',
        'no_macro': 'No Macro',
        'no_micro': 'No Micro',
        'fixed': 'Fixed Parameters'
    }
    
    mode_colors = {
        'full_hrl': 'green',
        'no_macro': 'blue',
        'no_micro': 'orange',
        'fixed': 'red'
    }
    
    # Dictionary to store data for each mode
    mode_data = {}
    
    # Load data for each mode
    for mode, display_name in mode_dirs.items():
        file_path = f'ablation_results/{mode}/training_val_test_results_allfolds.csv'
        
        if not os.path.exists(file_path):
            print(f"Warning: File not found: {file_path}")
            continue
            
        try:
            # Load results data
            print(f"Reading file: {file_path}")
            df = pd.read_csv(file_path)
            print(f"  * Loaded {len(df)} rows from {mode}")
            print(f"  * Sample columns: {list(df.columns)[:5]}")
            
            # Check if required columns exist
            required_cols = ['Fold', 'Epoch']
            for metric in ['TestAcc', 'TestF1', 'TestAUC', 'TrainReward']:
                if metric not in df.columns:
                    print(f"  * WARNING: Column '{metric}' not found in {mode} data!")
            
            # Get the maximum number of epochs across all folds
            try:
                max_epochs = df.groupby('Fold')['Epoch'].max().min()  # Use min to ensure all folds have the same epochs
                print(f"  * Max epochs: {max_epochs}")
                
                # Filter out any epochs beyond the minimum max_epochs
                df = df[df['Epoch'] <= max_epochs]
                
                # Calculate mean and std across folds for each epoch
                mean_df = df.groupby('Epoch').mean(numeric_only=True).reset_index()
                std_df = df.groupby('Epoch').std(numeric_only=True).reset_index()
                print(f"  * Successfully processed {mode} data, found {len(mean_df)} epochs")
            except Exception as e:
                print(f"  * ERROR processing {mode} data: {e}")
                raise
            
            # Store the mode data
            mode_data[mode] = {
                'epochs': mean_df['Epoch'].values,
                'mean_df': mean_df,
                'std_df': std_df,
                'display_name': display_name,
                'color': mode_colors.get(mode, 'gray')
            }
            
        except Exception as e:
            print(f"Error loading data for {mode}: {e}")
    
    # If no data was loaded, return
    if not mode_data:
        print("No epoch data found for any ablation mode.")
        return
        
    # Create directory for processed data
    os.makedirs('processed_data', exist_ok=True)
    
    # Define metrics to plot
    metrics = [
        {'col': 'TestAcc', 'title': 'Test Accuracy', 'ylabel': 'Accuracy'},
        {'col': 'TestF1', 'title': 'Test F1 Score', 'ylabel': 'F1 Score'},
        {'col': 'TestAUC', 'title': 'Test AUC', 'ylabel': 'AUC'},
        {'col': 'TrainReward', 'title': 'Train Reward', 'ylabel': 'Reward'}
    ]
    
    # Create dataframes to store processed data for each metric
    metric_data_dict = {metric['col']: pd.DataFrame() for metric in metrics}
    
    # Plot each metric
    for metric in metrics:
        plt.figure(figsize=(12, 8))
        
        # Dataframe for storing processed data for this metric
        processed_df = pd.DataFrame()
        
        # Plot line for each mode
        for mode, data in mode_data.items():
            if metric['col'] not in data['mean_df'].columns:
                print(f"Column {metric['col']} not found for {mode}")
                continue
                
            try:
                if metric['col'] not in data['mean_df'].columns:
                    print(f"Column {metric['col']} not found for {mode}")
                    continue
                    
                mean_vals = data['mean_df'][metric['col']].values
                std_vals = data['std_df'][metric['col']].values
                epochs = data['epochs']
                
                print(f"Plotting {mode} {metric['title']} with {len(epochs)} points")
                
                # Plot mean line
                plt.plot(epochs, mean_vals, 
                        label=data['display_name'],
                        color=data['color'],
                        linewidth=2)
                        
                # Plot confidence interval
                plt.fill_between(epochs,
                                mean_vals - std_vals,
                                mean_vals + std_vals,
                                alpha=0.2,
                                color=data['color'])
            except Exception as e:
                print(f"Error plotting {mode} for {metric['title']}: {e}")
                continue
            
            # Store the processed data in a dataframe
            mode_df = pd.DataFrame({
                'Epoch': epochs,
                f'{mode}_Mean': mean_vals,
                f'{mode}_Std': std_vals
            })
            
            # Merge into the processed dataframe
            if processed_df.empty:
                processed_df = mode_df
            else:
                processed_df = pd.merge(processed_df, mode_df, on='Epoch', how='outer')
        
        plt.title(f'Ablation Study: {metric["title"]} vs Epoch', fontsize=20)
        plt.xlabel('Epoch', fontsize=18)
        plt.ylabel(metric['ylabel'], fontsize=18)
        plt.legend(fontsize=16)
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Format axis ticks
        format_ticks(plt.gca())
        plt.tight_layout()
        
        # Save figure
        plt.savefig(f'images/ablation/{metric["col"]}_vs_epoch.png', dpi=300)
        plt.close()
        
        # Save processed data to CSV
        processed_df.to_csv(f'processed_data/ablation_{metric["col"]}_data.csv', index=False)
        print(f"Processed data for {metric['title']} saved to processed_data/ablation_{metric['col']}_data.csv")
    
    print("Epoch-by-epoch performance plots saved to 'images/ablation' directory!")

def main():
    # 에폭 관련 그래프만 생성 (막대 그래프는 생성하지 않음)
    # plot_ablation_performance()  # 주석 처리하여 막대 그래프는 생성하지 않음
    plot_epoch_performance_trends()

if __name__ == '__main__':
    main()
