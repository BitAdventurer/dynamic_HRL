#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

def plot_full_distribution_scatter(N_FOLDS=5):
    """
    Create scatter plot displaying all window size and step ratio data points individually
    (Uses all individual data points instead of patient-wise averages)
    """
    # Load existing pickle files - with exception handling
    try:
        with open('fold_window_usage_mdd.pkl','rb') as f:
            fold_window_usage_mdd = pickle.load(f)
        with open('fold_step_usage_mdd.pkl','rb') as f:
            fold_step_usage_mdd = pickle.load(f)
        with open('fold_window_usage_nc.pkl','rb') as f:
            fold_window_usage_nc = pickle.load(f)
        with open('fold_step_usage_nc.pkl','rb') as f:
            fold_step_usage_nc = pickle.load(f)
        # Additionally load reward data (already saved as window, step, reward tuples)
        with open('fold_reward_mdd.pkl','rb') as f:
            fold_reward_mdd = pickle.load(f)
        with open('fold_reward_nc.pkl','rb') as f:
            fold_reward_nc = pickle.load(f)
    except (FileNotFoundError, EOFError) as e:
        print(f"Failed to load data file: {e}")
        print("Please run train.py first to generate the data.")
        return

    # Extract MDD data points (all individual data)
    mdd_points = []
    for fidx in range(1, N_FOLDS+1):
        fold_data = fold_reward_mdd.get(fidx, fold_reward_mdd.get(str(fidx), {}))
        for pid, data_list in fold_data.items():
            for w, s, r in data_list:
                mdd_points.append((w, s, r))  # window size, step ratio, reward

    # Extract NC data points (all individual data)
    nc_points = []
    for fidx in range(1, N_FOLDS+1):
        fold_data = fold_reward_nc.get(fidx, fold_reward_nc.get(str(fidx), {}))
        for pid, data_list in fold_data.items():
            for w, s, r in data_list:
                nc_points.append((w, s, r))  # window size, step ratio, reward

    print(f"Number of MDD data points: {len(mdd_points)}")
    print(f"Number of NC data points: {len(nc_points)}")

    # Output descriptive statistics for overall distribution
    df_mdd = pd.DataFrame(mdd_points, columns=['WindowSize', 'StepRatio', 'Reward'])
    df_nc = pd.DataFrame(nc_points, columns=['WindowSize', 'StepRatio', 'Reward'])
    
    print("\nMDD data statistics:")
    print(df_mdd[['WindowSize', 'StepRatio']].describe())
    
    print("\nNC data statistics:")
    print(df_nc[['WindowSize', 'StepRatio']].describe())

    # 1) Overall MDD Distribution Scatter Plot
    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        data=df_mdd,
        x='WindowSize',
        y='StepRatio',
        color='red',
        alpha=0.3,
        edgecolor=None,
        s=20  # Set small point size (due to many data points)
    )
    plt.title("MDD Full Distribution (All Data Points)")
    plt.xlabel("Window Size")
    plt.ylabel("Step Ratio")
    plt.tight_layout()
    os.makedirs('images', exist_ok=True)
    os.makedirs('images/full_distribution', exist_ok=True)
    plt.savefig('images/full_distribution/mdd_full_scatter.png', dpi=300)
    plt.close()

    # 2) Overall NC Distribution Scatter Plot
    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        data=df_nc,
        x='WindowSize',
        y='StepRatio',
        color='blue',
        alpha=0.3,
        edgecolor=None,
        s=20  # Set small point size
    )
    plt.title("NC Full Distribution (All Data Points)")
    plt.xlabel("Window Size")
    plt.ylabel("Step Ratio")
    plt.tight_layout()
    plt.savefig('images/full_distribution/nc_full_scatter.png', dpi=300)
    plt.close()

    # 3) MDD and NC Overlapped Scatter Plot
    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        data=df_mdd,
        x='WindowSize',
        y='StepRatio',
        color='red',
        alpha=0.3,
        edgecolor=None,
        s=20,
        label='MDD'
    )
    sns.scatterplot(
        data=df_nc,
        x='WindowSize',
        y='StepRatio',
        color='blue',
        alpha=0.3,
        edgecolor=None,
        s=20,
        label='NC'
    )
    plt.title("MDD vs NC Full Distribution (All Data Points)")
    plt.xlabel("Window Size")
    plt.ylabel("Step Ratio")
    plt.legend()
    plt.tight_layout()
    plt.savefig('images/full_distribution/mdd_nc_full_scatter.png', dpi=300)
    plt.close()

    # 4) Express as 2D histogram (density map)
    plt.figure(figsize=(16, 6))
    
    # MDD 2D histogram
    plt.subplot(1, 2, 1)
    h_mdd = plt.hist2d(df_mdd['WindowSize'], df_mdd['StepRatio'], 
               bins=[50, 20], cmap='Reds', alpha=0.8, density=True)
    plt.colorbar(label='Density')
    plt.title("MDD Distribution (Density)")
    plt.xlabel("Window Size")
    plt.ylabel("Step Ratio")
    
    # NC 2D histogram
    plt.subplot(1, 2, 2)
    h_nc = plt.hist2d(df_nc['WindowSize'], df_nc['StepRatio'], 
               bins=[50, 20], cmap='Blues', alpha=0.8, density=True)
    plt.colorbar(label='Density')
    plt.title("NC Distribution (Density)")
    plt.xlabel("Window Size")
    plt.ylabel("Step Ratio")
    
    plt.tight_layout()
    plt.savefig('images/full_distribution/density_histogram.png', dpi=300)
    plt.close()

    # 5) Reward-based scatter plot (reward shown as color)
    plt.figure(figsize=(16, 6))
    
    # MDD Reward scatter plot
    plt.subplot(1, 2, 1)
    scatter_mdd = plt.scatter(df_mdd['WindowSize'], df_mdd['StepRatio'], 
                   c=df_mdd['Reward'], cmap='RdYlGn', alpha=0.5, s=15)
    plt.colorbar(scatter_mdd, label='Reward')
    plt.title("MDD - Window Size vs Step Ratio (Reward)")
    plt.xlabel("Window Size")
    plt.ylabel("Step Ratio")
    
    # NC Reward scatter plot
    plt.subplot(1, 2, 2)
    scatter_nc = plt.scatter(df_nc['WindowSize'], df_nc['StepRatio'], 
                   c=df_nc['Reward'], cmap='RdYlGn', alpha=0.5, s=15)
    plt.colorbar(scatter_nc, label='Reward')
    plt.title("NC - Window Size vs Step Ratio (Reward)")
    plt.xlabel("Window Size")
    plt.ylabel("Step Ratio")
    
    plt.tight_layout()
    plt.savefig('images/full_distribution/reward_scatter.png', dpi=300)
    plt.close()
    
    print("Full distribution visualization completed. Check the 'images/full_distribution/' folder.")

if __name__ == "__main__":
    plot_full_distribution_scatter()
