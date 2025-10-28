import pickle
import os
os.makedirs('images', exist_ok=True)
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns


# --- Set global font to Times New Roman, size=12 ---
# plt.rcParams['font.family'] = 'Times New Roman'
matplotlib.rcParams['font.size'] = 18
matplotlib.rcParams['axes.titlesize'] = 18
matplotlib.rcParams['axes.labelsize'] = 18
matplotlib.rcParams['legend.fontsize'] = 18
matplotlib.rcParams['xtick.labelsize'] = 18
matplotlib.rcParams['ytick.labelsize'] = 18

def plot_window_step_hist_per_fold(N_FOLDS):
    # Add exception handling for file loading
    try:
        with open('fold_window_usage_mdd.pkl','rb') as f:
            fold_window_usage_mdd = pickle.load(f)
        with open('fold_window_usage_nc.pkl','rb') as f:
            fold_window_usage_nc = pickle.load(f)
        with open('fold_step_usage_mdd.pkl','rb') as f:
            fold_step_usage_mdd = pickle.load(f)
        with open('fold_step_usage_nc.pkl','rb') as f:
            fold_step_usage_nc = pickle.load(f)
    except (FileNotFoundError, EOFError) as e:
        print(f"Failed to load data file: {e}")
        print("Please run train.py first to generate data.")
        # Return empty data
        from collections import defaultdict
        fold_window_usage_mdd = defaultdict(lambda: defaultdict(dict))
        fold_window_usage_nc = defaultdict(lambda: defaultdict(dict))
        fold_step_usage_mdd = defaultdict(lambda: defaultdict(dict))
        fold_step_usage_nc = defaultdict(lambda: defaultdict(dict))
    
    # Process only folds with data
    available_folds = [idx for idx in range(1, N_FOLDS+1) 
                       if idx in fold_window_usage_mdd and fold_window_usage_mdd[idx]]
    
    if not available_folds:
        print("Warning: No data available in any fold. Check your data collection.")
        return
        
    print(f"Processing {len(available_folds)} folds with data: {available_folds}")
    
    for fidx in available_folds:
        mdd_window_list = []
        mdd_step_list = []
        for pid, wset in fold_window_usage_mdd.get(fidx, {}).items():
            mdd_window_list.extend(wset)
        for pid, sset in fold_step_usage_mdd.get(fidx, {}).items():
            mdd_step_list.extend(sset)

        nc_window_list = []
        nc_step_list = []
        for pid, wset in fold_window_usage_nc.get(fidx, {}).items():
            nc_window_list.extend(wset)
        for pid, sset in fold_step_usage_nc.get(fidx, {}).items():
            nc_step_list.extend(sset)

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # Top-left: MDD Step
        axes[0][0].hist(mdd_step_list, bins=20, color='crimson', alpha=0.7, edgecolor='black')
        axes[0][0].set_title(f'Fold {fidx} - MDD Step')
        axes[0][0].set_xlabel('Step size')
        axes[0][0].set_ylabel('Subjects')

        # Top-right: MDD Window
        axes[0][1].hist(mdd_window_list, bins=20, color='crimson', alpha=0.7, edgecolor='black')
        axes[0][1].set_title(f'Fold {fidx} - MDD Window')
        axes[0][1].set_xlabel('Window size')
        axes[0][1].set_ylabel('Subjects')

        # Bottom-left: NC Step
        axes[1][0].hist(nc_step_list, bins=20, color='navy', alpha=0.7, edgecolor='black')
        axes[1][0].set_title(f'Fold {fidx} - NC Step')
        axes[1][0].set_xlabel('Step size')
        axes[1][0].set_ylabel('Subjects')

        # Bottom-right: NC Window
        axes[1][1].hist(nc_window_list, bins=20, color='navy', alpha=0.7, edgecolor='black')
        axes[1][1].set_title(f'Fold {fidx} - NC Window')
        axes[1][1].set_xlabel('Window size')
        axes[1][1].set_ylabel('Subjects')

        plt.tight_layout()
        os.makedirs('images', exist_ok=True)
        os.makedirs('images/ratio', exist_ok=True)
        plt.savefig(f'images/ratio/fold_{fidx}_mdd_nc_window_step_hist.png', dpi=300)
        plt.close()

    # Overall (MDD/NC) histogram
    overall_mdd_window_list = []
    overall_mdd_step_list   = []
    overall_nc_window_list  = []
    overall_nc_step_list    = []

    for fidx in range(1, N_FOLDS+1):
        for pid, wset in fold_window_usage_mdd.get(fidx, {}).items():
            overall_mdd_window_list.extend(wset)
        for pid, sset in fold_step_usage_mdd.get(fidx, {}).items():
            overall_mdd_step_list.extend(sset)
        for pid, wset in fold_window_usage_nc.get(fidx, {}).items():
            overall_nc_window_list.extend(wset)
        for pid, sset in fold_step_usage_nc.get(fidx, {}).items():
            overall_nc_step_list.extend(sset)

    # Helper for scaling the y-axis if needed
    def scale_yticks(ax):
        yticks = ax.get_yticks()
        ax.set_yticklabels([f'{int(y)}' for y in yticks])  

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    # Overall MDD Step
    axes[0][0].hist(overall_mdd_step_list, bins=30, color='crimson', alpha=0.7, edgecolor='black')
    axes[0][0].set_title('MDD Step size')
    axes[0][0].set_xlabel('Step size')
    axes[0][0].set_ylabel('Subjects')
    scale_yticks(axes[0][0])

    # Overall MDD Window
    axes[0][1].hist(overall_mdd_window_list, bins=45, color='crimson', alpha=0.7, edgecolor='black')
    axes[0][1].set_title('MDD Window size')
    axes[0][1].set_xlabel('Window size')
    axes[0][1].set_ylabel('Subjects')
    scale_yticks(axes[0][1])

    # Overall NC Step
    axes[1][0].hist(overall_nc_step_list, bins=30, color='navy', alpha=0.7, edgecolor='black')
    axes[1][0].set_title('NC Step size')
    axes[1][0].set_xlabel('Step size')
    axes[1][0].set_ylabel('Subjects')
    scale_yticks(axes[1][0])

    # Overall NC Window
    axes[1][1].hist(overall_nc_window_list, bins=45, color='navy', alpha=0.7, edgecolor='black')
    axes[1][1].set_title('NC Window size')
    axes[1][1].set_xlabel('Window size')
    axes[1][1].set_ylabel('Subjects')
    scale_yticks(axes[1][1])

    plt.tight_layout()
    os.makedirs('images', exist_ok=True)
    os.makedirs('images/ratio', exist_ok=True)
    plt.savefig('images/ratio/allfolds_mdd_nc_window_step_hist.png', dpi=300)
    plt.close()


def plot_metrics_csv():
    df_results = pd.read_csv('training_val_test_results_allfolds.csv')
    grouped = df_results.groupby('Epoch')
    mean_df = grouped.mean(numeric_only=True)
    std_df  = grouped.std(numeric_only=True)
    epochs_arr = mean_df.index

    def get_mean_std(df_m, df_s, col):
        if col in df_m.columns:
            return df_m[col], df_s[col]
        else:
            return None, None

    mean_train_ce, std_train_ce = get_mean_std(mean_df, std_df, 'TrainCE')
    mean_val_ce,   std_val_ce   = get_mean_std(mean_df, std_df, 'ValCE')
    mean_test_ce,  std_test_ce  = get_mean_std(mean_df, std_df, 'TestCE')

    mean_macro_loss, std_macro_loss = get_mean_std(mean_df, std_df, 'MacroLoss')
    mean_micro_loss, std_micro_loss = get_mean_std(mean_df, std_df, 'MicroLoss')

    mean_train_acc, std_train_acc = get_mean_std(mean_df, std_df, 'TrainAcc')
    mean_val_acc,   std_val_acc   = get_mean_std(mean_df, std_df, 'ValAcc')
    mean_test_acc,  std_test_acc  = get_mean_std(mean_df, std_df, 'TestAcc')

    mean_train_sen, std_train_sen = get_mean_std(mean_df, std_df, 'TrainSen')
    mean_val_sen,   std_val_sen   = get_mean_std(mean_df, std_df, 'ValSen')
    mean_test_sen,  std_test_sen  = get_mean_std(mean_df, std_df, 'TestSen')

    mean_train_spec, std_train_spec = get_mean_std(mean_df, std_df, 'TrainSpec')
    mean_val_spec,   std_val_spec   = get_mean_std(mean_df, std_df, 'ValSpec')
    mean_test_spec,  std_test_spec  = get_mean_std(mean_df, std_df, 'TestSpec')

    mean_train_f1, std_train_f1 = get_mean_std(mean_df, std_df, 'TrainF1')
    mean_val_f1,   std_val_f1   = get_mean_std(mean_df, std_df, 'ValF1')
    mean_test_f1,  std_test_f1  = get_mean_std(mean_df, std_df, 'TestF1')

    mean_rewards, std_rewards = get_mean_std(mean_df, std_df, 'TrainReward')  # Assuming 'TrainReward' column

    plt.figure(figsize=(28, 21))
    # -- Subplot(1): CE Loss
    plt.subplot(3,2,1)
    if mean_train_ce is not None:
        plt.plot(epochs_arr, mean_train_ce, color='blue', label='Train CE')
        plt.fill_between(epochs_arr, mean_train_ce - std_train_ce, mean_train_ce + std_train_ce,
                         alpha=0.2, color='blue')
    if mean_val_ce is not None:
        plt.plot(epochs_arr, mean_val_ce, color='green', label='Val CE')
        plt.fill_between(epochs_arr, mean_val_ce - std_val_ce, mean_val_ce + std_val_ce,
                         alpha=0.2, color='green')
    if mean_test_ce is not None:
        plt.plot(epochs_arr, mean_test_ce, color='red', label='Test CE')
        plt.fill_between(epochs_arr, mean_test_ce - std_test_ce, mean_test_ce + std_test_ce,
                         alpha=0.2, color='red')
    plt.xlabel('Epoch')
    plt.ylabel('CrossEntropy Loss')
    plt.legend()
    plt.title('CE Loss')

    # -- Subplot(2): RL Macro/Micro Loss
    plt.subplot(3,2,2)
    if mean_macro_loss is not None:
        plt.plot(epochs_arr, mean_macro_loss, color='green', label='MacroLoss')
        plt.fill_between(epochs_arr, mean_macro_loss - std_macro_loss, mean_macro_loss + std_macro_loss,
                         alpha=0.2, color='green')
    if mean_micro_loss is not None:
        plt.plot(epochs_arr, mean_micro_loss, color='purple', label='MicroLoss')
        plt.fill_between(epochs_arr, mean_micro_loss - std_micro_loss, mean_micro_loss + std_micro_loss,
                         alpha=0.2, color='purple')
    plt.xlabel('Epoch')
    plt.ylabel('RL Agent Loss')
    plt.legend()
    plt.title('RL Agent Loss')

    # -- Subplot(3): Accuracy
    plt.subplot(3,2,3)
    if mean_train_acc is not None:
        plt.plot(epochs_arr, mean_train_acc, color='blue', label='Train Acc')
        plt.fill_between(epochs_arr, mean_train_acc - std_train_acc, mean_train_acc + std_train_acc,
                         alpha=0.2, color='blue')
    if mean_val_acc is not None:
        plt.plot(epochs_arr, mean_val_acc, color='green', label='Val Acc')
        plt.fill_between(epochs_arr, mean_val_acc - std_val_acc, mean_val_acc + std_val_acc,
                         alpha=0.2, color='green')
    if mean_test_acc is not None:
        plt.plot(epochs_arr, mean_test_acc, color='red', label='Test Acc')
        plt.fill_between(epochs_arr, mean_test_acc - std_test_acc, mean_test_acc + std_test_acc,
                         alpha=0.2, color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Accuracy')

    # -- Subplot(4): Sensitivity
    plt.subplot(3,2,4)
    if mean_train_sen is not None:
        plt.plot(epochs_arr, mean_train_sen, color='blue', label='Train Sen')
        plt.fill_between(epochs_arr, mean_train_sen - std_train_sen, mean_train_sen + std_train_sen,
                         alpha=0.2, color='blue')
    if mean_val_sen is not None:
        plt.plot(epochs_arr, mean_val_sen, color='green', label='Val Sen')
        plt.fill_between(epochs_arr, mean_val_sen - std_val_sen, mean_val_sen + std_val_sen,
                         alpha=0.2, color='green')
    if mean_test_sen is not None:
        plt.plot(epochs_arr, mean_test_sen, color='red', label='Test Sen')
        plt.fill_between(epochs_arr, mean_test_sen - std_test_sen, mean_test_sen + std_test_sen,
                         alpha=0.2, color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Sensitivity')
    plt.legend()
    plt.title('Sensitivity')

    # -- Subplot(5): Specificity
    plt.subplot(3,2,5)
    if mean_train_spec is not None:
        plt.plot(epochs_arr, mean_train_spec, color='blue', label='Train Spec')
        plt.fill_between(epochs_arr, mean_train_spec - std_train_spec, mean_train_spec + std_train_spec,
                         alpha=0.2, color='blue')
    if mean_val_spec is not None:
        plt.plot(epochs_arr, mean_val_spec, color='green', label='Val Spec')
        plt.fill_between(epochs_arr, mean_val_spec - std_val_spec, mean_val_spec + std_val_spec,
                         alpha=0.2, color='green')
    if mean_test_spec is not None:
        plt.plot(epochs_arr, mean_test_spec, color='red', label='Test Spec')
        plt.fill_between(epochs_arr, mean_test_spec - std_test_spec, mean_test_spec + std_test_spec,
                         alpha=0.2, color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Specificity')
    plt.legend()
    plt.title('Specificity')

    # -- Subplot(6): F1 Score
    plt.subplot(3,2,6)
    if mean_train_f1 is not None:
        plt.plot(epochs_arr, mean_train_f1, color='blue', label='Train F1')
        plt.fill_between(epochs_arr, mean_train_f1 - std_train_f1, mean_train_f1 + std_train_f1,
                         alpha=0.2, color='blue')
    if mean_val_f1 is not None:
        plt.plot(epochs_arr, mean_val_f1, color='green', label='Val F1')
        plt.fill_between(epochs_arr, mean_val_f1 - std_val_f1, mean_val_f1 + std_val_f1,
                         alpha=0.2, color='green')
    if mean_test_f1 is not None:
        plt.plot(epochs_arr, mean_test_f1, color='red', label='Test F1')
        plt.fill_between(epochs_arr, mean_test_f1 - std_test_f1, mean_test_f1 + std_test_f1,
                         alpha=0.2, color='red')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.legend()
    plt.title('F1 Score')

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'all_folds_mean_std.png'), dpi=300)
    plt.close()

    # Reward uses separate figure
    if mean_rewards is not None:
        plt.figure(figsize=(12,8))
        plt.plot(epochs_arr, mean_rewards, label='Train Reward', color='blue')
        plt.fill_between(epochs_arr,
                         mean_rewards - std_rewards,
                         mean_rewards + std_rewards,
                         alpha=0.2, color='blue')
        plt.xlabel('Epoch')
        plt.ylabel('Average Reward')
        plt.legend()
        os.makedirs('images', exist_ok=True)
        plt.savefig('images/reward_plot.png', dpi=300)
        plt.close()

def plot_mdd_nc_distribution(N_FOLDS=5):
    # Data loading - add exception handling
    try:
        with open('fold_window_usage_mdd.pkl','rb') as f:
            fold_window_usage_mdd = pickle.load(f)
        with open('fold_step_usage_mdd.pkl','rb') as f:
            fold_step_usage_mdd = pickle.load(f)
        with open('fold_window_usage_nc.pkl','rb') as f:
            fold_window_usage_nc = pickle.load(f)
        with open('fold_step_usage_nc.pkl','rb') as f:
            fold_step_usage_nc = pickle.load(f)
    except (FileNotFoundError, EOFError) as e:
        print(f"Failed to load data file: {e}")
        print("Please run train.py first to generate data.")
        # Return empty data
        from collections import defaultdict
        fold_window_usage_mdd = defaultdict(lambda: defaultdict(dict))
        fold_window_usage_nc = defaultdict(lambda: defaultdict(dict))
        fold_step_usage_mdd = defaultdict(lambda: defaultdict(dict))
        fold_step_usage_nc = defaultdict(lambda: defaultdict(dict))

    # MDD data collection (no averaging)
    mdd_points = []
    for fidx in range(1, N_FOLDS+1):
        fold_mdd = fold_window_usage_mdd.get(fidx, fold_window_usage_mdd.get(str(fidx), {}))
        step_mdd = fold_step_usage_mdd.get(fidx, fold_step_usage_mdd.get(str(fidx), {}))
        for pid, wset in fold_mdd.items():
            sset = step_mdd.get(pid, [])
            for w, s in zip(wset, sset):  # Save all values individually
                mdd_points.append((w, s))

    # NC data collection (no averaging)
    nc_points = []
    for fidx in range(1, N_FOLDS+1):
        fold_nc = fold_window_usage_nc.get(fidx, fold_window_usage_nc.get(str(fidx), {}))
        step_nc = fold_step_usage_nc.get(fidx, fold_step_usage_nc.get(str(fidx), {}))
        for pid, wset in fold_nc.items():
            sset = step_nc.get(pid, [])
            # print('KDE',len(wset), len(sset))
            for w, s in zip(wset, sset):  # Save all values individually
                nc_points.append((w, s))

    # Convert to dataframe
    df_mdd = pd.DataFrame(mdd_points, columns=['WindowSize', 'StepRatio'])
    df_nc = pd.DataFrame(nc_points, columns=['WindowSize', 'StepRatio'])
    # Verify data
    print(df_nc.describe())

    # plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 12

    # MDD KDE Plot
    plt.figure(figsize=(8,6))
    sns.kdeplot(
        data=df_mdd,
        x='WindowSize',
        y='StepRatio',
        fill=True,
        cmap='Reds',
        levels=9,
        thresh=0.08,
        cut=1.5
    )
    plt.title("MDD Distribution")
    plt.xlabel("Window Size")
    plt.ylabel("Step Ratio")
    plt.tight_layout()
    os.makedirs('images', exist_ok=True)
    os.makedirs('images/ratio', exist_ok=True)
    plt.savefig('images/ratio/mdd_distribution_no_mean.png', dpi=300)
    plt.close()

    # NC KDE Plot
    plt.figure(figsize=(8,6))
    sns.kdeplot(
        data=df_nc,
        x='WindowSize',
        y='StepRatio',
        fill=True,
        cmap='Blues',
        levels=9,
        thresh=0.08,
        cut=1.5
    )
    plt.title("NC Distribution")
    plt.xlabel("Window Size")
    plt.ylabel("Step Ratio")
    plt.tight_layout()
    os.makedirs('images', exist_ok=True)
    os.makedirs('images/ratio', exist_ok=True)
    plt.savefig('images/ratio/nc_distribution_no_mean.png', dpi=300)
    plt.close()

    # MDD vs NC KDE Plot (overlapped comparison)
    plt.figure(figsize=(8,6))
    sns.kdeplot(
        data=df_mdd, x='WindowSize', y='StepRatio',
        fill=True, cmap='Reds', levels=5,
        alpha=0.7, thresh=0.1, cut=1.5
    )
    sns.kdeplot(
        data=df_nc, x='WindowSize', y='StepRatio',
        fill=True, cmap='Blues', levels=5,
        alpha=0.3, thresh=0.1, cut=1.5
    )
    plt.title("MDD vs NC (Overlapped KDE)")
    plt.xlabel("Window Size")
    plt.ylabel("Step Ratio")
    plt.tight_layout()
    os.makedirs('images', exist_ok=True)
    os.makedirs('images/ratio', exist_ok=True)
    plt.savefig('images/ratio/mdd_nc_overlap_no_mean.png', dpi=300)
    plt.close()


def plot_mdd_nc_scatter(N_FOLDS=5, output_dir=None):
    mdd_points = []
    nc_points = []
    fold_reward_mdd_data = None
    fold_reward_nc_data = None

    try:
        with open('fold_reward_mdd.pkl', 'rb') as f:
            fold_reward_mdd_data = pickle.load(f)
    except (FileNotFoundError, EOFError, pickle.UnpicklingError) as e:
        print(f"Failed to load MDD Reward data file 'fold_reward_mdd.pkl': {e}")

    try:
        with open('fold_reward_nc.pkl', 'rb') as f:
            fold_reward_nc_data = pickle.load(f)
    except (FileNotFoundError, EOFError, pickle.UnpicklingError) as e:
        print(f"Failed to load NC Reward data file 'fold_reward_nc.pkl': {e}")

    if fold_reward_mdd_data:
        for fidx in range(1, N_FOLDS + 1):
            # Try integer key first, then string key for fold index
            fold_data = fold_reward_mdd_data.get(fidx, fold_reward_mdd_data.get(str(fidx), {}))
            for pid, data_list in fold_data.items():
                for point_tuple in data_list:
                    if len(point_tuple) >= 2: # Ensure tuple has at least window and step
                        window_size, step_ratio = point_tuple[0], point_tuple[1]
                        mdd_points.append((window_size, step_ratio))
    
    if fold_reward_nc_data:
        for fidx in range(1, N_FOLDS + 1):
            fold_data = fold_reward_nc_data.get(fidx, fold_reward_nc_data.get(str(fidx), {}))
            for pid, data_list in fold_data.items():
                for point_tuple in data_list:
                    if len(point_tuple) >= 2:
                        window_size, step_ratio = point_tuple[0], point_tuple[1]
                        nc_points.append((window_size, step_ratio))

    if not mdd_points and not nc_points:
        print("Could not find data for MDD and NC groups in any fold.")
        if not fold_reward_mdd_data and not fold_reward_nc_data:
             print("Please run train.py first to generate 'fold_reward_mdd.pkl' and 'fold_reward_nc.pkl' data.")
        # Proceed to create empty plots or return

    df_mdd = pd.DataFrame(mdd_points, columns=['WindowSize', 'StepRatio'])
    df_nc  = pd.DataFrame(nc_points,  columns=['WindowSize', 'StepRatio'])

    os.makedirs('images/ratio', exist_ok=True)

    # 1) MDD Scatter (Individual Points)
    plt.figure(figsize=(8, 6))
    if not df_mdd.empty:
        sns.scatterplot(
            data=df_mdd,
            x='WindowSize',
            y='StepRatio',
            color='red',
            alpha=0.5, 
            edgecolor=None,
            s=10 
        )
    plt.title("MDD Distribution (Individual Points)")
    plt.xlabel("Window Size")
    plt.ylabel("Step Ratio")
    plt.xlim(0, 101)
    plt.ylim(0, 1.01)
    plt.tight_layout()
    plt.savefig('images/ratio/mdd_distribution_scatter.png', dpi=300)
    plt.close()

    # 2) NC Scatter (Individual Points)
    plt.figure(figsize=(8, 6))
    if not df_nc.empty:
        sns.scatterplot(
            data=df_nc,
            x='WindowSize',
            y='StepRatio',
            color='blue',
            alpha=0.5,
            edgecolor=None,
            s=10
        )
    plt.title("NC Distribution (Individual Points)")
    plt.xlabel("Window Size")
    plt.ylabel("Step Ratio")
    plt.xlim(0, 101)
    plt.ylim(0, 1.01)
    plt.tight_layout()
    plt.savefig('images/ratio/nc_distribution_scatter.png', dpi=300)
    plt.close()

    # 3) MDD vs NC Overlapped Scatter (Individual Points)
    plt.figure(figsize=(8, 6))
    plot_legend = False
    if not df_mdd.empty:
        sns.scatterplot(
            data=df_mdd,
            x='WindowSize',
            y='StepRatio',
            color='red',
            alpha=0.3,
            edgecolor=None,
            label='MDD',
            s=10
        )
        plot_legend = True
    if not df_nc.empty:
        sns.scatterplot(
            data=df_nc,
            x='WindowSize',
            y='StepRatio',
            color='blue',
            alpha=0.3,
            edgecolor=None,
            label='NC',
            s=10
        )
        plot_legend = True
        
    plt.title("MDD vs NC (Overlapped Scatter - Individual Points)")
    plt.xlabel("Window Size")
    plt.ylabel("Step Ratio")
    if plot_legend:
        plt.legend()
    plt.xlim(0, 101)
    plt.ylim(0, 1.01)
    plt.tight_layout()
    if output_dir is None:
        output_dir = 'images/ratio'
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, 'mdd_nc_scatter_overlap.png')
    plt.savefig(save_path, dpi=300)
    plt.close()

    if df_mdd.empty and df_nc.empty:
        print("No data points found to plot for MDD or NC groups.")
    else:
        print(f"Individual point scatter plots saved to {output_dir}. MDD points: {len(df_mdd)}, NC points: {len(df_nc)}")

import glob

def get_latest_results_csv():
    # Return the most recent file from results/run_*/training_val_test_results_allfolds.csv
    result_files = glob.glob('results/run_*/training_val_test_results_allfolds.csv')
    if not result_files:
        return None
    def get_run_dir_from_csv(csv_path):
        return os.path.dirname(csv_path)
    run_dirs = [get_run_dir_from_csv(p) for p in result_files]
    run_dirs.sort(key=os.path.getmtime, reverse=True)
    latest_dir = run_dirs[0]
    csv_file = os.path.join(latest_dir, 'training_val_test_results_allfolds.csv')
    return csv_file


def plot_fold_final_metrics():
    csv_path = get_latest_results_csv()
    if csv_path is None:
        print("Cannot find training result file. Please run train.py first.")
        return
    df = pd.read_csv(csv_path)
    metrics = ['val_auc', 'val_acc', 'test_auc', 'test_acc', 'test_sen', 'test_spec']
    os.makedirs('images', exist_ok=True)
    for metric in metrics:
        plt.figure(figsize=(7, 5))
        plt.bar(df['fold'], df[metric], color='skyblue')
        plt.xlabel('Fold')
        plt.ylabel(metric)
        plt.title(f'Fold-wise {metric}')
        plt.ylim(0, 1.1)
        plt.tight_layout()
        plt.savefig(f'images/foldwise_{metric}.png', dpi=200)
        plt.close()
    print("Fold-wise final metric bar plots saved to images/.")

def get_latest_run_dir():
    import glob, os
    run_dirs = glob.glob('results/run_*')
    if not run_dirs:
        return None
    # Select the most recent directory based on creation/modification time
    latest_run_dir = max(run_dirs, key=os.path.getmtime)
    return latest_run_dir

def get_latest_epoch_log_csv():
    # Return epoch_metrics_log.csv from the most recent run directory (fallback to existing method if not found)
    import csv, os
    latest_run_dir = get_latest_run_dir()
    if latest_run_dir:
        csv_path = os.path.join(latest_run_dir, 'epoch_metrics_log.csv')
        if os.path.exists(csv_path):
            return csv_path
    # fallback: file with the most rows
    result_files = glob.glob('results/run_*/epoch_metrics_log.csv')
    if not result_files:
        return None
    max_rows = 0
    best_file = None
    for f in result_files:
        try:
            with open(f, 'r') as csvfile:
                reader = csv.reader(csvfile)
                row_count = sum(1 for row in reader) - 1  # Exclude header
                if row_count > max_rows:
                    max_rows = row_count
                    best_file = f
        except Exception as e:
            continue
    return best_file

def plot_epoch_metrics(result_dir=None, output_dir=None):
    import os
    if result_dir is not None:
        # Find epoch_metrics_log.csv in result_dir
        csv_path = os.path.join(result_dir, "epoch_metrics_log.csv")
    else:
        csv_path = get_latest_epoch_log_csv()
    if csv_path is None or not os.path.exists(csv_path):
        print("Cannot find epoch_metrics_log.csv file. Please run train.py first.")
        return
    # Force: Read only first 9 columns and specify header (ignore unnecessary columns after)
    df = pd.read_csv(csv_path, usecols=range(9), header=0)
    df.columns = ['Fold', 'Epoch', 'Phase', 'Loss', 'Acc', 'Sen', 'Spec', 'F1', 'AUC']
    for col in ['Fold', 'Epoch', 'Phase']:
        if col not in df.columns:
            print(f"CSV does not have {col} column. Please check log save structure.")
            return
    # Remove whitespace from Phase values
    df['Phase'] = df['Phase'].astype(str).str.strip()
    print("Unique Phase values:", df['Phase'].unique())
    # Convert Epoch to int (for sorting and groupby), remove NaN
    df['Epoch'] = pd.to_numeric(df['Epoch'], errors='coerce')
    df = df.dropna(subset=['Epoch'])
    df['Epoch'] = df['Epoch'].astype(int)
    metrics = ['Loss', 'Acc', 'Sen', 'Spec', 'F1', 'AUC']
    phases = ['train', 'val', 'test']
    print("Data count by Phase:")
    print(df.groupby('Phase').size())
    print("Data count by Fold/Phase:")
    print(df.groupby(['Fold', 'Phase']).size())
    # If output_dir not specified, behave same as before
    if output_dir is None:
        output_dir = 'images/epoch'
    os.makedirs(output_dir, exist_ok=True)
    for metric in metrics:
        if metric not in df.columns:
            print(f"[WARNING] Metric {metric} not in columns. Skipping.")
            continue
        plt.figure(figsize=(8,6))
        for phase in phases:
            df_phase = df[(df['Phase'] == phase) & (~df[metric].isna())]
            print(f"[DEBUG] phase={phase}, metric={metric}: {len(df_phase)} rows")
            grouped = df_phase.groupby('Epoch')[metric]
            mean = grouped.mean()
            std = grouped.std()
            print(f"[DEBUG] metric={metric}, phase={phase}, group sizes:", grouped.count().to_dict())
            epochs = mean.index
            if len(mean) > 0:
                plt.plot(epochs, mean, label=f'{phase} mean')
                if std.notna().any() and (grouped.count() > 1).any():
                    plt.fill_between(epochs, mean-std, mean+std, alpha=0.2)
        plt.xlabel('Epoch')
        plt.ylabel(metric)
        plt.title(f'Epoch-wise {metric} (mean±std, all folds)')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'epochwise_{metric}.png'), dpi=200)
        plt.close()
    print("Epoch-wise metric line plots saved to", output_dir)

def plot_reward(result_dir=None, output_dir=None):
    import sys
    import importlib
    try:
        sys.path.append(os.path.dirname(__file__))
        plot_reward_mod = importlib.import_module('plot_reward')
        plot_reward_mod.main()
    except Exception as e:
        print(f"[plot_reward] Error: {e}")

def plot_full_distribution(result_dir=None, output_dir=None):
    import sys
    import importlib
    try:
        sys.path.append(os.path.dirname(__file__))
        plot_full_distribution_mod = importlib.import_module('plot_full_distribution')
        plot_full_distribution_mod.plot_full_distribution_scatter()
    except Exception as e:
        print(f"[plot_full_distribution] Error: {e}")

def plot_ablation(result_dir=None, output_dir=None):
    import sys
    import importlib
    try:
        sys.path.append(os.path.dirname(__file__))
        plot_ablation_mod = importlib.import_module('plot_ablation')
        plot_ablation_mod.plot_epoch_performance_trends()
    except Exception as e:
        print(f"[plot_ablation] Error: {e}")

def plot_gradient(result_dir=None, output_dir=None):
    import sys
    import importlib
    try:
        sys.path.append(os.path.dirname(__file__))
        plot_gradient_mod = importlib.import_module('plot_gradient')
        plot_gradient_mod.plot_gradient_norms()
    except Exception as e:
        print(f"[plot_gradient] Error: {e}")

def get_latest_policy_log_csv():
    import glob, os
    run_dirs = glob.glob('results/run_*')
    if not run_dirs:
        return None
    run_dirs.sort(key=os.path.getmtime, reverse=True)
    for run_dir in run_dirs:
        csv_path = os.path.join(run_dir, 'policy_log.csv')
        if os.path.exists(csv_path):
            return csv_path
    return None

def plot_policy_log(policy_log_path, output_dir='images/policy_log'):
    # Always use 'images/policy_log' as the default directory (no timestamp)
    output_dir = 'images/policy_log' if output_dir is None else output_dir
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(policy_log_path)

    # 1. Prediction class distribution by label
    plt.figure(figsize=(7,5))
    sns.countplot(data=df, x='prediction', hue='label')
    plt.title('Prediction distribution by true label')
    plt.xlabel('Predicted Class')
    plt.ylabel('Count')
    plt.legend(title='True Label', labels=['NC (0)','MDD (1)'])
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'prediction_by_label.png'))
    plt.close()

    # 2. Confidence distribution by label
    plt.figure(figsize=(7,5))
    sns.violinplot(data=df, x='label', y='confidence', inner='quartile')
    plt.title('Confidence distribution by true label')
    plt.xlabel('True Label (0: NC, 1: MDD)')
    plt.ylabel('Confidence')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confidence_by_label.png'))
    plt.close()

    # 3. Confusion matrix by prediction/ground truth (heatmap)
    cm = pd.crosstab(df['label'], df['prediction'])
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix (Raw Counts)')
    plt.xlabel('Predicted')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    plt.close()

    # 4. (Optional) Patient-wise average confidence
    plt.figure(figsize=(10,5))
    mean_conf = df.groupby('patient_id')['confidence'].mean().sort_values()
    mean_conf.plot(kind='bar', color='skyblue')
    plt.title('Mean confidence per patient')
    plt.xlabel('Patient ID')
    plt.ylabel('Mean Confidence')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'mean_conf_per_patient.png'))
    plt.close()

    print(f"[policy_log] Plots saved to {output_dir}")


def main(result_dir=None, epoch_dir=None, reward_dir=None, ratio_dir=None, full_dist_dir=None, ablation_dir=None, grad_dir=None):
    plot_epoch_metrics(result_dir=result_dir, output_dir=epoch_dir)
    plot_reward(result_dir=result_dir, output_dir=reward_dir)
    plot_mdd_nc_scatter(N_FOLDS=5, output_dir=ratio_dir)
    plot_full_distribution(result_dir=result_dir, output_dir=full_dist_dir)
    plot_ablation(result_dir=result_dir, output_dir=ablation_dir)
    plot_gradient(result_dir=result_dir, output_dir=grad_dir)

    # === Automatic visualization of policy_log.csv ===
    policy_log_path = get_latest_policy_log_csv()
    if policy_log_path is not None:
        plot_policy_log(policy_log_path, output_dir='images/policy_log')
    else:
        print('[policy_log] Cannot find policy_log.csv file.')

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--result_dir', type=str, default=None, help='Result folder path to analyze')
    parser.add_argument('--epoch_dir', type=str, default=None, help='Epoch plot save folder')
    parser.add_argument('--reward_dir', type=str, default=None, help='Reward plot save folder')
    parser.add_argument('--ratio_dir', type=str, default=None, help='Ratio plot save folder')
    parser.add_argument('--full_dist_dir', type=str, default=None, help='Full distribution plot save folder')
    parser.add_argument('--ablation_dir', type=str, default=None, help='Ablation plot save folder')
    parser.add_argument('--grad_dir', type=str, default=None, help='Gradient plot save folder')
    args = parser.parse_args()

    main(
        result_dir=args.result_dir,
        epoch_dir=args.epoch_dir,
        reward_dir=args.reward_dir,
        ratio_dir=args.ratio_dir,
        full_dist_dir=args.full_dist_dir,
        ablation_dir=args.ablation_dir,
        grad_dir=args.grad_dir,
    )