#!/bin/bash

# Ablation study inference performance evaluation script
# Measures inference performance only using pre-trained models.

# Create storage directory
mkdir -p ablation_results

# Create CSV file to store results
echo "Fold,Mode,Window,Step,TestAcc,TestSen,TestSpec,TestF1,TestAUC,TrainReward" > ablation_results/ablation_comparison.csv

# Base model (Full HRL) inference
# echo "====== Ablation: Base model (Full HRL) inference ======"
# for fold in {1..5}; do
#   echo "Evaluating Full HRL model - Fold $fold"
#   python main/train.py --inference_only --model_fold $fold --ablation_mode full
#   # Get contents of ablation_result.csv file and append
#   if [ -f "ablation_result.csv" ]; then
#     tail -n 1 ablation_result.csv >> ablation_results/ablation_comparison.csv
#   fi
# done

# 1. Remove Macro agent (fixed window size 30)
echo "====== Ablation: Remove Macro agent (fixed window) inference ======"
for fold in {1..5}; do
  echo "Evaluating No-Macro model - Fold $fold"
  python main/train.py --inference_only --model_fold $fold --ablation_mode no_macro --fixed_window_size 30
  # Get contents of ablation_result.csv file and append
  if [ -f "ablation_result.csv" ]; then
    tail -n 1 ablation_result.csv >> ablation_results/ablation_comparison.csv
  fi
done

# 2. Remove Micro agent (fixed shift interval 0.5)
echo "====== Ablation: Remove Micro agent (fixed shift interval) inference ======"
for fold in {1..5}; do
  echo "Evaluating No-Micro model - Fold $fold"
  python main/train.py --inference_only --model_fold $fold --ablation_mode no_micro --fixed_shift_ratio 0.5
  # Get contents of ablation_result.csv file and append
  if [ -f "ablation_result.csv" ]; then
    tail -n 1 ablation_result.csv >> ablation_results/ablation_comparison.csv
  fi
done

# 3. Remove both agents (use fixed parameters)
echo "====== Ablation: Remove both agents (fixed parameters) inference ======"
for fold in {1..5}; do
  echo "Evaluating Fixed Parameters model - Fold $fold"
  python main/train.py --inference_only --model_fold $fold --ablation_mode fixed --fixed_window_size 30 --fixed_shift_ratio 0.5
  # Get contents of ablation_result.csv file and append
  if [ -f "ablation_result.csv" ]; then
    tail -n 1 ablation_result.csv >> ablation_results/ablation_comparison.csv
  fi
done

# Print result summary
echo "===== Ablation experiment result summary ====="
echo "Results have been saved to ablation_results/ablation_comparison.csv file."
echo ""

# Calculate 5-fold average and standard deviation for each mode
# Create CSV file for storing average and standard deviation
echo "Mode,AccMean,AccStd,SenMean,SenStd,SpecMean,SpecStd,F1Mean,F1Std,AUCMean,AUCStd,RewardMean,RewardStd" > ablation_results/ablation_stats.csv

# Calculate average and standard deviation using Python
echo "[Average performance by mode]" 

cat > calculate_stats.py << 'EOL'
import pandas as pd
import numpy as np

# Read CSV file
df = pd.read_csv('ablation_results/ablation_comparison.csv')

# Initialize dataframe to store results
result_df = pd.DataFrame(columns=['Mode', 'AccMean', 'AccStd', 'SenMean', 'SenStd', 'SpecMean', 'SpecStd', 'F1Mean', 'F1Std', 'AUCMean', 'AUCStd'])

# Calculate statistics for each mode
for mode in ['full', 'no_macro', 'no_micro', 'fixed']:
    mode_df = df[df['Mode'] == mode]
    
    if len(mode_df) > 0:
        # Calculate mean and standard deviation
        acc_mean = mode_df['TestAcc'].mean()
        acc_std = mode_df['TestAcc'].std()
        sen_mean = mode_df['TestSen'].mean()
        sen_std = mode_df['TestSen'].std()
        spec_mean = mode_df['TestSpec'].mean()
        spec_std = mode_df['TestSpec'].std()
        f1_mean = mode_df['TestF1'].mean()
        f1_std = mode_df['TestF1'].std()
        auc_mean = mode_df['TestAUC'].mean()
        auc_std = mode_df['TestAUC'].std()
        
        # Calculate Reward (only if column exists)
        if 'TrainReward' in mode_df.columns:
            reward_mean = mode_df['TrainReward'].mean()
            reward_std = mode_df['TrainReward'].std()
        else:
            reward_mean = 0
            reward_std = 0
        
        # Output to console
        mode_name = 'Full HRL' if mode == 'full' else 'No Macro' if mode == 'no_macro' else 'No Micro' if mode == 'no_micro' else 'Fixed'
        reward_info = f", Reward={reward_mean:.4f}±{reward_std:.4f}" if 'TrainReward' in mode_df.columns else ""
        print(f"{mode_name}: Acc={acc_mean:.4f}±{acc_std:.4f}, Sen={sen_mean:.4f}±{sen_std:.4f}, Spec={spec_mean:.4f}±{spec_std:.4f}, F1={f1_mean:.4f}±{f1_std:.4f}, AUC={auc_mean:.4f}±{auc_std:.4f}{reward_info}")
        
        # Append to results dataframe
        result_dict = {
            'Mode': [mode],
            'AccMean': [acc_mean],
            'AccStd': [acc_std],
            'SenMean': [sen_mean],
            'SenStd': [sen_std],
            'SpecMean': [spec_mean],
            'SpecStd': [spec_std],
            'F1Mean': [f1_mean],
            'F1Std': [f1_std],
            'AUCMean': [auc_mean],
            'AUCStd': [auc_std],
        }
        
        # Add Reward values (only if column exists)
        if 'TrainReward' in mode_df.columns:
            result_dict['RewardMean'] = [reward_mean]
            result_dict['RewardStd'] = [reward_std]
        else:
            result_dict['RewardMean'] = [0]
            result_dict['RewardStd'] = [0]
            
        result_df = pd.concat([result_df, pd.DataFrame(result_dict)], ignore_index=True)

# Save as CSV file
result_df.to_csv('ablation_results/ablation_stats.csv', index=False)

print("\nResults have also been saved to ablation_results/ablation_stats.csv file.")
EOL

# Execute Python script
python3 calculate_stats.py

# Execute visualization
echo "[Generating visualization by ablation mode...]"
python main/plot_ablation.py
echo "Visualization images have been saved to 'images/ablation' folder."
