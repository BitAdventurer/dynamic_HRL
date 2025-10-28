#!/bin/bash

# Ablation study 추론 성능 평가 스크립트
# 기학습된 모델을 사용하여 추론 성능만 측정합니다.

# 저장 디렉토리 생성
mkdir -p ablation_results

# 결과를 저장할 CSV 파일 생성
echo "Fold,Mode,Window,Step,TestAcc,TestSen,TestSpec,TestF1,TestAUC,TrainReward" > ablation_results/ablation_comparison.csv

# 기본 모델 (Full HRL) 추론
# echo "====== Ablation: 기본 모델 (Full HRL) 추론 ======"
# for fold in {1..5}; do
#   echo "Evaluating Full HRL model - Fold $fold"
#   python main/train.py --inference_only --model_fold $fold --ablation_mode full
#   # ablation_result.csv 파일의 내용을 가져와서 추가
#   if [ -f "ablation_result.csv" ]; then
#     tail -n 1 ablation_result.csv >> ablation_results/ablation_comparison.csv
#   fi
# done

# 1. Macro 에이전트 제거 (고정된 윈도우 크기 30)
echo "====== Ablation: Macro 에이전트 제거 (고정 윈도우) 추론 ======"
for fold in {1..5}; do
  echo "Evaluating No-Macro model - Fold $fold"
  python main/train.py --inference_only --model_fold $fold --ablation_mode no_macro --fixed_window_size 30
  # ablation_result.csv 파일의 내용을 가져와서 추가
  if [ -f "ablation_result.csv" ]; then
    tail -n 1 ablation_result.csv >> ablation_results/ablation_comparison.csv
  fi
done

# 2. Micro 에이전트 제거 (고정된 이동 간격 0.5)
echo "====== Ablation: Micro 에이전트 제거 (고정 이동 간격) 추론 ======"
for fold in {1..5}; do
  echo "Evaluating No-Micro model - Fold $fold"
  python main/train.py --inference_only --model_fold $fold --ablation_mode no_micro --fixed_shift_ratio 0.5
  # ablation_result.csv 파일의 내용을 가져와서 추가
  if [ -f "ablation_result.csv" ]; then
    tail -n 1 ablation_result.csv >> ablation_results/ablation_comparison.csv
  fi
done

# 3. 두 에이전트 모두 제거 (고정 파라미터 사용)
echo "====== Ablation: 두 에이전트 모두 제거 (고정 파라미터) 추론 ======"
for fold in {1..5}; do
  echo "Evaluating Fixed Parameters model - Fold $fold"
  python main/train.py --inference_only --model_fold $fold --ablation_mode fixed --fixed_window_size 30 --fixed_shift_ratio 0.5
  # ablation_result.csv 파일의 내용을 가져와서 추가
  if [ -f "ablation_result.csv" ]; then
    tail -n 1 ablation_result.csv >> ablation_results/ablation_comparison.csv
  fi
done

# 결과 요약 출력
echo "===== Ablation 실험 결과 요약 ====="
echo "결과는 ablation_results/ablation_comparison.csv 파일에 저장되었습니다."
echo ""

# 모드별 5-fold 평균 및 표준편차 계산
# 평균 및 표준편차 저장용 CSV 파일 생성
echo "Mode,AccMean,AccStd,SenMean,SenStd,SpecMean,SpecStd,F1Mean,F1Std,AUCMean,AUCStd,RewardMean,RewardStd" > ablation_results/ablation_stats.csv

# Python을 사용하여 평균과 표준편차 계산
echo "[모드별 평균 성능]" 

cat > calculate_stats.py << 'EOL'
import pandas as pd
import numpy as np

# CSV 파일 읽기
df = pd.read_csv('ablation_results/ablation_comparison.csv')

# 결과를 저장할 데이터프레임 초기화
result_df = pd.DataFrame(columns=['Mode', 'AccMean', 'AccStd', 'SenMean', 'SenStd', 'SpecMean', 'SpecStd', 'F1Mean', 'F1Std', 'AUCMean', 'AUCStd'])

# 각 모드에 대해 통계치 계산
for mode in ['full', 'no_macro', 'no_micro', 'fixed']:
    mode_df = df[df['Mode'] == mode]
    
    if len(mode_df) > 0:
        # 평균과 표준편차 계산
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
        
        # Reward 계산 (열이 존재하는 경우에만)
        if 'TrainReward' in mode_df.columns:
            reward_mean = mode_df['TrainReward'].mean()
            reward_std = mode_df['TrainReward'].std()
        else:
            reward_mean = 0
            reward_std = 0
        
        # 콘솔에 출력
        mode_name = 'Full HRL' if mode == 'full' else 'No Macro' if mode == 'no_macro' else 'No Micro' if mode == 'no_micro' else 'Fixed'
        reward_info = f", Reward={reward_mean:.4f}±{reward_std:.4f}" if 'TrainReward' in mode_df.columns else ""
        print(f"{mode_name}: Acc={acc_mean:.4f}±{acc_std:.4f}, Sen={sen_mean:.4f}±{sen_std:.4f}, Spec={spec_mean:.4f}±{spec_std:.4f}, F1={f1_mean:.4f}±{f1_std:.4f}, AUC={auc_mean:.4f}±{auc_std:.4f}{reward_info}")
        
        # 결과 데이터프레임에 추가
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
        
        # Reward 값 추가 (열이 존재하는 경우에만)
        if 'TrainReward' in mode_df.columns:
            result_dict['RewardMean'] = [reward_mean]
            result_dict['RewardStd'] = [reward_std]
        else:
            result_dict['RewardMean'] = [0]
            result_dict['RewardStd'] = [0]
            
        result_df = pd.concat([result_df, pd.DataFrame(result_dict)], ignore_index=True)

# CSV 파일로 저장
result_df.to_csv('ablation_results/ablation_stats.csv', index=False)

print("\n결과는 ablation_results/ablation_stats.csv 파일에도 저장되었습니다.")
EOL

# Python 스크립트 실행
python3 calculate_stats.py

# 시각화 실행
echo "[Ablation 모드별 시각화 생성 중...]"
python main/plot_ablation.py
echo "시각화 이미지가 'images/ablation' 폴더에 저장되었습니다."
