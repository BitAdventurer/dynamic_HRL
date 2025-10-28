import pickle
import numpy as np
import pandas as pd

def save_fold_usage(fold_window_usage_nc, fold_step_usage_nc):
    with open('fold_window_usage_nc.pkl', 'wb') as f:
        pickle.dump(dict(fold_window_usage_nc), f)
    with open('fold_step_usage_nc.pkl', 'wb') as f:
        pickle.dump(dict(fold_step_usage_nc), f)

def save_final_performance(all_fold_test_acc, all_fold_test_sen, all_fold_test_spec, all_fold_test_f1, early_stop=False):
    # early_stop이 True이면 전달받은 리스트의 값을 그대로 사용, False이면 마지막 epoch 기준 성능 사용
    if early_stop:
        # early stopping 기준 성능 (best model)
        # 이미 각 fold의 best 성능이 리스트로 전달됨
        final_accs = all_fold_test_acc[0] if len(all_fold_test_acc) > 0 else []
        final_sens = all_fold_test_sen[0] if len(all_fold_test_sen) > 0 else []
        final_specs = all_fold_test_spec[0] if len(all_fold_test_spec) > 0 else []
        final_f1s = all_fold_test_f1[0] if len(all_fold_test_f1) > 0 else []
        output_filename = 'early_stop_performance_mean_std.csv'
    else:
        # 마지막 epoch 기준 final performance
        final_accs = [arr[-1] for arr in all_fold_test_acc if len(arr)>0]
        final_sens = [arr[-1] for arr in all_fold_test_sen if len(arr)>0]
        final_specs= [arr[-1] for arr in all_fold_test_spec if len(arr)>0]
        final_f1s  = [arr[-1] for arr in all_fold_test_f1 if len(arr)>0]
        output_filename = 'final_performance_mean_std.csv'
    if len(final_accs)>0:
        acc_mean, acc_std = np.mean(final_accs), np.std(final_accs)
        sen_mean, sen_std = np.mean(final_sens), np.std(final_sens)
        spec_mean, spec_std= np.mean(final_specs), np.std(final_specs)
        f1_mean, f1_std   = np.mean(final_f1s), np.std(final_f1s)
        df_final_perf = pd.DataFrame({
            'Metric': ['Accuracy','Sensitivity','Specificity','F1-score'],
            'Mean':   [acc_mean, sen_mean, spec_mean, f1_mean],
            'Std':    [acc_std,  sen_std,  spec_std,  f1_std]
        })
        df_final_perf.to_csv(output_filename, index=False)
