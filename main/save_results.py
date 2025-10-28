import pickle
import numpy as np
import pandas as pd

def save_fold_usage(fold_window_usage_nc, fold_step_usage_nc):
    with open('fold_window_usage_nc.pkl', 'wb') as f:
        pickle.dump(dict(fold_window_usage_nc), f)
    with open('fold_step_usage_nc.pkl', 'wb') as f:
        pickle.dump(dict(fold_step_usage_nc), f)

def save_final_performance(all_fold_test_acc, all_fold_test_sen, all_fold_test_spec, all_fold_test_f1, early_stop=False):
    # If early_stop is True, use the received list values as-is; if False, use performance based on last epoch
    if early_stop:
        # Early stopping-based performance (best model)
        # Best performance of each fold is already passed as a list
        final_accs = all_fold_test_acc[0] if len(all_fold_test_acc) > 0 else []
        final_sens = all_fold_test_sen[0] if len(all_fold_test_sen) > 0 else []
        final_specs = all_fold_test_spec[0] if len(all_fold_test_spec) > 0 else []
        final_f1s = all_fold_test_f1[0] if len(all_fold_test_f1) > 0 else []
        output_filename = 'early_stop_performance_mean_std.csv'
    else:
        # Final performance based on last epoch
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
