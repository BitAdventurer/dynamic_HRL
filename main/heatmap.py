import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import utils.util as util  # <-- your custom utility for loading data

########################################
# Helper: compute dFC from window
########################################
def compute_dFC(window_data):
    """
    window_data: shape (w, nROIs)
      w = window length, nROIs = # of ROIs
    Returns:
      corr_mat: shape (nROIs, nROIs), correlation
    """
    X = window_data - window_data.mean(axis=0, keepdims=True)
    cov = (X.T @ X) / (X.shape[0] - 1)
    diag = np.diag(cov)
    std_ = np.sqrt(diag + 1e-12)
    corr_mat = cov / (std_[:, None] * std_[None, :])
    return corr_mat

########################################
# Helper: plot correlation matrix
########################################
def plot_dFC_heatmap(corr_mat, save_path="example_dFC_heatmap.png"):
    # Configure consistent font styles (Times New Roman, etc.)
    # plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size']   = 12
    plt.rcParams['axes.labelsize']   = 12
    plt.rcParams['axes.titlesize']   = 12
    plt.rcParams['xtick.labelsize']  = 10
    plt.rcParams['ytick.labelsize']  = 10
    plt.rcParams['figure.dpi']       = 300

    plt.figure(figsize=(6,5))
    sns.heatmap(corr_mat, cmap='coolwarm', vmin=-1, vmax=1, square=True, cbar=False, xticklabels=[], yticklabels=[])
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

########################################
# Main demo
########################################
def main_dFC_demo():
    # 1) Load all data (using your custom utility).
    all_data = util.load_signal_data()

    # 2) Extract fold_1's training data as an example
    #    shape is assumed (N_subjects, T, nROIs)
    train_bd = all_data['fold_1_train_data.npy']

    # 3) Pick one subject, e.g., subject_idx=0
    subject_idx = 44
    bold_data = train_bd[subject_idx]  # shape (T, nROIs)

    window_size = 30

    # We'll create 5 different windows by shifting the start index
    # e.g., [0, 50, 100, 150, 200], but you can pick your own offsets
    start_offsets = [0, 20, 50, 100, 150]

    for i, start_idx in enumerate(start_offsets):
        end_idx = min(start_idx + window_size, bold_data.shape[0])
        window_data = bold_data[start_idx:end_idx, :]  # shape (w, nROIs)

        # Compute correlation (dFC)
        corr_mat = compute_dFC(window_data)

        # Plot heatmap
        save_path = f"subject_window{i}.png"
        plot_dFC_heatmap(corr_mat, save_path=save_path)
        print(f"Saved dFC heatmap for window {i} (start={start_idx}) to {save_path}")

if __name__ == "__main__":
    main_dFC_demo()