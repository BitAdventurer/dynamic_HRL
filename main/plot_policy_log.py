import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Visualization function for policy_log.csv
def plot_policy_log(policy_log_path, output_dir='images/policy_log'):
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

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--policy_log', type=str, required=True, help='Path to policy_log.csv')
    parser.add_argument('--output_dir', type=str, default='images/policy_log', help='Plot save folder')
    args = parser.parse_args()
    plot_policy_log(args.policy_log, args.output_dir)
