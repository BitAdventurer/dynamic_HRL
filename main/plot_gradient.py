import os
import glob
import pandas as pd
import matplotlib.pyplot as plt

def get_latest_gradient_csv():
    files = glob.glob('results/run_*/gradient_norms.csv')
    if not files:
        return None
    files.sort(key=os.path.getmtime, reverse=True)
    return files[0]

def plot_gradient_norms():
    csv_path = get_latest_gradient_csv()
    if csv_path is None:
        print("gradient_norms.csv 파일을 찾을 수 없습니다. train.py에서 gradient 기록 기능을 먼저 활성화하세요.")
        return
    df = pd.read_csv(csv_path)
    os.makedirs('images/gradient_plots', exist_ok=True)
    plt.figure(figsize=(10,6))
    for col in df.columns:
        if col == 'Epoch':
            continue
        plt.plot(df['Epoch'], df[col], label=col)
    plt.xlabel('Epoch')
    plt.ylabel('Gradient Norm')
    plt.title('Gradient Norms per Epoch (by module)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('images/gradient_plots/gradient_norms.png', dpi=200)
    plt.close()
    print("Gradient norm plot saved to images/gradient_plots/gradient_norms.png")

if __name__ == '__main__':
    plot_gradient_norms()
