import yaml 
import os 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

if __name__ == "__main__":
    with open('config/data_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    paths = config['paths']
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    merged_save_path = os.path.join(base_dir, paths['raw'], 'merged_data.csv')
    merged_df = pd.read_csv(merged_save_path, index_col='timestamp', parse_dates=True)
    log_cols = [col for col in merged_df.columns if 'log' in col]
    cov_matrix = merged_df[log_cols].cov()
    plt.figure(figsize=(12,10))
    sns.heatmap(cov_matrix, annot=False, cmap='coolwarm')
    plt.title('Log Returns Covariance Matrix')
    plt.tight_layout()
    path = os.path.join(base_dir, paths['plots'], 'log_returns_cov_matrix.png')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path)
    plt.show()

