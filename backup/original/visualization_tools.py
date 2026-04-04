"""
The Hong Kong Polytechnic University
Student ID: 24027277d
Name: Yuen Tsz Ki

Used to generate charts demonstrating model performance.
"""

import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")
sns.set_palette("husl")


def plot_model_comparison(df):

    metrics_to_plot = [
        'class 1 precision',
        'class 1 recall',
        'class 1 f1-score',
        'class 0 precision',
        'class 0 recall',
        'class 0 f1-score',
        'macro precision',
        'macro recall',
        'macro f1-score',
        'accuracy',
        'auc',
    ]

    for metric in metrics_to_plot:
        if metric not in df.columns:
            continue
            
        plt.figure(figsize=(12, 6))
        
        df_sorted = df.sort_values(metric, ascending=False)
        
        ax = sns.barplot(x='model', y=metric, data=df_sorted, palette='magma')
        
        for p in ax.patches:
            ax.annotate(format(p.get_height(), '.3f'), 
                        (p.get_x() + p.get_width() / 2., p.get_height()), 
                        ha = 'center', va = 'center', 
                        xytext = (0, 9), 
                        textcoords = 'offset points')

        plt.xticks(rotation=45, ha='right')
        plt.title(f'GNN Models Comparison: {metric.upper()}')
        plt.ylabel(metric.capitalize())
        plt.xlabel('Model Name')
        plt.ylim(0, 1.1) 
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        current_time = datetime.now().strftime("%m%d_%H%M")
        file_name = f"results/comparison_{metric.replace(' ', '_')}_{current_time}.png"
        plt.savefig(file_name, dpi=300)
        
        print(f"Save: {file_name}")