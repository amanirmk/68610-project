from functools import reduce
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import seaborn as sns


def plots():
    plt.style.use('ggplot')

    df = reduce(
        lambda left, right: pd.merge(left, right, on='model', how='outer'),
        [
            pd.read_csv('results/all_models--vipr.csv', index_col='model'),
            pd.read_csv('results/all_models--sizes.csv', index_col='model'),
            pd.read_csv('results/all_models--nlvr_zeroshot.csv', index_col='model'),
        ]
    ).drop(columns=['Unnamed: 0'])

    predictors = [c for c in df.columns if c.startswith('ViPr') or c.startswith('DV') or c=='model_size']
    task_metrics = [c for c in df.columns if c not in predictors]

    # aggregate prediction (predict overall rank)
    for task_metric in task_metrics:
        df[task_metric + '_rank'] = df[task_metric].rank(ascending=False)
    df['avg_rank'] = df[[task_metric + '_rank' for task_metric in task_metrics]].mean(axis=1)
    df['avg_rank'] = df['avg_rank'].rank(ascending=True)
    df['dowdall_rank'] = (1 / df[[task_metric + '_rank' for task_metric in task_metrics]].values).sum(axis=1)
    df['dowdall_rank'] = df['dowdall_rank'].rank(ascending=False)

    for predictor in predictors:
        df[predictor + '_rank'] = df[predictor].rank(ascending=False)
    
    predictors = [predictor + '_rank' for predictor in predictors]
    task_metrics = [task_metric + '_rank' for task_metric in task_metrics] + ['avg_rank', 'dowdall_rank']

    model_groups = set([model.split('/')[0] for model in df.index])
    for predictor in predictors:
        for task_metric in task_metrics:
            r, p = stats.spearmanr(df[predictor], df[task_metric])
            for do_rank in [True, False]:
                if do_rank:
                    x_name = predictor
                    y_name = task_metric
                else:
                    x_name = predictor.replace('_rank', '')
                    y_name = task_metric.replace('_rank', '') if task_metric not in ['avg_rank', 'dowdall_rank'] else task_metric
                plt.figure()
                for model_group in model_groups:
                    plt.scatter(
                        df[x_name][df.index.str.startswith(model_group)],
                        df[y_name][df.index.str.startswith(model_group)],
                        label=model_group
                    )
                plt.scatter([], [], c='k', label=f'Spearman r={r:.2f} (p={p:.2f})')
                plt.xlabel(x_name)
                plt.ylabel(y_name)
                plt.legend()
                plt.title(f"{x_name} vs {y_name}")
                plt.tight_layout()
                plt.savefig(f'plots/{x_name}--{y_name}.png')
                plt.close()

    # pairwise rank correlation in heatmap
    r2_heatmap = pd.DataFrame(index=predictors + task_metrics, columns=predictors + task_metrics)
    r_heatmap = pd.DataFrame(index=predictors + task_metrics, columns=predictors + task_metrics)
    for i, predictor1 in enumerate(predictors + task_metrics):
        for j, predictor2 in enumerate(predictors + task_metrics):
            r, p = stats.spearmanr(df[predictor1], df[predictor2])
            r_heatmap.iloc[i, j] = r
            r2_heatmap.iloc[i, j] = r**2

    plt.figure()
    sns.heatmap(r_heatmap.astype(float), annot=False, vmin=-1, vmax=1, center=0)
    plt.title('Rank Correlation (R)')
    plt.tight_layout()
    plt.savefig('plots/r_heatmap.png')
    plt.close()
    plt.figure()
    sns.heatmap(r2_heatmap.astype(float), annot=False, vmin=0, vmax=1)
    plt.title('Rank Correlation (R^2)')
    plt.tight_layout()
    plt.savefig('plots/r2_heatmap.png')
    plt.close()


if __name__ == '__main__':
    plots()