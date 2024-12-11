import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns


def update_ticks():
    plt.gca().set_xticklabels(
        [
            " ".join(s[0].upper() + s[1:] for s in tick_label.get_text().split("_"))
            for tick_label in plt.gca().get_xticklabels()
        ]
    )
    plt.gca().set_yticklabels(
        [
            " ".join(s[0].upper() + s[1:] for s in tick_label.get_text().split("_"))
            for tick_label in plt.gca().get_yticklabels()
        ]
    )
    plt.xticks(rotation=45, ha="right", rotation_mode="anchor")


def heatmaps(df, cols, rank_cols, pdf=False, exclude_r_ticks=True):
    df = df[cols].dropna()
    print(len(df))
    hmap_r = pd.DataFrame(index=cols, columns=cols)
    hmap_r2 = pd.DataFrame(index=cols, columns=cols)
    for i, c1 in enumerate(cols):
        x = df[c1]
        if c1 in rank_cols:
            x = -x
        for j, c2 in enumerate(cols):
            y = df[c2]
            if c2 in rank_cols:
                y = -y
            r, p = stats.spearmanr(x, y)
            hmap_r.iloc[i, j] = r
            hmap_r2.iloc[i, j] = r**2

    fig = plt.figure(figsize=(8, 8))
    sns.heatmap(
        hmap_r.astype(float),
        annot=True,
        vmin=-1,
        vmax=1,
        center=0,
        square=True,
        cbar=False,
    )
    update_ticks()
    if exclude_r_ticks:
        plt.xticks([], [])
    plt.title("Rank Correlation ($R$)")
    plt.tight_layout()
    if pdf:
        plt.savefig("heatmap_r.pdf")
    else:
        plt.savefig("heatmap_r.png", dpi=300)
    plt.close()

    fig = plt.figure(figsize=(8, 8))
    sns.heatmap(
        hmap_r2.astype(float), annot=True, vmin=0, vmax=1, square=True, cbar=False
    )
    update_ticks()
    plt.title("Rank Correlation ($R^2$)")
    plt.tight_layout()
    if pdf:
        plt.savefig("heatmap_r2.pdf")
    else:
        plt.savefig("heatmap_r2.png", dpi=300)
    plt.close()


if __name__ == "__main__":
    df = pd.read_csv("results_for_paper/all.csv", index_col=None)
    # missing paligemma, as run again afterwards (fine for heatmap, add afterwards to table)

    # sort models by num nan
    ind = df.isna().sum(axis=1).sort_values().index
    df = df.loc[ind]

    df["model_size"] = df["model_size"] / 1_000_000_000

    table_cols = [
        "model",
        "model_size",
        "DV",
        "DT",
        "ViPr",
        "ViPr_prime",
        "VQA2_zeroshot_exact_acc",
        "VQA2_zeroshot_partial_acc",
        "NLVR_zeroshot_precision",
        "NLVR_zeroshot_consistency",
        "VQA2_finetune_exact_acc",
        "VQA2_finetune_partial_acc",
    ]

    df[table_cols].to_latex(
        "table.tex",
        na_rep=" ",
        index=False,
        float_format="%.4f",
        columns=table_cols,
        header=[
            "Model",
            "Size",
            r"$D_V$",
            r"$D_T$",
            "vipr",
            "vipr_prime",
            r"VQA2$_ze$",
            r"VQA2$_zp$",
            r"NLVR$_zp$",
            r"NLVR$_zc$",
            r"VQA2$_fe$",
            r"VQA2$_fp$",
        ],
        # still need to be adjusted in tex (couldn't use brackets)
    )

    agg_cols = [
        "VQA2_zeroshot_exact_acc",
        "VQA2_zeroshot_partial_acc",
        "NLVR_zeroshot_precision",
        "NLVR_zeroshot_consistency",
    ]

    ranks_list = []
    for c in agg_cols:
        ranks_list.append(df[c].rank(ascending=False).values)

    ranks = np.vstack(ranks_list).T
    df["rank_by_avg"] = pd.Series(ranks.mean(axis=1)).rank().values
    df["rank_by_dowdall"] = (
        pd.Series((1 / ranks).sum(axis=1)).rank(ascending=False).values
    )

    rank_cols = ["rank_by_avg", "rank_by_dowdall"]
    cols = ["ViPr", "model_size"] + agg_cols + rank_cols
    heatmaps(df, cols=cols, rank_cols=rank_cols)
