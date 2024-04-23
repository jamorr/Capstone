import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


def jointplot_survival_results(
    df: pd.DataFrame, strata_col: str, gt_col: str, pred_col: str
):
    # Define a color palette for different complaint types
    palette = sns.color_palette("husl", len(df[strata_col].unique()))

    # Create a scatter plot with Seaborn
    jg = sns.jointplot(
        data=df, x=gt_col, y=pred_col, hue=strata_col, palette=palette, alpha=0.7
    )
    ax = jg.ax_joint
    # Show the plot
    ax.set_ylim(-1, 20)
    ax.set_xlim(ax.get_ylim())
    plt.plot([-10, 100], [-10, 100], color="k")
    fig = ax.get_figure()
    fig.set_size_inches((20, 20))
    # ax.legend(bbox_to_anchor=[1,0.55])
    ax.set_xlabel("True Time to Close Complaint (hours)")
    ax.set_ylabel("Expected Time to Close Complaint (hours)")
    ax.set_title("Expected vs True Time to Resolve Complaint in Hours")
    return ax
