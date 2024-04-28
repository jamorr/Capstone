import pathlib
from matplotlib.axes import Axes
import heapq
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.integrate import trapezoid
from sksurv.ensemble import (
    RandomSurvivalForest,
)
from sksurv.metrics import cumulative_dynamic_auc, concordance_index_ipcw
from tqdm import tqdm
from lifelines import CoxPHFitter, utils


def rmse(df: pd.DataFrame, gt_lbl: str, pre_lbl: str):
    # TODO: seek DoF for better calc
    diff = df[gt_lbl] - df[pre_lbl]
    return np.sqrt(np.dot(diff.values, diff.values) / diff.shape[0])


def batched_concordance(
    sf: RandomSurvivalForest,
    X_test: np.ndarray,
    y_test: np.ndarray,
    chunk_size: int = 8000,
):

    total = 0
    for i in tqdm(range(0, X_test.shape[0], chunk_size)):
        x_chunk = X_test[i : i + chunk_size]
        score = sf.score(x_chunk, y_test[i : i + chunk_size])
        total += score * x_chunk.shape[0] / X_test.shape[0]
    return score


def predict_surv_exp_med(
    sf: RandomSurvivalForest,
    X_test: np.ndarray,
    chunk_size: int = 8000,
    plot: bool = True,
):
    pm = pe = None
    mapper = pd.Series(sf.unique_times_)
    for i in range(0, X_test.shape[0], chunk_size):
        surv = sf.predict_survival_function(
            X_test[i : i + chunk_size], return_array=True
        )
        # list time steps
        subjects = list(range(X_test[i : i + chunk_size].shape[0]))
        # find median
        pred_med = utils.qth_survival_times(0.5, surv[subjects].T).T.squeeze()
        pred_med = pred_med.map(mapper)
        # approx integral
        pred_exp = pd.Series(trapezoid(surv, sf.unique_times_))
        if pe is None:
            pe = pd.Series(pred_exp, name="expected_htc")
            pm = pd.Series(pred_med, name="med_htc")
        else:
            pe = pd.concat((pe, pred_exp))
            pm = pd.concat((pm, pred_med))

    if plot:
        plot_surv(surv, mapper.values)

    return pe, pm


def plot_surv(surv, times):
    # plot survival curves ie. probability that request is not closed
    # at time t for all t \in T
    for i, s in tqdm(enumerate(surv)):
        plt.step(times, s, where="post")
    plt.ylabel("Probability the Request is Still Open")
    plt.xlabel("Time in Hours")
    plt.legend([])
    plt.xlim((0, 24))
    plt.grid(True)


def plot_joint(
    df,
    yvar: str,
    xlab: str = "True Time to Close Complaint (hours)",
    ylab: str = "Predicted Expected Time to Close Complaint (hours)",
    title: str = "Predicted vs True Time to Close Complaint in Hours",
):
    # Define a color palette for different complaint types
    # palette = sns.color_palette("husl", len(df["complaint_type"].unique()))

    # Create a scatter plot with Seaborn
    jg = sns.jointplot(
        data=df,
        x="hours_to_complete",
        y=yvar,
    )
    ax = jg.ax_joint
    # Show the plot
    ax.set_ylim(-1, 20)
    ax.set_xlim(ax.get_ylim())
    plt.plot([-10, 100], [-10, 100], color="k")
    fig = ax.get_figure()
    fig.set_size_inches((20, 20))
    # ax.legend(bbox_to_anchor=[1,0.55])
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.set_title(title)


def plot_coefficients(coefs, n_highlight):
    _, ax = plt.subplots(figsize=(9, 6))
    n_features = coefs.shape[0]
    alphas = coefs.columns
    for row in coefs.itertuples():
        ax.semilogx(alphas, row[1:], ".-", label=row.Index)

    alpha_min = alphas.min()
    top_coefs = coefs.loc[:, alpha_min].map(abs).sort_values().tail(n_highlight)
    for name in top_coefs.index:
        coef = coefs.loc[name, alpha_min]
        plt.text(
            alpha_min,
            coef,
            name + "   ",
            horizontalalignment="right",
            verticalalignment="center",
        )

    ax.yaxis.set_label_position("right")
    ax.yaxis.tick_right()
    ax.grid(True)
    ax.set_xlabel("alpha")
    ax.set_ylabel("coefficient")


def plot_cda(
    sf,
    X_test,
    y_train,
    y_test,
    times,
    save_path: str | pathlib.Path,
    model_name: str = "Survival Model",
    chunk_size: int = 1000,
):

    auc = np.zeros_like(times)
    mean_auc = 0.0
    for i in tqdm(range(0, y_test.shape[0], chunk_size)):
        if hasattr(sf, "predict_cumulative_hazard_function"):
            sf_chf_funcs  = sf.predict_cumulative_hazard_function(X_test[i : i + chunk_size], return_array=False)
            sf_risk_scores = np.row_stack([chf(times) for chf in sf_chf_funcs])
        else:
            sf_risk_scores = sf.predict(X_test[i : i + chunk_size])

        y_chunk = y_test[i : i + chunk_size]
        a, ma = cumulative_dynamic_auc(
            y_train, y_chunk, sf_risk_scores, times
        )

        auc += a * (y_chunk.shape[0] / y_test.shape[0])
        mean_auc += ma * (y_chunk.shape[0] / y_test.shape[0])

    # auc, mean_auc = cumulative_dynamic_auc(
    #     y_train[:chunk_size], y_test[:chunk_size], risk_scores, times
    # )


    ax = plot_cda_curves(auc, mean_auc, times, f"Cumulative Dynamic AUC of {model_name}")
    fig = ax.get_figure()
    fig.set_size_inches(12, 6)
    fig.savefig(save_path/f"{model_name}_cda_total.png", transparent=True)
    return auc, mean_auc


def get_cda_by_var(X_test, y_train, y_test, times, do_concord):
    heap = []
    for i, col in tqdm(enumerate(X_test.columns)):
        auc, mean_auc = cumulative_dynamic_auc(y_train, y_test, X_test[col], times)
        in_heap = [mean_auc, col, auc]
        if do_concord:
            ret = concordance_index_ipcw(
                y_train,
                y_test,
                X_test[col],
                tau=times[-1],
            )
            in_heap.append(ret)
        heapq.heappush(heap, in_heap)
    return heap


def get_variable_impact_auc(
    X_test,
    y_train,
    y_test,
    times,
    save_path: str | pathlib.Path,
    model_name: str,
    num_cols: int = 10,
    do_concord: bool = False,
    do_top: bool = True,
    do_bottom: bool = True,
):
    heap = get_cda_by_var(X_test, y_train, y_test, times, do_concord)
    if do_top:
        save_file = save_path / f"{model_name}_top{num_cols}_cda_impact.png"
        top_n = heapq.nlargest(num_cols, heap)
        plot_cda_by_var(top_n, times, save_file)
    if do_bottom:
        save_file = save_path / f"{model_name}_bottom{num_cols}_cda_impact.png"
        bottom_n = heapq.nsmallest(num_cols, heap)
        plot_cda_by_var(bottom_n, times, save_file)


def plot_cda_by_var(cda_auc_vals, times, save_file: str | pathlib.Path):  # noqa: F821
    fig, ax = plt.subplots()
    fig.set_size_inches(12, 6)
    for i, (mean_auc, col, auc) in enumerate(cda_auc_vals):
        plot_cda_curves(
            auc,
            mean_auc,
            times,
            "Cumulative Dynamic AUC by Column",
            ax,
            color=f"C{i}",
            label=col,
        )

    plt.legend()
    print(f"Saving {save_file.absolute()}")
    fig.savefig(save_file, transparent=True)


def plot_cda_curves(auc, mean_auc, times, title, ax: Axes = None, color="C0", label=None):
    if not ax:
        _, ax = plt.subplots()
    ax.plot(times, auc, marker="o", color=color, label=label)
    ax.axhline(
        mean_auc,
        linestyle="--",
        color=color,
    )
    ax.set_title(title)
    ax.set_xlabel("Hour from Complaint Made")
    ax.set_ylabel("Time-Dependent AUC")
    ax.grid(True)
    return ax
