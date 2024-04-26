import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.integrate import trapezoid
from sksurv.ensemble import (
    RandomSurvivalForest,
)
from tqdm import tqdm
from lifelines import CoxPHFitter, utils


def rmse(df: pd.DataFrame, gt_lbl: str, pre_lbl: str):
    # TODO: seek DoF for better calc
    diff = df[gt_lbl] - df[pre_lbl]
    return np.sqrt(np.dot(diff.values, diff.values) / diff.shape[0])


def predict_surv_exp_med(sf: RandomSurvivalForest, X_test:np.ndarray, chunk_size:int=8000, plot:bool=True):
    pm = pe = None
    mapper = pd.Series(sf.unique_times_)
    for i in range(0, X_test.shape[0], chunk_size):
        surv = sf.predict_survival_function(
            X_test[i : i + chunk_size], return_array=True
        )

        subjects = list(range(X_test[i : i + chunk_size].shape[0]))
        pred_med = utils.qth_survival_times(0.5, surv[subjects].T).T.squeeze()
        pred_med = pred_med.map(mapper)

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

    for i, s in tqdm(enumerate(surv)):
        plt.step(times, s, where="post")
    plt.ylabel("Probability the Request is Still Open")
    plt.xlabel("Time in Hours")
    plt.legend([])
    plt.xlim((0,24))
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
        plt.text(alpha_min, coef, name + "   ", horizontalalignment="right", verticalalignment="center")

    ax.yaxis.set_label_position("right")
    ax.yaxis.tick_right()
    ax.grid(True)
    ax.set_xlabel("alpha")
    ax.set_ylabel("coefficient")
