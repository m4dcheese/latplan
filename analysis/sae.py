from turtle import color
from cv2 import mean
import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline, BSpline
import scipy.optimize as opt
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

colors = [
    "tab:red",
    "tab:blue",
    "tab:green",
    "tab:orange"
]

data = pd.read_csv("/home/madcheese/thesis-latplan/analysis/StateAE color_shapes - Latent100.csv")
data.dropna(subset=["Effective bits", "Recon error A"], inplace=True)
data["Beta value"] = pd.to_numeric(data["Beta value"].str.replace(",", "."))
data["Recon error A"] = pd.to_numeric(data["Recon error A"].str.replace(",", "."))
data["Recon error B"] = pd.to_numeric(data["Recon error B"].str.replace(",", "."))
data["Bit variance A"] = pd.to_numeric(data["Bit variance A"].str.replace(",", "."))
data["Bit variance B"] = pd.to_numeric(data["Bit variance B"].str.replace(",", "."))

# data = data[data["Beta value"] <= .3]

data_root_step = data[(data["ZS Version"] == "root") & (data["Beta schedule"] == "step")]
data_mean_step = data[(data["ZS Version"] == "mean") & (data["Beta schedule"] == "step")]
data_weighted_step = data[(data["ZS Version"] == "weighted") & (data["Beta schedule"] == "step")]
data_weighted_root_step = data[(data["ZS Version"] == "weighted_root") & (data["Beta schedule"] == "step")]
data_mean_increase = data[(data["ZS Version"] == "mean") & (data["Beta schedule"] == "increase")]
data_root_increase = data[(data["ZS Version"] == "root") & (data["Beta schedule"] == "increase")]
data_root = data[(data["ZS Version"] == "root")]


def param_sigmoid(x, L, x0, k, b):
    y = L / (1 + np.exp(-k*(x-x0))) + b
    return (y)

def draw_curve_poly(df: pd.DataFrame, ax: plt.Axes, color: str, linestyle: str = '-'):
    x = df["Effective bits"].to_numpy()
    y = df["Recon error A"].to_numpy()
    polynomial_features = PolynomialFeatures(degree=3,
                                             include_bias=False)
    linear_regression = LinearRegression()
    pipeline = Pipeline([("polynomial_features", polynomial_features),
                         ("linear_regression", linear_regression)])
    pipeline.fit(x[:, np.newaxis], y)
    xnew = np.arange(x.min(), x.max(), 0.1)
    ynew = pipeline.predict(xnew[:, np.newaxis])
    ax.plot(xnew, ynew, linestyle, color=color)

def draw_curve(df: pd.DataFrame, ax: plt.Axes, color: str, linestyle: str = '-', label: str = ''):
    p0 = [max(df["Recon error A"]), np.median(df["Effective bits"]),1,min(df["Recon error A"])]
    popt, pcov = opt.curve_fit(param_sigmoid, df["Effective bits"], df["Recon error A"], p0=p0, method="lm", maxfev=8000)
    xnew = np.arange(0, df["Effective bits"].max(), 0.1)
    ynew = param_sigmoid(xnew, *popt)
    ax.plot(xnew, ynew, linestyle, color=color, label=label)

def generate_effbits_recerr():
    axes = plt.axes()
    axes.set_yscale("log")
    # axes.scatter(data_mean_step[data_mean_step["Beta value"] < 0.7]["Effective bits"], data_mean_step[data_mean_step["Beta value"] < 0.7]["Recon error A"], label="Mean Step", color=colors[0])
    
    axes.scatter(data_weighted_step["Effective bits"], data_weighted_step["Recon error A"], label="weighted", color="grey")
    # axes.scatter(data_weighted_root_step["Effective bits"], data_weighted_root_step["Recon error A"], label="weighted_root", color=colors[3])
    # draw_curve(data_mean_step[data_mean_step["Beta value"] < 0.7], axes, colors[0], label="Mean Step Curve")
    # axes.scatter(data_root_step["Effective bits"], data_root_step["Recon error A"], label="Root Step", color=colors[1])
    # draw_curve(data_root_step[data_root_step["Recon error A"] < 1.6e-02], axes, colors[1], label="Root Step Curve")
    draw_curve(data_weighted_step, axes, "grey")
    # draw_curve(data_weighted_root_step, axes, colors[3])
    # axes.scatter(data_mean_increase[data_mean_increase["Beta value"] < 0.7]["Effective bits"], data_mean_increase[data_mean_increase["Beta value"] < 0.7]["Recon error A"], label="Mean Increase", color=colors[0], marker="x")
    # draw_curve(data_mean_increase[data_mean_increase["Beta value"] < 0.7], axes, colors[1], linestyle='--')
    # draw_curve_poly(data_mean_increase[data_mean_increase["Beta value"] < 0.7], axes, colors[0], linestyle='--')
    # axes.scatter(data_root_increase["Effective bits"], data_root_increase["Recon error A"], label="Root Increase", color=colors[1], marker="x")
    # draw_curve(data_root_increase, axes, colors[1], linestyle='--', label="Root Increase Curve")
    axes.hlines(RECERR_CRIT, 0, EFFBITS_CRIT, color="grey", linestyles="dashed")
    axes.vlines(EFFBITS_CRIT, 0, RECERR_CRIT, color="grey", linestyles="dashed")
    lim_data = data[(data["Beta schedule"] == "step") & (data["Beta value"] < 0.7)]
    axes.set_ylim([lim_data["Recon error A"].min()*0.9, lim_data["Recon error A"].max()*1.1])
    axes.set_xlim([lim_data["Effective bits"].min()*0.9, lim_data["Effective bits"].max()*1.1])
    axes.set_xlabel("Effective bits")
    axes.set_ylabel("Reconstruction error")
    # axes.legend()
    plt.show()

def generate_beta_impact(y="Effective bits"):
    betas = data.loc[data["Beta value"] != .2, "Beta value"].sort_values().unique()
    axes = plt.axes()
    for df in [data_mean_step, data_mean_increase, data_root_step, data_root_increase]:
        df_local = df[df["Beta value"] != .2].groupby("Beta value")[y].mean().sort_index()
        print(df_local)
        plt.plot(betas, df_local, label=f"{df['ZS Version'].iloc[0]} {df['Beta schedule'].iloc[0]}")
    
    plt.xlabel("Beta value")
    plt.ylabel(y)
    plt.legend()
    axes.set_yscale("log")
    plt.show()

width = .003

def generate_noise_impact(y="Recon error", absolute=False):
    betas = data.loc[data["Beta value"] < .2, "Beta value"].sort_values().unique()
    axes = plt.axes()
    for i, df in enumerate([data_mean_step, data_root_step, data_weighted_step, data_weighted_root_step]):
        df_local = df[df["Beta value"] < .2]
        if absolute:
            df_local["diff"] = df_local[f"{y} B"] - df_local[f"{y} A"]
        else:
            df_local["diff"] = df_local[f"{y} B"] / df_local[f"{y} A"]
        plt.bar(
            betas + (i - 1.5) * width,
            df_local.groupby("Beta value")["diff"].mean(),
            width=width,
            label=f"{df['ZS Version'].iloc[0]} {df['Beta schedule'].iloc[0]}"
        )
    
    plt.xlabel("Beta value")
    plt.ylabel(f"{y} diff {'absolute' if absolute else 'relative'}")
    plt.legend()
    plt.show()


EFFBITS_CRIT = 50
RECERR_CRIT = 1e-2

def plot_hps(df: pd.DataFrame, ax: plt.Axes, title: str) -> plt.Figure:
    width = 0.01
    grouped = df.groupby("Beta value").mean().reset_index().sort_values("Beta value")
    beta_min = df["Beta value"].min()
    beta_max = df["Beta value"].max()

    sp1 = make_interp_spline(grouped["Beta value"], grouped["Recon error A"], k=1)
    sp2 = make_interp_spline(grouped["Beta value"], grouped["Effective bits"], k=1)

    xnew = np.linspace(beta_min, beta_max, 100)

    recerr = sp1(xnew)
    effbits = sp2(xnew)

    recerr_diff = lambda x: sp1(x) - RECERR_CRIT
    effbits_diff = lambda x: sp2(x) - EFFBITS_CRIT

    recerr_intersect = opt.bisect(recerr_diff, beta_min, beta_max)
    effbits_intersect = opt.bisect(effbits_diff, beta_min, beta_max)

    ax.set_xlabel("Beta value")
    ax2 = ax.twinx()

    # print(np.array(np.meshgrid([0, 100], [beta_min, beta_max])).T.reshape(-1, 2))
    # ax1.fill(np.array(np.meshgrid([0, 100], [beta_min, beta_max])).T.reshape(-1, 2), color="grey")
    ax.fill_betweenx([0, 100], [recerr_intersect, recerr_intersect], [effbits_intersect, effbits_intersect], color="#3333")
    ax.set_title(f"{title}\nBeta width: {round(recerr_intersect - effbits_intersect, ndigits=2)}")

    ax.plot(xnew, effbits, color=colors[1])
    ax.set_ylabel("Effective Bits", color=colors[1])
    ax.hlines(EFFBITS_CRIT, beta_min, beta_max, linestyles="dashed", color=colors[1])

    ax2.plot(xnew, recerr, color=colors[0])
    ax2.set_ylabel("Reconstruction Error", color=colors[0])
    ax2.hlines(RECERR_CRIT, beta_min, beta_max, linestyles="dashed", color=colors[0])
    ax2.set_yticks(np.arange(0, recerr.max(), 0.01))

def generate_hps_step_increase():
    fix, axes = plt.subplots(1, 2)
    plot_hps(data_root_step[data_root_step["Beta value"] < .5], axes.flat[0], title="Root Step")
    plot_hps(data_root_increase[data_root_increase["Beta value"] < .5], axes.flat[1], title="Root Increase")
    # plot_hps(data_weighted_step, axes.flat[2], title="Weighted")
    # plot_hps(data_weighted_root_step, axes.flat[3], title="Weighted Root")
    plt.tight_layout()
    plt.subplots_adjust(top=.8, bottom=.2)
    plt.show()
    

generate_hps_step_increase()