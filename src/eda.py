"""Exploratory analysis (report Section 4). Run: python -m src.eda"""
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal, stats
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import acf, adfuller, kpss, pacf

from src import config, utils

OUT = config.OUT_DIR / "eda"
SPD = config.SLOTS_PER_DAY


def gini(x):
    x = np.sort(x[x >= 0]); n = len(x)
    return float((2 * np.sum(np.arange(1, n + 1) * x) / (n * x.sum())) - (n + 1) / n)


# ------------------------------------------------------------------ Task 2.1 : distribution ---
def fig_total_distribution(totals):
    pos = totals[totals > 0]
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.2))
    ax[0].hist(np.log10(pos), bins=60, color="steelblue")
    ax[0].set(xlabel="log10(total Internet activity per area)", ylabel="# areas", title="(a) Distribution")
    ax[1].loglog(np.arange(1, len(pos) + 1), np.sort(pos)[::-1], ".", ms=2)
    ax[1].set(xlabel="rank", ylabel="total activity", title="(b) Rank-size (log-log)")
    if config.N_SQUARES == 10_000:
        im = ax[2].imshow(np.log10(totals.reshape(100, 100) + 1), origin="lower", cmap="viridis")
        plt.colorbar(im, ax=ax[2], label="log10(total)")
        ax[2].set(title="(c) Spatial map (100x100 grid)", xlabel="column", ylabel="row")
    else:
        ax[2].axis("off")
    fig.tight_layout(); fig.savefig(OUT / "fig_total_distribution.png", dpi=150); plt.close(fig)

    s = np.sort(pos)[::-1]
    stats_d = dict(n_areas=len(totals), n_zero=int((totals == 0).sum()), mean=totals.mean(),
                   median=float(np.median(totals)), std=totals.std(), max=totals.max(),
                   max_over_median=float(totals.max() / np.median(totals)),
                   skewness=float(stats.skew(totals)), kurtosis=float(stats.kurtosis(totals)),
                   top1pct_share=float(s[:max(1, len(s) // 100)].sum() / s.sum()),
                   top10pct_share=float(s[:max(1, len(s) // 10)].sum() / s.sum()), gini=gini(totals))
    utils.save_json(stats_d, OUT / "distribution_stats.json")
    return stats_d


# ------------------------------------------------------------------ Task 2.2 : five areas ------
def fig_five_areas(top3, extra=(4159, 4556), days=14):
    ids = [a for a in list(top3) + list(extra) if a <= config.N_SQUARES]
    n = days * SPD
    idx = config.TIME_INDEX[:n]
    fig, axes = plt.subplots(len(ids), 1, figsize=(15, 2.3 * len(ids)), sharex=True)
    series = {}
    for ax, a in zip(np.atleast_1d(axes), ids):
        y = utils.load_area(a)[:n]; series[a] = y
        ax.plot(idx, y, lw=0.8)
        for d in range(days):                                     # shade weekends
            if (config.START_DATE + pd.Timedelta(days=d)).dayofweek >= 5:
                ax.axvspan(idx[d * SPD], idx[(d + 1) * SPD - 1], color="orange", alpha=0.15)
        lab = f"Square {a}" + (f" (top-{top3.index(a) + 1})" if a in top3 else "")
        ax.set_ylabel("activity"); ax.set_title(lab, loc="left", fontsize=10)
    axes[-1].set_xlabel("time (orange = weekend; 2013-11-01 is a public holiday)")
    fig.tight_layout(); fig.savefig(OUT / "fig_five_areas_first_2_weeks.png", dpi=150); plt.close(fig)
    pd.DataFrame(series).corr().round(3).to_csv(OUT / "five_areas_correlation.csv")
    pd.DataFrame({a: [np.mean(y), np.std(y), np.max(y), np.mean(y == 0) * 100] for a, y in series.items()},
                 index=["mean", "std", "max", "pct_zero"]).round(2).to_csv(OUT / "five_areas_summary.csv")


# ------------------------------------------------------------------ Analysis 1 : ACF + spectrum
def analysis_acf_periodogram(x, area):
    acf_v = acf(x, nlags=1100, fft=True)
    pacf_v = pacf(x, nlags=200, method="ywm")
    f, pxx = signal.periodogram(x - x.mean(), fs=SPD)             # frequency in cycles/day
    top = np.argsort(pxx)[::-1][:6]
    peaks = pd.DataFrame({"cycles_per_day": f[top], "period_hours": 24 / f[top], "power": pxx[top]})
    peaks.to_csv(OUT / "periodogram_top_peaks.csv", index=False)
    fig, ax = plt.subplots(1, 3, figsize=(17, 4))
    ax[0].plot(acf_v, lw=0.8)
    for lag, c in [(SPD, "r"), (7 * SPD, "g")]:
        ax[0].axvline(lag, color=c, ls="--", lw=0.8, label=f"lag {lag} ({lag // SPD} d)")
    ax[0].set(title=f"ACF, area {area}", xlabel="lag (x10 min)"); ax[0].legend()
    ax[1].stem(pacf_v[1:], markerfmt=" ", basefmt=" "); ax[1].set(title="PACF (first 200 lags)", xlabel="lag")
    ax[2].loglog(f[1:], pxx[1:], lw=0.7); ax[2].set(title="Periodogram", xlabel="cycles/day", ylabel="power")
    for c in (1, 2, 3, 7 / 7):
        ax[2].axvline(c, color="r", ls=":", lw=0.6)
    fig.tight_layout(); fig.savefig(OUT / "fig_analysis1_acf_periodogram.png", dpi=150); plt.close(fig)
    return dict(acf_lag1=float(acf_v[1]), acf_lag6=float(acf_v[6]), acf_lag144=float(acf_v[SPD]),
                acf_lag1008=float(acf_v[7 * SPD]), top_periods_hours=peaks["period_hours"].round(2).tolist())


# ------------------------------------------------------------------ Analysis 2 : stationarity + STL
def analysis_stationarity_stl(x, area):
    res = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for name, s in [("raw", x), ("diff1", np.diff(x)), ("seasonal_diff_144", x[SPD:] - x[:-SPD])]:
            adf = adfuller(s, autolag="AIC")
            kp = kpss(s, regression="c", nlags="auto")
            res[name] = dict(adf_stat=adf[0], adf_p=adf[1], kpss_stat=kp[0], kpss_p=kp[1])
    stl = STL(pd.Series(x, index=config.TIME_INDEX), period=SPD, robust=True).fit()
    var = lambda v: float(np.var(v))
    res["STL_strength_seasonal_daily"] = max(0.0, 1 - var(stl.resid) / var(stl.seasonal + stl.resid))
    res["STL_strength_trend"] = max(0.0, 1 - var(stl.resid) / var(stl.trend + stl.resid))
    fig, ax = plt.subplots(4, 1, figsize=(15, 9), sharex=True)
    for a, (n, v) in zip(ax, [("observed", x), ("trend", stl.trend), ("daily seasonal", stl.seasonal),
                              ("residual", stl.resid)]):
        a.plot(config.TIME_INDEX, v, lw=0.6); a.set_ylabel(n)
    fig.suptitle(f"STL decomposition (period = 144), area {area}")
    fig.tight_layout(); fig.savefig(OUT / "fig_analysis2_stl.png", dpi=150); plt.close(fig)

    # weekday vs weekend profile (weekly seasonality evidence)
    df = pd.DataFrame({"y": x}, index=config.TIME_INDEX)
    df["tod"] = df.index.hour + df.index.minute / 60
    df["grp"] = np.where(df.index.dayofweek >= 5, "weekend", "weekday")
    prof = df.groupby(["grp", "tod"])["y"].mean().unstack(0)
    ax2 = prof.plot(figsize=(8, 3.5), title=f"Mean daily profile, area {area}")
    ax2.set_xlabel("hour of day"); plt.tight_layout(); plt.savefig(OUT / "fig_weekday_weekend_profile.png", dpi=150)
    plt.close()
    return res


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    totals = utils.load_totals()
    top3 = utils.top_areas(3)
    print("Top-3 areas by total traffic:", top3, [float(totals[a - 1]) for a in top3])
    out = {"top3": top3, "distribution": fig_total_distribution(totals)}
    fig_five_areas(top3)
    x = utils.load_area(top3[0]).astype(np.float64)
    out["analysis1_acf_periodogram"] = analysis_acf_periodogram(x, top3[0])
    out["analysis2_stationarity_stl"] = analysis_stationarity_stl(x, top3[0])
    utils.save_json(out, OUT / "eda_summary.json")
    print("saved ->", OUT)


if __name__ == "__main__":
    main()
