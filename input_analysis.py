"""Data reading and processing for homework 1."""
# %%  Intialize the file and set global settings.

# --- Setup ---
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.graphics.tsaplots import plot_acf
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
np.random.seed(42)


def fmt_params(params, nd=4):
    # Readable string for a scipy parameter tuple, e.g. (47.937, 3.9146)
    return "(" + ", ".join(f"{float(p):.{nd}f}" for p in params) + ")"


ALPHA = 0.05          # significance level used for every test in this notebook

plt.rcParams["figure.figsize"] = (9, 5)
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.3
pd.set_option("display.width", 140)

# %% Read the data
df = pd.read_excel("Interarrival_Data.xlsx", sheet_name="Sheet1")
data = df["InterArrival Times"].dropna().to_numpy()

data = np.asarray(data, dtype=float)

print(f"n = {len(data)} observations loaded")
print(f"first 10 values: {np.round(data[:10], 3)}")
df.head()

# %% Statistically summarize the data

# --- Descriptive statistics (slide 6 metrics) ---
n        = len(data)
mean     = data.mean()
median   = np.median(data)
mode_res = stats.mode(data, keepdims=False)
std      = data.std(ddof=1)            # sample standard deviation (n - 1)
variance = data.var(ddof=1)
kurt     = stats.kurtosis(data, bias=False)   # EXCESS kurtosis: 0 = Normal
skew     = stats.skew(data, bias=False)
data_rng = data.max() - data.min()

se       = std / np.sqrt(n)
t_crit   = stats.t.ppf(1 - ALPHA / 2, df=n - 1)
ci_low   = mean - t_crit * se
ci_high  = mean + t_crit * se

summary = pd.DataFrame(
    [
        ("Sample size (n)",            f"{n}"),
        ("Mean",                       f"{mean:.4f}"),
        ("Median",                     f"{median:.4f}"),
        ("Mode",                       f"{mode_res.mode:.4f} (appears {mode_res.count} times)"),
        ("Standard deviation",         f"{std:.4f}"),
        ("Variance",                   f"{variance:.4f}"),
        ("Kurtosis (excess)",          f"{kurt:.4f}"),
        ("Skewness",                   f"{skew:.4f}"),
        ("Minimum",                    f"{data.min():.4f}"),
        ("Maximum",                    f"{data.max():.4f}"),
        ("Range",                      f"{data_rng:.4f}"),
        ("Standard error of the mean", f"{se:.4f}"),
        ("95% CI for the mean",        f"[{ci_low:.4f}, {ci_high:.4f}]"),
    ],
    columns=["Metric", "Value"],
)

print("DESCRIPTIVE STATISTICS")
print("=" * 55)
summary
# %% Output a summary of the data.

# --- What the summary suggests about distribution shape ---
print("READING THE SUMMARY")
print("=" * 55)
print(f"Mean ({mean:.3f}) vs median ({median:.3f}): difference = {abs(mean - median):.3f}"
      f"  ->  {'close, consistent with a symmetric distribution' if abs(mean - median) < 0.25 * std else 'noticeably different, suggests asymmetry'}")
print(f"Skewness = {skew:.3f}  ->  {'approximately symmetric' if abs(skew) < 0.5 else ('right-skewed' if skew > 0 else 'left-skewed')}")
print(f"Excess kurtosis = {kurt:.3f}  ->  {'Normal-like peakedness' if abs(kurt) < 1 else ('more peaked / heavier tails than Normal' if kurt > 0 else 'flatter than Normal')}")
print(f"Support: all values {'positive' if data.min() > 0 else 'include zero or negatives'}, "
      f"range {data.min():.3f} to {data.max():.3f}")
print()

# %% Visualize the data in a summary.

# --- Visual summary: histogram and boxplot ---
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

axes[0].hist(data, bins="sturges", color="steelblue", edgecolor="white")
axes[0].axvline(mean,   color="red",       linestyle="--", linewidth=1.5, label=f"Mean = {mean:.2f}")
axes[0].axvline(median, color="darkorange", linestyle=":",  linewidth=2,   label=f"Median = {median:.2f}")
axes[0].set_title("Histogram: shape of the data")
axes[0].set_xlabel("Value"); axes[0].set_ylabel("Frequency"); axes[0].legend()

axes[1].boxplot(data, vert=False, widths=0.6,
                patch_artist=True, boxprops=dict(facecolor="steelblue", alpha=0.6),
                flierprops={
                    'marker': 'o',
                    'markersize': 6,
                    'markerfacecolor': 'red',
                    'markeredgecolor': 'black',
                    },
                    )
axes[1].set_title("Boxplot: center, spread, and outliers")
axes[1].set_xlabel("Value"); axes[1].set_yticks([])

plt.tight_layout()
plt.show()

# %% Stationarity check.

# --- Visual check 1: time-series (observation-order) plot, slide 8 ---
fig, ax = plt.subplots(figsize=(11, 4.5))
ax.plot(data, marker=".", linestyle="-", linewidth=0.8, markersize=5, color="steelblue", label="Observation")
ax.axhline(mean, color="red", linestyle="--", linewidth=1.5, label=f"Overall mean = {mean:.2f}")

# rolling mean: if the data are stationary this should stay flat near the overall mean
window = max(5, len(data) // 10)
rolling = pd.Series(data).rolling(window, center=True).mean()
ax.plot(rolling, color="darkorange", linewidth=2, label=f"Rolling mean (window = {window})")

ax.set_title("Time / Index Series Plot: look for trends with respect to time")
ax.set_xlabel("Observation order"); ax.set_ylabel("Value"); ax.legend()
plt.tight_layout()
plt.show()

# supporting numbers: split the series in half and compare
half = len(data) // 2
print(f"First half : mean = {data[:half].mean():.4f}, std = {data[:half].std(ddof=1):.4f}")
print(f"Second half: mean = {data[half:].mean():.4f}, std = {data[half:].std(ddof=1):.4f}")
print("\nNo definite pattern in the plot and similar halves -> the series looks stationary.")
# %% Independence check.

# --- Visual check 2: lag-1 scatterplot x(i) vs x(i+1), slide 10 ---
x_i, x_next = data[:-1], data[1:]
lag1_corr = np.corrcoef(x_i, x_next)[0, 1]

fig, ax = plt.subplots(figsize=(6.5, 6))
ax.scatter(x_i, x_next, s=25, color="steelblue", alpha=0.7, label="(x(i), x(i+1))")

# least-squares line: its slope is what "is there a line?" refers to
slope, intercept = np.polyfit(x_i, x_next, 1)
xs = np.linspace(x_i.min(), x_i.max(), 100)
ax.plot(xs, slope * xs + intercept, "r--", linewidth=1.5,
        label=f"Least-squares line (slope = {slope:.3f})")

ax.set_title(f"Lag-1 Scatterplot - correlation = {lag1_corr:.3f}")
ax.set_xlabel("x(i)"); ax.set_ylabel("x(i+1)"); ax.legend()
plt.tight_layout()
plt.show()

print(f"Lag-1 correlation coefficient = {lag1_corr:.4f}")
if abs(lag1_corr) < 0.2:
    print("No readily apparent line in the scatterplot -> no evidence of lag-1 correlation.")
else:
    print("A line is apparent -> lag-1 correlation is present; the independence assumption is questionable.")

# %% Further stationarity and independence checking. The Augmented Dickey-Fuller (ADF) test is below.

"""
Augmented Dickey-Fuller (ADF) test for stationarity. H0: the series has a unit root, i.e. it is non-stationary. 
A small p-value (< 0.05) lets us reject non-stationarity and conclude the series is stationary. Note that the null here is the 
opposite of what we want, so a small p-value is the good outcome.
"""

# --- Stationarity: Augmented Dickey-Fuller test ---
adf_stat, adf_p, adf_lags, adf_nobs, adf_crit, _ = adfuller(data, autolag="AIC")

print("AUGMENTED DICKEY-FULLER TEST (stationarity)")
print("=" * 55)
print("H0: the series is NON-stationary (has a unit root)")
print("H1: the series is stationary")
print(f"  ADF statistic : {adf_stat:.4f}")
print(f"  p-value       : {adf_p:.4f}")
print(f"  lags used     : {adf_lags}")
for k_, v_ in adf_crit.items():
    print(f"  critical value ({k_}): {v_:.4f}")

if adf_p < ALPHA:
    print(f"\n  -> p = {adf_p:.4f} < {ALPHA}: REJECT H0. The series appears STATIONARY.")
else:
    print(f"\n  -> p = {adf_p:.4f} >= {ALPHA}: FAIL TO REJECT H0. The series may be NON-STATIONARY.")

# %% Lijung-Box test for autocorrelation.

"""
Ljung-Box test for independence. H0: there is no autocorrelation up to the chosen lag.
A large p-value (>= 0.05) means no significant autocorrelation was detected, which supports independence.
"""
# --- Independence: Ljung-Box test for autocorrelation ---
lb_lags   = min(20, max(1, len(data) // 5))
lb_result = acorr_ljungbox(data, lags=[lb_lags], return_df=True)
lb_stat   = lb_result["lb_stat"].iloc[0]
lb_p      = lb_result["lb_pvalue"].iloc[0]

print(f"LJUNG-BOX TEST (independence, lags = {lb_lags})")
print("=" * 55)
print("H0: no autocorrelation up to the given lag (observations are independent)")
print("H1: significant autocorrelation is present")
print(f"  LB statistic : {lb_stat:.4f}")
print(f"  p-value      : {lb_p:.4f}")

if lb_p >= ALPHA:
    print(f"\n  -> p = {lb_p:.4f} >= {ALPHA}: FAIL TO REJECT H0. No significant autocorrelation; independence is supported.")
else:
    print(f"\n  -> p = {lb_p:.4f} < {ALPHA}: REJECT H0. Significant autocorrelation; independence is questionable.")

# --- Supporting visual: autocorrelation function ---
fig, ax = plt.subplots(figsize=(9, 4))
plot_acf(data, lags=lb_lags, ax=ax)
ax.set_title("Autocorrelation Function (ACF): bars inside the shaded band are not significant")
plt.tight_layout()
plt.show()
# %% Independent and identically distributed (i.i.d) verdict.

# --- i.i.d. verdict ---
stationary  = adf_p < ALPHA
independent = lb_p >= ALPHA

print("i.i.d. CHECK SUMMARY")
print("=" * 55)
print(f"  Stationary  (ADF)       : {'YES' if stationary  else 'NO'}   (p = {adf_p:.4f})")
print(f"  Independent (Ljung-Box) : {'YES' if independent else 'NO'}   (p = {lb_p:.4f})")
print(f"  Lag-1 correlation       : {lag1_corr:.4f}")
print()
if stationary and independent:
    print("The data are well behaved (i.i.d.). Proceed to hypothesize a distribution and")
    print("test goodness of fit, as on slide 11.")
else:
    print("At least one assumption is flagged. Note that the visual checks and the formal tests can disagree")
    print()
# %% Assuming the stationarity and independence checks pass, state and test the hypothesis.
"""
Goodness of fit is a hypothesis test (slide 11). Before running anything, state the hypotheses and the significance level.

H0 (null): The Normal distribution adequately fits the data. That is, the sample was drawn from a Normal distribution with mean mu and standard deviation sigma estimated from the data.

H1 (alternative): The Normal distribution does not adequately fit the data.

Significance level: alpha = 0.05

Two points to keep in mind:

1. The null is the fit, so failing to reject H0 is the favorable outcome. A large p-value means the observed counts are consistent with the Normal distribution; it is not proof the data are Normal, only that there is no evidence against it.
2. The parameters mu and sigma are estimated from this same data, so the degrees of freedom of the chi-squared test must be reduced by the number of estimated parameters (2 for the Normal).
"""

# --- The hypothesized distribution ---
hypothesized_name   = "Exponential"
hypothesized_dist   = stats.expon
hypothesized_params = stats.expon.fit(data)   # MLE: (loc, scale) = (mean, population sd)

mu_hat, sigma_hat = hypothesized_params

print("HYPOTHESIS TEST SETUP")
print("=" * 55)
print(f"  H0: the data follow a {hypothesized_name} distribution")
print(f"  H1: the data do not follow a {hypothesized_name} distribution")
print(f"  alpha = {ALPHA}")
print()
print(f"  Parameters estimated from the data (MLE):")
print(f"    mu    = {mu_hat:.4f}")
print(f"    sigma = {sigma_hat:.4f}   (MLE uses n; the sample sd with n-1 is {std:.4f})")
print(f"  Number of estimated parameters p = {len(hypothesized_params)}")
# %% Chi-Squared Goodness-of-Fit

def sturges_bins(n):
    """Sturges' formula from slide 13: k = log2(n) + 1."""
    k_raw = np.log2(n) + 1
    k     = int(np.ceil(k_raw)) # Adjusting from np.floor to np.ceil to round up.
    return k, k_raw


def equiprobable_chi_square(sample, dist, params, k=None, alpha=ALPHA, n_estimated=None):
    """
    Chi-squared goodness-of-fit test using EQUIPROBABLE bins, exactly as on slides 13-17.

    Each of the k bins is given probability 1/k under the hypothesized distribution,
    so every expected count equals n/k. Boundaries are the distribution's quantiles
    at 1/k, 2/k, ..., (k-1)/k.

    n_estimated : number of parameters estimated from this same sample.
                  Defaults to len(params). Pass 0 when the parameters were
                  specified in advance rather than fitted to the data.

    Returns (table DataFrame, results dict).
    """
    sample = np.asarray(sample, dtype=float)
    n = len(sample)
    if k is None:
        k, _ = sturges_bins(n)
    if n_estimated is None:
        n_estimated = len(params)

    probs    = np.arange(1, k) / k
    interior = np.asarray(dist.ppf(probs, *params), dtype=float)
    edges    = np.concatenate(([-np.inf], interior, [np.inf]))

    observed = np.histogram(sample, bins=edges)[0].astype(float)
    expected = np.full(k, n / k)
    contrib  = (observed - expected) ** 2 / expected
    chi2_stat = contrib.sum()

    dof = k - 1 - n_estimated
    if dof > 0:
        crit   = stats.chi2.ppf(1 - alpha, dof)
        p_val  = stats.chi2.sf(chi2_stat, dof)
    else:
        crit, p_val = np.nan, np.nan

    labels = ([f"< {interior[0]:.2f}"]
              + [f"{interior[i]:.2f} - {interior[i+1]:.2f}" for i in range(k - 2)]
              + [f">= {interior[-1]:.2f}"])

    table = pd.DataFrame({
        "Bin":              np.arange(1, k + 1),
        "Range":            labels,
        "Cumulative P":     [f"{i/k:.4f}" for i in range(1, k + 1)],
        "Expected (Ei)":    np.round(expected, 2),
        "Observed (Oi)":    observed.astype(int),
        "Oi - Ei":          np.round(observed - expected, 2),
        "(Oi - Ei)^2":      np.round((observed - expected) ** 2, 2),
        "(Oi - Ei)^2 / Ei": np.round(contrib, 3),
    })

    results = {"chi2_stat": chi2_stat, "p_value": p_val, "dof": dof,
               "critical_value": crit, "k": k, "n_estimated": n_estimated,
               "edges": edges, "observed": observed, "expected": expected,
               "min_expected": expected.min()}
    return table, results
# %% Goodness-of-Fit cont.
"""
Set up bins for Sturges' formula, set bin boundaries, see observed vs expected results, check
degrees of freedom, critical value, and decision.
"""

# --- Step 1: number of bins from Sturges' formula ---
k_bins, k_raw = sturges_bins(n)

print("STEP 1 - BIN SIZE (STURGES' FORMULA)")
print("=" * 55)
print(f"  n          = {n}")
print(f"  log2(n)    = {np.log2(n):.4f}")
print(f"  k = log2(n) + 1 = {k_raw:.4f}")
print(f"  bins used  = {k_bins}")
print(f"  probability per bin = 1/{k_bins} = {1/k_bins:.4f}")
print(f"  expected count per bin = n/k = {n}/{k_bins} = {n/k_bins:.2f}")
print()
print("  Rule of thumb: every expected count should be at least 5 for the chi-squared")
print(f"  approximation to hold. Here Ei = {n/k_bins:.2f}, so the requirement is "
      f"{'satisfied' if n/k_bins >= 5 else 'NOT satisfied - use fewer bins or collect more data'}.")

# --- Steps 2 and 3: bin boundaries, observed vs expected counts ---
chi_table, chi_res = equiprobable_chi_square(data, hypothesized_dist, hypothesized_params, k=k_bins)

print("\nSTEPS 2-3 - EQUIPROBABLE BINS AND THE CHI-SQUARED STATISTIC")
print("=" * 55)
print(f"  Boundaries from x = mu + z * sigma with mu = {mu_hat:.2f}, sigma = {sigma_hat:.2f}")
print(f"  (the z-values are the standard-normal quantiles at 1/{k_bins}, 2/{k_bins}, ...)")
print()
print(f"  Totals: observed = {int(chi_res['observed'].sum())}, expected = {chi_res['expected'].sum():.0f}")
print(f"  Chi-squared statistic = {chi_res['chi2_stat']:.4f}")
chi_table

# --- Step 4: degrees of freedom, critical value, decision ---
print(f"\nSTEP 4 - DECISION: DOES THE {hypothesized_name.upper()} DISTRIBUTION FIT?")
print("=" * 55)
print(f"  Degrees of freedom  df = k - 1 - p = {chi_res['k']} - 1 - {chi_res['n_estimated']} = {chi_res['dof']}")
print(f"  Calculated chi^2       = {chi_res['chi2_stat']:.4f}")
print(f"  Critical value at alpha = {ALPHA}, df = {chi_res['dof']}:  {chi_res['critical_value']:.4f}")
print(f"  p-value                = {chi_res['p_value']:.4f}")
print()

if chi_res["chi2_stat"] < chi_res["critical_value"]:
    print(f"  Since {chi_res['chi2_stat']:.4f} < {chi_res['critical_value']:.4f}, we FAIL TO REJECT H0.")
    print(f"  VERDICT: the {hypothesized_name} distribution is an ACCEPTABLE fit to the data")
    print(f"           (p = {chi_res['p_value']:.4f} >= {ALPHA}).")
else:
    print(f"  Since {chi_res['chi2_stat']:.4f} >= {chi_res['critical_value']:.4f}, we REJECT H0.")
    print(f"  VERDICT: the {hypothesized_name} distribution is NOT an acceptable fit")
    print(f"           (p = {chi_res['p_value']:.4f} < {ALPHA}).")

print()
print(f"  Remember: failing to reject H0 does not prove the data are {hypothesized_name}. It means")
print(f"  the observed counts are consistent with a {hypothesized_name} distribution at this sample size.")

# %% Visualize the results from the GoF test.

# --- Visualize the bins behind the test ---
edges_plot = chi_res["edges"].copy()
edges_plot[0]  = min(data.min(), edges_plot[1]  - 1)
edges_plot[-1] = max(data.max(), edges_plot[-2] + 1)

fig, ax = plt.subplots(figsize=(11, 5))
ax.hist(data, bins=edges_plot, color="steelblue", edgecolor="white", alpha=0.75, label="Observed counts")
ax.axhline(n / k_bins, color="red", linestyle="--", linewidth=2,
           label=f"Expected count per bin = {n/k_bins:.2f}")
for b in chi_res["edges"][1:-1]:
    ax.axvline(b, color="grey", linestyle=":", linewidth=1)

for i, (o, left, right) in enumerate(zip(chi_res["observed"], edges_plot[:-1], edges_plot[1:])):
    ax.text((left + right) / 2, o + 0.4, f"{int(o)}", ha="center", fontsize=10, color="black")

ax.set_title(f"Equiprobable bins for the chi-squared test (k = {k_bins}, each bin P = 1/{k_bins})")
ax.set_xlabel("Value"); ax.set_ylabel("Count"); ax.legend()
plt.tight_layout()
plt.show()

# %% Secondary check: Komogrov-Smirnov test.

# --- K-S test on the hypothesized distribution ---
ks_stat, ks_p = stats.kstest(data, hypothesized_dist.cdf, args=hypothesized_params)

print(f"KOLMOGOROV-SMIRNOV TEST ({hypothesized_name})")
print("=" * 55)
print("H0: the data follow the fitted distribution")
print(f"  K-S statistic (largest gap between ECDF and CDF) : {ks_stat:.4f}")
print(f"  p-value                                          : {ks_p:.4f}")
print()
if ks_p >= ALPHA:
    print(f"  -> p = {ks_p:.4f} >= {ALPHA}: fail to reject H0. Consistent with the chi-squared result.")
else:
    print(f"  -> p = {ks_p:.4f} < {ALPHA}: reject H0. This disagrees with the chi-squared result;")
    print("     inspect the diagnostic plots below before deciding.")

# %% Visual verification of the fit.

"""
Four standard diagnostics:

1. Histogram with the fitted PDF overlaid - a quick sanity check on shape.
2. Empirical CDF vs theoretical CDF - the largest vertical gap *is* the K-S statistic.
3. P-P plot - empirical vs theoretical cumulative *probabilities*, sensitive to deviations in the middle of the distribution.
4. Q-Q plot - empirical vs theoretical *quantiles*, sensitive to deviations in the tails. Tails matter in simulation because they drive rare-event and worst-case results.

A good fit puts the points on the 45-degree reference line in both the P-P and Q-Q plots.
"""

def fit_diagnostics(sample, dist, params, title):
    """Four-panel goodness-of-fit diagnostic figure."""
    sorted_s  = np.sort(np.asarray(sample, dtype=float))
    m         = len(sorted_s)
    emp_cdf   = (np.arange(1, m + 1) - 0.5) / m          # Hazen plotting position
    theo_cdf  = dist.cdf(sorted_s, *params)
    theo_quan = dist.ppf(emp_cdf, *params)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    ax = axes[0, 0]
    ax.hist(sample, bins="sturges", density=True, color="steelblue",
            edgecolor="white", alpha=0.7, label="Data")
    x_grid = np.linspace(sorted_s[0], sorted_s[-1], 300)
    ax.plot(x_grid, dist.pdf(x_grid, *params), "r-", linewidth=2, label=f"Fitted {title}")
    ax.set_title(f"Histogram vs Fitted PDF ({title})")
    ax.set_xlabel("Value"); ax.set_ylabel("Density"); ax.legend()

    ax = axes[0, 1]
    ax.step(sorted_s, emp_cdf, where="post", color="steelblue", label="Empirical CDF")
    ax.plot(sorted_s, theo_cdf, "r-", linewidth=2, label=f"Theoretical CDF ({title})")
    gap_idx = np.argmax(np.abs(emp_cdf - theo_cdf))
    ax.vlines(sorted_s[gap_idx], theo_cdf[gap_idx], emp_cdf[gap_idx],
              color="darkorange", linewidth=3, label="Largest gap (K-S statistic)")
    ax.set_title("Empirical CDF vs Theoretical CDF")
    ax.set_xlabel("Value"); ax.set_ylabel("Cumulative probability"); ax.legend()

    ax = axes[1, 0]
    ax.scatter(emp_cdf, theo_cdf, s=14, color="steelblue", alpha=0.7)
    ax.plot([0, 1], [0, 1], "r--", linewidth=1.5, label="45 degree reference")
    ax.set_title("P-P Plot")
    ax.set_xlabel("Empirical probability"); ax.set_ylabel("Theoretical probability"); ax.legend()

    ax = axes[1, 1]
    ax.scatter(theo_quan, sorted_s, s=14, color="steelblue", alpha=0.7)
    lims = [min(np.nanmin(theo_quan), sorted_s.min()), max(np.nanmax(theo_quan), sorted_s.max())]
    ax.plot(lims, lims, "r--", linewidth=1.5, label="45 degree reference")
    ax.set_title("Q-Q Plot")
    ax.set_xlabel("Theoretical quantiles"); ax.set_ylabel("Empirical quantiles"); ax.legend()

    plt.tight_layout()
    plt.show()


fit_diagnostics(data, hypothesized_dist, hypothesized_params, hypothesized_name)

# %% Finding the best fit distribution.

# --- Candidate distributions (slide 12) ---
candidate_distributions = {
    "Normal":      stats.norm,
    "Lognormal":   stats.lognorm,
    "Gamma":       stats.gamma,
    "Weibull":     stats.weibull_min,
    "Exponential": stats.expon,
    "Uniform":     stats.uniform,
    "Triangular":  stats.triang,
    "Beta":        stats.beta,
    "Logistic":    stats.logistic,
}

fitted_params = {}
for name, dist in candidate_distributions.items():
    try:
        fitted_params[name] = dist.fit(data)
    except Exception as e:
        print(f"  {name}: fit failed ({e})")

print("FITTED PARAMETERS (scipy shape / loc / scale convention)")
print("=" * 55)
for name, params in fitted_params.items():
    print(f"  {name:12s}: {fmt_params(params)}")

# --- Score every candidate with the same chi-squared procedure used in Section 5 ---
rows = []
for name, dist in candidate_distributions.items():
    if name not in fitted_params:
        continue
    params = fitted_params[name]
    try:
        _, res = equiprobable_chi_square(data, dist, params, k=k_bins)
    except Exception as e:
        print(f"  {name}: chi-squared failed ({e})")
        continue
    ks_s, ks_pv = stats.kstest(data, dist.cdf, args=params)
    rows.append({
        "Distribution": name,
        "n params":     len(params),
        "Chi2 stat":    round(res["chi2_stat"], 4),
        "Chi2 df":      res["dof"],
        "Chi2 p-value": round(res["p_value"], 4) if np.isfinite(res["p_value"]) else np.nan,
        "KS stat":      round(ks_s, 4),
        "KS p-value":   round(ks_pv, 4),
        "Acceptable (p >= 0.05)": "YES" if (np.isfinite(res["p_value"]) and res["p_value"] >= ALPHA) else "NO",
        "Params":       tuple(round(p, 4) for p in params),
    })

results_df = (pd.DataFrame(rows)
                .sort_values("Chi2 p-value", ascending=False, na_position="last")
                .reset_index(drop=True))
results_df.insert(0, "Rank", range(1, len(results_df) + 1))

print(f"GOODNESS-OF-FIT COMPARISON (equiprobable chi-squared, k = {k_bins} bins, ranked by chi-squared p-value)")
print("=" * 55)
print("Note: distributions with different numbers of parameters have different df,")
print("      which is why a lower statistic does not always mean a higher p-value.")


results_df.drop(columns=["Params"])

# --- Best fit and where the hypothesized Normal distribution ranks ---
best_row    = results_df.iloc[0]
best_name   = best_row["Distribution"]
best_params = fitted_params[best_name]
best_dist   = candidate_distributions[best_name]

normal_row  = results_df[results_df["Distribution"] == "Normal"].iloc[0]

print("BEST FIT")
print("=" * 55)
print(f"  Highest chi-squared p-value: {best_name}")
print(f"    parameters   : {fmt_params(best_params)}")
print(f"    chi^2        : {best_row['Chi2 stat']} (df = {best_row['Chi2 df']}), p = {best_row['Chi2 p-value']}")
print(f"    K-S          : {best_row['KS stat']}, p = {best_row['KS p-value']}")
print()
print(f"  Normal ranks #{int(normal_row['Rank'])} of {len(results_df)} "
      f"(chi^2 p = {normal_row['Chi2 p-value']}, acceptable: {normal_row['Acceptable (p >= 0.05)']})")
print()
n_acceptable = (results_df["Acceptable (p >= 0.05)"] == "YES").sum()
print(f"  {n_acceptable} of {len(results_df)} candidates are statistically acceptable at alpha = {ALPHA}:")
print("   ", ", ".join(results_df.loc[results_df["Acceptable (p >= 0.05)"] == "YES", "Distribution"]))
print()
print("  Among acceptable fits, prefer the simplest distribution that makes sense for the quantity being modeled rather than simply the top row of the table.")

# %% Compare the other distribution candidates.

# --- Compare the top candidates visually ---
top_k = min(4, len(results_df))
top_names = results_df.head(top_k)["Distribution"].tolist()
colors = ["red", "darkorange", "green", "purple"]

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
x_grid = np.linspace(data.min(), data.max(), 300)

ax = axes[0]
ax.hist(data, bins="sturges", density=True, color="steelblue", edgecolor="white", alpha=0.6, label="Data")
for c, name in zip(colors, top_names):
    ax.plot(x_grid, candidate_distributions[name].pdf(x_grid, *fitted_params[name]),
            color=c, linewidth=2, label=name)
ax.set_title(f"Top {top_k} fitted PDFs over the data")
ax.set_xlabel("Value"); ax.set_ylabel("Density"); ax.legend()

ax = axes[1]
sorted_data = np.sort(data)
emp_cdf = (np.arange(1, n + 1) - 0.5) / n
ax.step(sorted_data, emp_cdf, where="post", color="steelblue", linewidth=2, label="Empirical CDF")
for c, name in zip(colors, top_names):
    ax.plot(sorted_data, candidate_distributions[name].cdf(sorted_data, *fitted_params[name]),
            color=c, linewidth=1.8, linestyle="--", label=name)
ax.set_title("Empirical CDF vs top fitted CDFs")
ax.set_xlabel("Value"); ax.set_ylabel("Cumulative probability"); ax.legend()

plt.tight_layout()
plt.show()

# Full diagnostics for the best-ranked distribution
fit_diagnostics(data, best_dist, best_params, best_name)

# %% Generate synthetic distribution from a chosen distribution to verify.

"""
Once a distribution is selected, you can draw as many synthetic observations from it as the simulation needs. This is how a fitted input model becomes a random-variate generator inside a simulation, and it is also how you build test datasets when real data are scarce.

Set the three switches in the next cell:

SYNTH_DIST - which distribution to sample from, by name from candidate_distributions.
SYNTH_PARAMS - None to reuse the parameters fitted from the real data, or an explicit tuple in scipy's (shape..., loc, scale) convention to specify your own.
SYNTH_N - how many observations to generate.

Verifying the generated data matters. Running the same pipeline on the synthetic sample is the check that the generator does what you think it does. One subtlety: when the parameters were specified in advance rather than estimated from the sample, nothing was fitted to the data, so no degrees of freedom are lost and df = k - 1. That is why n_estimated=0 is passed below in that case.
"""

# ============================ SETTINGS ============================
SYNTH_DIST   = "Normal"   # any key of candidate_distributions, e.g. "Gamma", "Weibull"
SYNTH_PARAMS = None       # None -> use parameters fitted from the real data
                          # or specify your own, e.g. (50.0, 4.0) for Normal(mu=50, sigma=4)
                          #                          (2.5, 0.0, 3.0) for Gamma(shape=2.5, scale=3.0)
SYNTH_N      = 1000       # number of synthetic observations
SYNTH_SEED   = 42         # fix the seed for reproducibility
# =========================================================================

synth_dist = candidate_distributions[SYNTH_DIST]
if SYNTH_PARAMS is None:
    synth_params   = fitted_params[SYNTH_DIST]
    params_fitted  = True
else:
    synth_params   = tuple(SYNTH_PARAMS)
    params_fitted  = False

synthetic_data = synth_dist.rvs(*synth_params, size=SYNTH_N, random_state=SYNTH_SEED)

print("SYNTHETIC DATA GENERATED")
print("=" * 55)
print(f"  Distribution : {SYNTH_DIST}{fmt_params(synth_params)}")
print(f"  Source of parameters : {'fitted from the real data' if params_fitted else 'specified by the user'}")
print(f"  Sample size  : {SYNTH_N}")
print(f"  First 10 values: {np.round(synthetic_data[:10], 3)}")

# --- Verification 1: do the summary statistics match? ---
def describe(sample):
    s = np.asarray(sample, dtype=float)
    return {
        "n":        len(s),
        "Mean":     s.mean(),
        "Median":   np.median(s),
        "Std dev":  s.std(ddof=1),
        "Variance": s.var(ddof=1),
        "Skewness": stats.skew(s, bias=False),
        "Kurtosis": stats.kurtosis(s, bias=False),
        "Min":      s.min(),
        "Max":      s.max(),
    }

theoretical_mean = synth_dist.mean(*synth_params)
theoretical_std  = synth_dist.std(*synth_params)

comparison = pd.DataFrame({
    "Original data":  describe(data),
    "Synthetic data": describe(synthetic_data),
}).round(4)

print("\nVERIFICATION 1 - SUMMARY STATISTICS")
print("=" * 55)
print(f"  Theoretical mean of {SYNTH_DIST}{fmt_params(synth_params, 3)} = {theoretical_mean:.4f}")
print(f"  Theoretical sd   of {SYNTH_DIST}{fmt_params(synth_params, 3)} = {theoretical_std:.4f}")
print(f"  Synthetic sample mean = {synthetic_data.mean():.4f}, sd = {synthetic_data.std(ddof=1):.4f}")
print()
print("  The synthetic sample should match the THEORETICAL values closely (it is drawn")
print("  from that distribution). It matches the original data only as well as the")
print("  distribution itself describes the original data.")
comparison

# --- Verification 2: run the chi-squared test on the synthetic sample ---
n_est = len(synth_params) if params_fitted else 0
k_synth, k_synth_raw = sturges_bins(len(synthetic_data))

synth_table, synth_res = equiprobable_chi_square(
    synthetic_data, synth_dist, synth_params, k=k_synth, n_estimated=n_est
)

print(f"\nVERIFICATION 2 - CHI-SQUARED TEST ON THE SYNTHETIC SAMPLE")
print("=" * 55)
print(f"  H0: the synthetic sample follows {SYNTH_DIST}{fmt_params(synth_params)}")
print(f"  Sturges: k = log2({len(synthetic_data)}) + 1 = {k_synth_raw:.2f} -> {k_synth} bins, "
      f"Ei = {len(synthetic_data)/k_synth:.2f} per bin")
print(f"  Parameters estimated from this sample: p = {n_est} "
      f"({'fitted' if params_fitted else 'specified in advance, so no df are lost'})")
print(f"  chi^2 = {synth_res['chi2_stat']:.4f}, df = {synth_res['dof']}, "
      f"critical = {synth_res['critical_value']:.4f}, p = {synth_res['p_value']:.4f}")
print()
if synth_res["p_value"] >= ALPHA:
    print(f"  -> FAIL TO REJECT H0. The generator is producing data consistent with {SYNTH_DIST}.")
else:
    print(f"  -> REJECT H0. The synthetic sample does not match the requested distribution;")
    print("     check the parameter tuple and the scipy shape/loc/scale convention.")

ks_s_syn, ks_p_syn = stats.kstest(synthetic_data, synth_dist.cdf, args=synth_params)
print(f"  K-S check: statistic = {ks_s_syn:.4f}, p = {ks_p_syn:.4f}")
synth_table

# %% Visual comparison of the synthetic data.

# --- Verification 3: visual comparison ---
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

ax = axes[0]
ax.hist(data, bins="sturges", density=True, alpha=0.5, color="steelblue",
        edgecolor="white", label=f"Original data (n = {n})")
ax.hist(synthetic_data, bins="sturges", density=True, alpha=0.5, color="darkorange",
        edgecolor="white", label=f"Synthetic data (n = {SYNTH_N})")
x_grid = np.linspace(min(data.min(), synthetic_data.min()),
                     max(data.max(), synthetic_data.max()), 300)
ax.plot(x_grid, synth_dist.pdf(x_grid, *synth_params), "r-", linewidth=2,
        label=f"{SYNTH_DIST} PDF")
ax.set_title("Original vs Synthetic Data")
ax.set_xlabel("Value"); ax.set_ylabel("Density"); ax.legend()

ax = axes[1]
m_syn   = len(synthetic_data)
emp_syn = (np.arange(1, m_syn + 1) - 0.5) / m_syn
ax.scatter(synth_dist.ppf(emp_syn, *synth_params), np.sort(synthetic_data),
           s=8, color="darkorange", alpha=0.6)
lims = [synthetic_data.min(), synthetic_data.max()]
ax.plot(lims, lims, "r--", linewidth=1.5, label="45 degree reference")
ax.set_title(f"Q-Q Plot: synthetic sample vs {SYNTH_DIST}")
ax.set_xlabel("Theoretical quantiles"); ax.set_ylabel("Synthetic quantiles"); ax.legend()

plt.tight_layout()
plt.show()


# --- Generate from a distribution with parameters YOU choose ---
# Change the three lines below and re-run. The verification is identical, but note
# that n_estimated = 0 because nothing was fitted to the generated sample.

demo_dist   = stats.gamma          # e.g. stats.expon, stats.weibull_min, stats.lognorm
demo_params = (2.5, 0.0, 3.0)      # Gamma(shape = 2.5, loc = 0, scale = 3.0)
demo_n      = 500

demo_sample = demo_dist.rvs(*demo_params, size=demo_n, random_state=7)
demo_k, _   = sturges_bins(demo_n)
_, demo_res = equiprobable_chi_square(demo_sample, demo_dist, demo_params,
                                      k=demo_k, n_estimated=0)

print("Class Test")
print("=" * 55)
print(f"  Generated {demo_n} values from Gamma(shape=2.5, scale=3.0)")
print(f"  Sample mean = {demo_sample.mean():.4f}  (theoretical {demo_dist.mean(*demo_params):.4f})")
print(f"  Sample sd   = {demo_sample.std(ddof=1):.4f}  (theoretical {demo_dist.std(*demo_params):.4f})")
print(f"  Chi-squared: statistic = {demo_res['chi2_stat']:.4f}, df = {demo_res['dof']}, "
      f"p = {demo_res['p_value']:.4f}  -> "
      f"{'generator verified' if demo_res['p_value'] >= ALPHA else 'mismatch, check the parameters'}")

fig, ax = plt.subplots(figsize=(9, 4.5))
ax.hist(demo_sample, bins="sturges", density=True, color="seagreen", edgecolor="white", alpha=0.7,
        label="Generated sample")
xg = np.linspace(demo_sample.min(), demo_sample.max(), 300)
ax.plot(xg, demo_dist.pdf(xg, *demo_params), "r-", linewidth=2, label="Requested PDF")
ax.set_title("Generated sample vs the distribution it was requested from")
ax.set_xlabel("Value"); ax.set_ylabel("Density"); ax.legend()
plt.tight_layout()
plt.show()
# %% Sensitivity of Simulation Output to the Input Distribution

"""
The last step of the pipeline is checking the sensitivity of inputs on simulation outputs. The question it answers:
if two distributions both pass the goodness-of-fit tests, does the choice between them actually change what the 
simulation says? If it does not, the input-modeling decision is not worth agonizing over. If it does, 
it matters more than which one had the higher p-value.

The downstream model used here is a generic single-server FIFO queue: interarrival times are exponential,
service times are drawn from a fitted distribution, and the output is average waiting time. Replace 'run_queue_simulation'
with your own model to apply the same pattern to a term project.
"""

def run_queue_simulation(service_time_dist, service_params, n_customers=2000,
                         arrival_rate=0.28, seed=None):
    """
    Single-server FIFO queue. Interarrival times ~ Exponential(arrival_rate).
    Service times are drawn from the given fitted distribution.
    Returns the average customer waiting time.
    """
    rng = np.random.default_rng(seed)
    interarrival_times = rng.exponential(1 / arrival_rate, size=n_customers)
    arrival_times = np.cumsum(interarrival_times)

    service_times = service_time_dist.rvs(*service_params, size=n_customers, random_state=rng)
    service_times = np.clip(service_times, 1e-6, None)   # guard against negative draws

    departure_times = np.zeros(n_customers)
    wait_times = np.zeros(n_customers)
    departure_times[0] = arrival_times[0] + service_times[0]
    for i in range(1, n_customers):
        service_start = max(arrival_times[i], departure_times[i - 1])
        wait_times[i] = service_start - arrival_times[i]
        departure_times[i] = service_start + service_times[i]

    return wait_times.mean()

# --- Sensitivity to the choice of input distribution ---
top_candidates = results_df.head(4)["Distribution"].tolist()
n_reps = 20

print(f"SENSITIVITY TO INPUT DISTRIBUTION CHOICE (avg waiting time, {n_reps} replications)")
print("=" * 55)
dist_choice_results = {}
for name in top_candidates:
    dist   = candidate_distributions[name]
    params = fitted_params[name]
    reps   = [run_queue_simulation(dist, params, seed=1000 + r) for r in range(n_reps)]
    dist_choice_results[name] = reps
    print(f"  {name:12s}: mean wait = {np.mean(reps):8.4f}   (std across reps = {np.std(reps):.4f})")

fig, ax = plt.subplots(figsize=(9, 5))
try:   # matplotlib >= 3.9
    ax.boxplot(list(dist_choice_results.values()), tick_labels=list(dist_choice_results.keys()))
except TypeError:  # older matplotlib
    ax.boxplot(list(dist_choice_results.values()), labels=list(dist_choice_results.keys()))
ax.set_title("Simulation Output Sensitivity to Input Distribution Choice")
ax.set_ylabel("Average waiting time"); ax.set_xlabel("Distribution used for service times")
plt.tight_layout()
plt.show()

print("Boxes far apart: the input-distribution choice meaningfully changes the answer.")
print("Boxes overlapping: the simulation output is robust to this choice.")

# --- Sensitivity to parameter estimation error (+/- 20% on the scale parameter) ---
perturbation_pcts  = [-0.20, -0.10, 0.0, 0.10, 0.20]
param_sens_results = {}

base_params = list(best_params)
scale_idx   = len(base_params) - 1     # scipy convention: scale is the last element

print(f"PERTURBING THE SCALE PARAMETER OF {best_name.upper()} (base = {base_params[scale_idx]:.4f})")
print("=" * 55)
for pct in perturbation_pcts:
    perturbed = base_params.copy()
    perturbed[scale_idx] = base_params[scale_idx] * (1 + pct)
    reps = [run_queue_simulation(best_dist, tuple(perturbed), seed=2000 + r) for r in range(n_reps)]
    param_sens_results[f"{pct:+.0%}"] = reps
    print(f"  scale {pct:+.0%} ({perturbed[scale_idx]:.4f}): mean wait = {np.mean(reps):8.4f}")

fig, ax = plt.subplots(figsize=(9, 5))
means = [np.mean(v) for v in param_sens_results.values()]
stds  = [np.std(v)  for v in param_sens_results.values()]
ax.errorbar(list(param_sens_results.keys()), means, yerr=stds, marker="o", capsize=4, color="steelblue")
ax.set_title(f"Output Sensitivity to the {best_name} Scale Parameter (+/- 20%)")
ax.set_xlabel("Perturbation to the scale parameter"); ax.set_ylabel("Average waiting time")
plt.tight_layout()
plt.show()

print("A steep slope means the output is highly sensitive to precise parameter estimation,")
print("so more data or a tighter fit is worth the effort. A flat line means small estimation")
print("error will not change the conclusions.")
# %% Calculating parameters for questions 1c, 1d, and 1e.
avg_arrival_rate = 1 / (mean / 60)
service_rate = 12

# Question 1c
print(f"The average arrival rate is {avg_arrival_rate:.4f} arrivals per hour.")
print()
rho = avg_arrival_rate / service_rate
print(f"The utilization of the system is {rho:.4f}.")
print()

# Question 1d
avg_patients_in_system = (rho / (1 - rho))
print(f"The average number of patients in the system is {avg_patients_in_system:.4f} patients.")
print()

# Question 1e
avg_waiting_time_queue = rho / (service_rate * (1 - rho))
print(f"The average waiting time in the queue is {avg_waiting_time_queue:.4f} hours.")
print()
# %%
