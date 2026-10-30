# %% Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.oneway import anova_oneway
from statsmodels.stats.proportion import proportion_confint

ALPHA = 0.05
pd.set_option('display.precision', 3)

# %% Import the results in an excel file.
FILE = 'Homework_2_Results.xlsx'   # change as needed for your own AnyLogic results

df = pd.read_excel(FILE)
df['Policy'] = df['Iteration'].map(lambda c: f'c = {c:g}')
print(df.shape)
df.head()
# %% Check that the results were random.

# If the random draws were evaluated only once per iteration, every replication of a policy would have identical inputs and 
# identical outputs, and the analysis below would be meaningless. The standard deviation of each input within each policy must therefore be greater than zero.

inputs = ['Iteration','PeakSubscribers', 'PeakMonth', 'SubscribersAtM36']
check = df.groupby('Iteration')[inputs].agg(['mean', 'std', 'min', 'max'])
check

# Assertion guarded for the case of a single policy.
if df['Iteration'].max() > 1:
    assert (df.groupby('Iteration')[inputs].std() > 0).all().all(), \
        'Inputs do not vary within a policy: the replications are identical. Check how the random values are drawn.'
    print('The inputs vary between replications within every policy.')
else:
    print('Only one policy is present in the data, so there is no check for variance amongst replications in different polices.')
# %% Descriptive statistics.

# For each policy we report the sample size, mean, standard deviation, and a 95% confidence interval for the mean peak.

def ci_half_width(x, conf=0.95):
    n = len(x)
    return stats.t.ppf((1 + conf) / 2, n - 1) * x.std(ddof=1) / np.sqrt(n)

def t_val(x, conf=0.95):
    n = len(x)
    return stats.t.ppf((1 + conf) / 2, n - 1)

# Peak subscribers statistics.
summary = df.groupby('Iteration')['PeakSubscribers'].agg(
    n='count', mean='mean', sd='std', min='min', max='max')
summary['ci95_half_width'] = df.groupby('Iteration')['PeakSubscribers'].apply(ci_half_width)
summary['t_val'] = df.groupby('Iteration')['PeakSubscribers'].apply(t_val)
summary['ci95_low'] = summary['mean'] - summary['ci95_half_width']
summary['ci95_high'] = summary['mean'] + summary['ci95_half_width']
summary.round(1)

policies = sorted(df['Iteration'].unique())
groups = [df.loc[df['Iteration'] == c, 'PeakSubscribers'].values for c in policies]
labels = [f'{c:g}' for c in policies]

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

axes[0].boxplot(groups, showmeans=True)
axes[0].set_xticks(range(1, len(labels) + 1))
axes[0].set_xticklabels(labels)
for i, g in enumerate(groups, start=1):
    axes[0].scatter(np.random.normal(i, 0.05, len(g)), g, s=12, alpha=0.5)
axes[0].set_xlabel('Iteration (policy)')
axes[0].set_ylabel('Peak Subscribers')
axes[0].set_title('Peak of the users subscribed, 100 replications per policy')

axes[1].errorbar(labels, summary['mean'], yerr=summary['ci95_half_width'], fmt='o', capsize=6)
axes[1].set_xlabel('Iteration (policy)')
axes[1].set_ylabel('Mean peak subscribers')
axes[1].set_title('Mean peak with 95% confidence intervals')

for ax in axes:
    ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()

print(f"The mean of the Peak Subscribers: {summary['mean'].item():.3f} people.")
print(f"The standard deviation of the Peak Subscribers: {summary['sd'].item():.3f} people.")
print(f"The 95% CI of the Peak Subscribers: {summary['ci95_low'].item():.3f}, {summary['ci95_high'].item():.3f} people.")

# Check to see if sufficient number of replications were used.
required_margin = ALPHA
if summary['ci95_half_width'].item() / summary['mean'].item() > required_margin:
    print(f"100 replications is NOT sufficient for 95% confidence interval and a {(required_margin * 100):.1f}% margin on the mean.")
    replications_required= ((summary['t_val'].item() * summary['sd'].item()) / (required_margin * summary['mean'].item())) ** 2
    print(f"The required replications are {np.ceil(replications_required)} replications.")
else:
    print(f"100 replications is sufficient for 95% confidence interval and a {(required_margin * 100):.1f}% margin on the mean.")
    replications_required= ((summary['t_val'].item() * summary['sd'].item()) / (required_margin * summary['mean'].item())) ** 2
    print(f"The required replications are {np.ceil(replications_required)} replications.")

required_subscribers = 20000
successful_runs = df[df['PeakSubscribers'] >= required_subscribers].groupby('Iteration').size().item()
print(f"{successful_runs} out of {df.groupby('Iteration')['PeakSubscribers'].count().item()} runs had at least {required_subscribers} subscribers.")
lower, upper = proportion_confint(successful_runs, df.groupby('Iteration')['PeakSubscribers'].count().item(), alpha=ALPHA, method="normal")
print(f"95% Confidence Interval: ({lower:.4f}, {upper:.4f})")

# Peak month statistics.

summary = df.groupby('Iteration')['PeakMonth'].agg(
    n='count', mean='mean', sd='std', min='min', max='max')
summary['ci95_half_width'] = df.groupby('Iteration')['PeakMonth'].apply(ci_half_width)
summary['t_val'] = df.groupby('Iteration')['PeakMonth'].apply(t_val)
summary['ci95_low'] = summary['mean'] - summary['ci95_half_width']
summary['ci95_high'] = summary['mean'] + summary['ci95_half_width']
summary.round(1)

policies = sorted(df['Iteration'].unique())
groups = [df.loc[df['Iteration'] == c, 'PeakMonth'].values for c in policies]
labels = [f'{c:g}' for c in policies]

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

axes[0].boxplot(groups, showmeans=True)
axes[0].set_xticks(range(1, len(labels) + 1))
axes[0].set_xticklabels(labels)
for i, g in enumerate(groups, start=1):
    axes[0].scatter(np.random.normal(i, 0.05, len(g)), g, s=12, alpha=0.5)
axes[0].set_xlabel('Iteration (policy)')
axes[0].set_ylabel('Peak Months')
axes[0].set_title('Peak months of the users subscribed, 100 replications per policy')

axes[1].errorbar(labels, summary['mean'], yerr=summary['ci95_half_width'], fmt='o', capsize=6)
axes[1].set_xlabel('Iteration (policy)')
axes[1].set_ylabel('Mean peak months')
axes[1].set_title('Mean peak with 95% confidence intervals')

for ax in axes:
    ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()

print(f"The mean of the Peak Months: {summary['mean'].item():.3f} months.")
print(f"The standard deviation of the Peak Months: {summary['sd'].item():.3f} months.")
print(f"The 95% CI of the Peak Months: {summary['ci95_low'].item():.3f}, {summary['ci95_high'].item():.3f} months.")

# Check to see if sufficient number of replications were used.
if summary['ci95_half_width'].item() / summary['mean'].item() > required_margin:
    print(f"100 replications is NOT sufficient for 95% confidence interval and a {(required_margin * 100):.1f}% margin on the mean.")
    replications_required= ((summary['t_val'].item() * summary['sd'].item()) / (required_margin * summary['mean'].item())) ** 2
    print(f"The required replications are {np.ceil(replications_required)} replications.")
else:
    print(f"100 replications is sufficient for 95% confidence interval and a {(required_margin * 100):.1f}% margin on the mean.")
    replications_required= ((summary['t_val'].item() * summary['sd'].item()) / (required_margin * summary['mean'].item())) ** 2
    print(f"The required replications are {np.ceil(replications_required)} replications.")

early_months = df[df['PeakMonth'] < summary['mean'].item()].groupby('Iteration').size().item()
later_months = df[df['PeakMonth'] > summary['mean'].item()].groupby('Iteration').size().item()
upper_outlier_months = df[df['PeakMonth'] > summary['ci95_high'].item()].groupby('Iteration').size().item()
lower_outlier_months = df[df['PeakMonth'] < summary['ci95_low'].item()].groupby('Iteration').size().item()
print(f"{early_months} out of {df.groupby('Iteration')['PeakMonth'].count().item()} peak months were earlier than the mean month {summary['mean'].item():.3f}.")
print(f"{later_months} out of {df.groupby('Iteration')['PeakMonth'].count().item()} peak months were later than the mean month {summary['mean'].item():.3f}.")
print(f"{upper_outlier_months} out of {df.groupby('Iteration')['PeakMonth'].count().item()} peak months were later than the upper 95% CI month {summary['ci95_high'].item():.3f}.")
print(f"{lower_outlier_months} out of {df.groupby('Iteration')['PeakMonth'].count().item()} peak months were earlier than the lower 95% CI month {summary['ci95_low'].item():.3f}.")

# Subscribers at month 36 statistics.

summary = df.groupby('Iteration')['SubscribersAtM36'].agg(
    n='count', mean='mean', sd='std', min='min', max='max')
summary['ci95_half_width'] = df.groupby('Iteration')['SubscribersAtM36'].apply(ci_half_width)
summary['t_val'] = df.groupby('Iteration')['SubscribersAtM36'].apply(t_val)
summary['ci95_low'] = summary['mean'] - summary['ci95_half_width']
summary['ci95_high'] = summary['mean'] + summary['ci95_half_width']
summary.round(1)

policies = sorted(df['Iteration'].unique())
groups = [df.loc[df['Iteration'] == c, 'SubscribersAtM36'].values for c in policies]
labels = [f'{c:g}' for c in policies]

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

axes[0].boxplot(groups, showmeans=True)
axes[0].set_xticks(range(1, len(labels) + 1))
axes[0].set_xticklabels(labels)
for i, g in enumerate(groups, start=1):
    axes[0].scatter(np.random.normal(i, 0.05, len(g)), g, s=12, alpha=0.5)
axes[0].set_xlabel('Iteration (policy)')
axes[0].set_ylabel('Subscribers at Month 36')
axes[0].set_title('Users subscribed at month 36, 100 replications per policy')

axes[1].errorbar(labels, summary['mean'], yerr=summary['ci95_half_width'], fmt='o', capsize=6)
axes[1].set_xlabel('Iteration (policy)')
axes[1].set_ylabel('Mean subscribers at month 36')
axes[1].set_title('Mean subscribers at month 36 with 95% confidence intervals')

for ax in axes:
    ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()

print(f"The mean of the subscribers at month 36: {summary['mean'].item():.3f} people.")
print(f"The standard deviation of the subscribers at month 36: {summary['sd'].item():.3f} people.")
print(f"The 95% CI of the subscribers at month 36: {summary['ci95_low'].item():.3f}, {summary['ci95_high'].item():.3f} people.")

# Check to see if sufficient number of replications were used.
if summary['ci95_half_width'].item() / summary['mean'].item() > required_margin:
    print(f"100 replications is NOT sufficient for 95% confidence interval and a {(required_margin * 100):.1f}% margin on the mean.")
    replications_required= ((summary['t_val'].item() * summary['sd'].item()) / (required_margin * summary['mean'].item())) ** 2
    print(f"The required replications are {np.ceil(replications_required)} replications.")
else:
    print(f"100 replications is sufficient for 95% confidence interval and a {(required_margin * 100):.1f}% margin on the mean.")
    replications_required= ((summary['t_val'].item() * summary['sd'].item()) / (required_margin * summary['mean'].item())) ** 2
    print(f"The required replications are {np.ceil(replications_required)} replications.")     
# %% Hypothesis testing.

# Let mu_i be the mean peak number of infectious people under policy i.

# H_0: mu_0.5 = mu_0.75 = mu_1.0 = mu_1.25 (the policy has no effect on the mean peak)
# H_1: at least one mean differs from the others

# Significance level: alpha = 0.05.

print('Shapiro-Wilk test of normality within each policy')
for c, g in zip(policies, groups):
    w, p = stats.shapiro(g)
    print(f'  c = {c:<5g} W = {w:.3f}, p = {p:.3f}  ->  {"normality not rejected" if p > ALPHA else "normality rejected"}')

lev_stat, lev_p = stats.levene(*groups, center='median')
print(f'\nLevene test of equal variances: W = {lev_stat:.3f}, p = {lev_p:.3f}  ->  '
      f'{"equal variances not rejected" if lev_p > ALPHA else "equal variances rejected"}')

# %% One-way ANOVA

# Assumptions:
# 1. Independence: each run uses its own random draws, so the observations are independent by design.
# 2. Normality of the response within each group (or of the residuals): Shapiro-Wilk test and a Q-Q plot.
# 3. Equal variances across groups: Levene's test.

model = ols('PeakInfectious ~ C(ContactRateInfectious)', data=df).fit()

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
sm.qqplot(model.resid, line='s', ax=axes[0])
axes[0].set_title('Q-Q plot of ANOVA residuals')
axes[1].scatter(model.fittedvalues, model.resid, alpha=0.6)
axes[1].axhline(0, color='gray', lw=1)
axes[1].set_xlabel('Fitted value (policy mean)')
axes[1].set_ylabel('Residual')
axes[1].set_title('Residuals vs. fitted values')
for ax in axes:
    ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# ANOVA compares the variation between policy means with the variation within policies:

grand_mean = df['PeakInfectious'].mean()
fig, ax = plt.subplots(figsize=(9, 4.5))
for i, (c, g) in enumerate(zip(policies, groups)):
    x = np.random.normal(i, 0.06, len(g))
    ax.scatter(x, g, s=14, alpha=0.5)
    ax.hlines(g.mean(), i - 0.3, i + 0.3, colors='black', lw=2.5)
    ax.annotate('', xy=(i + 0.35, g.mean()), xytext=(i + 0.35, grand_mean),
                arrowprops=dict(arrowstyle='<->', color='crimson', lw=1.5))
ax.axhline(grand_mean, color='gray', ls='--', lw=1.5, label=f'Grand mean = {grand_mean:,.0f}')
ax.set_xticks(range(len(policies)))
ax.set_xticklabels([f'{c:g}' for c in policies])
ax.set_xlabel('ContactRateInfectious (policy)')
ax.set_ylabel('Peak infectious people')
ax.set_title('Between-policy variation (red arrows) vs. within-policy variation (scatter)')
ax.legend(loc='upper left')
ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# Before using a library, we compute every quantity in the ANOVA table directly from the formulas above.

k = len(groups)                                   # number of policies
N = sum(len(g) for g in groups)                   # total number of runs
grand_mean = np.concatenate(groups).mean()

ss_between = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in groups)
ss_within = sum(((g - g.mean()) ** 2).sum() for g in groups)
ss_total = ((np.concatenate(groups) - grand_mean) ** 2).sum()

df_between, df_within = k - 1, N - k
ms_between = ss_between / df_between
ms_within = ss_within / df_within
F_hand = ms_between / ms_within
p_hand = stats.f.sf(F_hand, df_between, df_within)     # P(F >= observed) under H0
F_crit = stats.f.ppf(1 - ALPHA, df_between, df_within)  # reject H0 if F > F_crit

print(f'Grand mean = {grand_mean:,.1f}')
print(f'Check: SS_between + SS_within = {ss_between + ss_within:,.1f}  vs  SS_total = {ss_total:,.1f}')
print(f'MS_within  = {ms_within:,.1f}  (noise variance; sd = {np.sqrt(ms_within):,.1f} people)')
print(f'MS_between = {ms_between:,.1f}')
print(f'F = {F_hand:.2f} on ({df_between}, {df_within}) df; critical F at alpha = {ALPHA} is {F_crit:.2f}')
print(f'p = {p_hand:.3g}; eta squared = SS_between / SS_total = {ss_between / ss_total:.3f}')

pd.DataFrame({'df': [df_between, df_within, N - 1],
              'sum_sq': [ss_between, ss_within, ss_total],
              'mean_sq': [ms_between, ms_within, np.nan],
              'F': [F_hand, np.nan, np.nan],
              'p': [p_hand, np.nan, np.nan]},
             index=['Between (policy)', 'Within (residual)', 'Total'])

# The library results below should match the table computed by hand.

# Quick version with SciPy
f_stat, p_val = stats.f_oneway(*groups)
print(f'SciPy f_oneway: F = {f_stat:.2f}, p = {p_val:.3g}')

# Full ANOVA table with statsmodels
anova_table = sm.stats.anova_lm(model, typ=2)
anova_table['mean_sq'] = anova_table['sum_sq'] / anova_table['df']
eta_sq = anova_table.loc['C(ContactRateInfectious)', 'sum_sq'] / anova_table['sum_sq'].sum()
print(f'Effect size eta squared = {eta_sq:.3f}')
anova_table[['df', 'sum_sq', 'mean_sq', 'F', 'PR(>F)']]

# Welch's ANOVA does not assume equal variances. Use it instead of the standard F test when Levene's test is significant,
# and use the Games-Howell procedure (Section 8) for the pairwise comparisons.

welch = anova_oneway(df['PeakInfectious'], df['Iteration'], use_var='unequal', welch_correction=True)
print(f'Welch ANOVA: F = {welch.statistic:.2f}, df = ({welch.df[0]:.0f}, {welch.df[1]:.1f}), p = {welch.pvalue:.3g}')
print('Use this result' if lev_p <= ALPHA else 'Not needed here: Levene did not reject equal variances (shown for comparison)')
# %% Post-hoc comparisons: which policies differ?

# A significant ANOVA only says that at least one mean differs. Post-hoc tests compare every pair of policies while controlling the family-wise error rate. With 4 policies there are 6 pairs.

# Tukey HSD when variances are equal
# Games-Howell when variances are unequal

tukey = pairwise_tukeyhsd(endog=df['PeakInfectious'], groups=df['Iteration'], alpha=ALPHA)
print(tukey.summary())

fig = tukey.plot_simultaneous(xlabel='Mean peak infectious people', ylabel='Iteration')
plt.title('Tukey HSD 95% simultaneous confidence intervals')
plt.grid(alpha=0.3)
plt.show()

def games_howell(data, group_col, value_col):
    # Games-Howell pairwise comparisons (unequal variances and sample sizes)
    g = data.groupby(group_col)[value_col]
    n, m, v = g.count(), g.mean(), g.var(ddof=1)
    names = list(n.index)
    k = len(names)
    rows = []
    for i in range(k):
        for j in range(i + 1, k):
            a, b = names[i], names[j]
            se = np.sqrt(v[a] / n[a] + v[b] / n[b])
            diff = m[b] - m[a]
            t = abs(diff) / se
            dof = (v[a] / n[a] + v[b] / n[b]) ** 2 / (
                (v[a] / n[a]) ** 2 / (n[a] - 1) + (v[b] / n[b]) ** 2 / (n[b] - 1))
            p = stats.studentized_range.sf(t * np.sqrt(2), k, dof)
            rows.append(dict(group1=a, group2=b, meandiff=diff, se=se, t=t, df=dof,
                             p_adj=p, reject=p < ALPHA))
    return pd.DataFrame(rows)

games_howell(df, 'Iteration', 'PeakInfectious').round(4)
# %% Interpretation

# The cell below writes the interpretation from the results computed above, so it updates automatically when the notebook is run on a different data set.
# It chooses the standard ANOVA and Tukey HSD when Levene's test does not reject equal variances, and Welch's ANOVA and Games-Howell when it does.

from IPython.display import Markdown, display
from itertools import combinations

def fmt_p(p):
    return 'p < 0.001' if p < 0.001 else f'p = {p:.3f}'

def effect_label(e):
    return 'large' if e >= 0.14 else 'medium' if e >= 0.06 else 'small'

out = []
k, N = len(policies), len(df)
sizes = ', '.join(f'{c:g} (n = {len(g)})' for c, g in zip(policies, groups))
out.append(f'**Design.** {k} policies were compared using {N} independent runs: {sizes}. '
           f'The response is the peak of the Infectious stock in each run.')

# Descriptive results
means = summary['mean'].reindex(policies)
lo, hi = means.idxmin(), means.idxmax()
trend = ('The mean peak increases steadily with the contact rate.' if means.is_monotonic_increasing else
         'The mean peak decreases steadily with the contact rate.' if means.is_monotonic_decreasing else
         'The mean peak does not change monotonically with the contact rate.')
out.append(f'**Descriptive results.** The mean peak ranges from {means[lo]:,.0f} people '
           f'(95% CI {summary.loc[lo, "ci95_low"]:,.0f} to {summary.loc[lo, "ci95_high"]:,.0f}) at a contact rate of {lo:g} '
           f'to {means[hi]:,.0f} people (95% CI {summary.loc[hi, "ci95_low"]:,.0f} to {summary.loc[hi, "ci95_high"]:,.0f}) '
           f'at a contact rate of {hi:g}, a difference of {means[hi] - means[lo]:,.0f} people. {trend} '
           f'The standard deviations within policies range from {summary["sd"].min():,.0f} to {summary["sd"].max():,.0f} people.')

# Assumptions
sw = {c: stats.shapiro(g).pvalue for c, g in zip(policies, groups)}
non_normal = [c for c, p in sw.items() if p <= ALPHA]
if non_normal:
    norm_txt = (f'Shapiro-Wilk rejects normality for contact rate(s) {", ".join(f"{c:g}" for c in non_normal)}. '
                f'ANOVA is fairly robust to moderate departures from normality when the groups are of equal size and '
                f'reasonably large, but the Q-Q plot should be inspected before relying on the result.')
else:
    norm_txt = f'Shapiro-Wilk does not reject normality for any policy (smallest p = {min(sw.values()):.3f}).'
equal_var = lev_p > ALPHA
var_txt = (f"Levene's test does not reject equal variances ({fmt_p(lev_p)}), so the standard one-way ANOVA and Tukey HSD are appropriate."
           if equal_var else
           f"Levene's test rejects equal variances ({fmt_p(lev_p)}), so Welch's ANOVA and the Games-Howell procedure are used instead of the standard ANOVA and Tukey HSD.")
out.append(f'**Assumptions.** The runs are independent by design. {norm_txt} {var_txt}')

# Omnibus test
if equal_var:
    F_used, df1, df2, p_used, test_name = f_stat, k - 1, N - k, p_val, 'One-way ANOVA'
else:
    F_used, df1, df2, p_used, test_name = welch.statistic, welch.df[0], welch.df[1], welch.pvalue, "Welch's ANOVA"
significant = p_used < ALPHA
df2_txt = f'{df2:.0f}' if equal_var else f'{df2:.1f}'
omni = (f'**{test_name}.** F({df1:.0f}, {df2_txt}) = {F_used:.2f}, {fmt_p(p_used)}; '
        f'eta squared = {eta_sq:.2f} ({effect_label(eta_sq)} effect). The policy explains about {eta_sq:.0%} of the '
        f'variation in the peak; the remaining {1 - eta_sq:.0%} comes from the uncertainty in the stochastic inputs. ')
if significant:
    omni += (f'Because p < {ALPHA}, we reject $H_0$: the mean peak is not the same under all policies.')
else:
    omni += (f'Because p >= {ALPHA}, we do not reject $H_0$: the data do not show a difference in the mean peak between '
             f'policies, and no post-hoc comparisons are needed.')
out.append(omni)

# Post-hoc comparisons
if significant:
    if equal_var:
        t = pd.DataFrame(tukey._results_table.data[1:], columns=tukey._results_table.data[0])
        pairs = pd.DataFrame({'g1': t['group1'], 'g2': t['group2'], 'diff': t['meandiff'].abs(),
                              'p': t['p-adj'], 'sig': t['reject']})
        method = 'Tukey HSD'
    else:
        gh = games_howell(df, 'Iteration', 'PeakInfectious')
        pairs = pd.DataFrame({'g1': gh['group1'], 'g2': gh['group2'], 'diff': gh['meandiff'].abs(),
                              'p': gh['p_adj'], 'sig': gh['reject']})
        method = 'Games-Howell'
    pairs['sig'] = pairs['sig'].astype(bool)
    n_sig = int(pairs['sig'].sum())
    verb = 'pair differs' if n_sig == 1 else 'pairs differ'
    lines = [f'**Post-hoc comparisons ({method}).** {n_sig} of {len(pairs)} {verb} significantly at alpha = {ALPHA}.']
    for _, r in pairs.sort_values('diff').iterrows():
        lines.append(f"* {r['g1']:g} vs. {r['g2']:g}: difference of {r['diff']:,.0f} people, {fmt_p(r['p'])} "
                     f"({'significant' if r['sig'] else 'not significant'})")
    smallest = pairs.sort_values('diff').iloc[0]
    if smallest['sig']:
        lines.append(f"\nThe smallest difference, between contact rates of {smallest['g1']:g} and {smallest['g2']:g} "
                     f"({smallest['diff']:,.0f} people, about {smallest['diff'] / means.mean():.0%} of the average peak), "
                     f"is statistically significant. Whether a difference of this size is practically important depends "
                     f"on the context, for example on the number of hospital beds available.")
    not_sig = pairs[~pairs['sig']]
    if len(not_sig):
        lines.append('\nThe data do not distinguish the following policies from each other: ' +
                     '; '.join(f"{r['g1']:g} and {r['g2']:g}" for _, r in not_sig.iterrows()) +
                     '. More replications would be needed to detect a difference of this size, if one exists.')
    out.append('\n'.join(lines))

# Conclusion
if significant:
    out.append(f'**Conclusion.** The contact rate has a statistically significant and {effect_label(eta_sq)} effect on the '
               f'peak number of infectious people. The lowest mean peak occurs at a contact rate of {lo:g} '
               f'({means[lo]:,.0f} people) and the highest at {hi:g} ({means[hi]:,.0f} people).')
else:
    out.append('**Conclusion.** With these data, there is no statistically significant evidence that the contact rate '
               'affects the mean peak number of infectious people.')

display(Markdown('\n\n'.join(out)))

# %%
