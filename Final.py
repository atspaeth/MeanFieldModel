# Final Figures
#
# This script generates all the figures from our new manuscript
# ``Model-agnostic neural mean-field with the Refractory SoftPlus
# transfer function''
from itertools import zip_longest

import matplotlib.pyplot as plt
import nest
import numpy as np
from tqdm import tqdm

from mfsupport import (AnnealedAverageConnectivity, RandomConnectivity, figure,
                       find_fps, firing_rates, fitted_curve,
                       generalization_errors, norm_err, parametrized_F_Finv,
                       relu, rs79, sigmoid, softplus_ref, softplus_ref_q_dep)

plt.ion()
if "elsevier" in plt.style.available:
    plt.style.use("elsevier")

model_names = {
    "iaf_psc_delta": "Leaky Integrate-and-Fire",
    "izhikevich": "Izhikevich",
    "hh_psc_alpha": "Hodgkin-Huxley",
}
model_line_styles = {
    "iaf_psc_delta": (1, (1, 0.5)),
    "izhikevich": (3, (6, 1)),
    "hh_psc_alpha": (2.5, (5, 1, 1, 1)),
}


# %%
# Figure 2
# ========
# The behavior of several different candidate transfer functions in
# comparison to a single set of refractory data with default parameters.

T = 1e5
q = 1.0
eta = 0.8
dt = 0.1
R, rates = firing_rates(T=T, eta=eta, q=q, dt=dt, model="iaf_psc_delta", sigma_max=10.0)

sub = R <= R[-1] / 2
tfs = {"Sigmoid": sigmoid, "ReLU": relu, "Refractory SoftPlus": softplus_ref}
x = R / 1e4

with figure("02 Refractory Softplus Extrapolation") as f:
    ax1, ax2 = f.subplots(2, 1)
    true = rs79(R, q, 10, 15, 2)
    ax1.plot(x, rates, ".", ms=1)
    ax2.plot(x, rates - true, ".", ms=1)

    for name, tf in tfs.items():
        rateshat = fitted_curve(tf, R[sub], rates[sub])(R)
        ax1.plot(x, rateshat, label=name)
        ax2.plot(x, rateshat - true, label=name)

    ax1.plot(x, true, "k:", label="Diffusion Limit")
    ax1.set_ylabel("Firing Rate (Hz)")
    ax1.set_xticks([])
    ax1.legend(ncol=2, loc="lower right")
    ax2.set_xlabel("Mean Rate of $10^4$ Presynaptic Neurons (Hz)")
    ax2.set_ylabel("Difference from\nAnalytical (Hz)")
    lim = ax2.get_ylim()[1]
    ax2.set_ylim(-lim - 10, lim)

    # Mark the boundary between training and extrapolation data.
    ax2.axvline(5, color="k", lw=0.75)
    ax1.axvline(5, color="k", lw=0.75)
    f.align_ylabels([ax1, ax2])


# %%
# Figure 3
# ========
# Three consistency curves for the LIF neuron, demonstrating that the model
# predicts a saddle-node bifurcation that we're going to be able to observe
# in the next set of simulations...

Ns = [25, 51, 75]

dt = 0.1
q = 5.0
eta = 0.8
Rb = 0.1e3
rs = np.linspace(0, 50, num=1000)
model = "iaf_psc_delta"

R, rates = firing_rates(model, q, eta=eta, dt=dt, T=1e5, M=100, sigma_max=10.0)
tf = fitted_curve(softplus_ref, R, rates)


def is_bistable(N):
    F, Finv = parametrized_F_Finv(tf.p, Rb, N, q)
    try:
        stables, _ = find_fps(50, F, Finv)
    # We'll asssume that failing to find a fixed point means that
    # the system isn't bistable anymore.
    except RuntimeError:
        return False
    return len(stables) == 2


with figure("03 Consistency Condition") as f:
    ax = f.gca()
    ax.set_aspect("equal")
    r_in = np.linspace(0, 100.0, num=1000)
    for i, N in enumerate(Ns):
        F, Finv = parametrized_F_Finv(tf.p, Rb, N, q)
        ax.plot(r_in, F(r_in), f"C{i}", label=f"$N = {N}$")[0]
        ax.plot(r_in, r_in, "k:")

        stables, unstables = find_fps(50, F, Finv)

        ax.plot(stables, stables, "ko", fillstyle="full")
        if len(unstables) > 0:
            ax.plot(unstables, unstables, "ko", fillstyle="none")

    ax.set_yticks(ax.get_xticks())
    ax.set_xlabel("Presynaptic Firing Rate (Hz)")
    ax.set_ylabel("Postsynaptic Firing Rate (Hz)")

    ax.plot([], [], "ko", fillstyle="right", label="Fixed Point")[0]
    ax.legend()

    upper, lower = Ns[2], Ns[1]
    while (upper - lower) > abs(upper + lower) * 1e-3:
        mid = (upper + lower) / 2
        p_mid = is_bistable(mid)
        if p_mid:
            upper = mid
        else:
            lower = mid
    N_star = (upper + lower) / 2

    print("Bifurcation to bistability at N =", N_star)
    hsfp = find_fps(80, *parametrized_F_Finv(tf.p, Rb, int(N_star + 1), q))[0][-1]
    print("Lower bifurcation has 0 Hz stable, saddle node at FR =", hsfp)
    ax.plot(hsfp, hsfp, "ko", fillstyle="right")


# %%
# Figure 4
# ========
# Convergence of the error as the number of neurons included in the fit
# increases, and as the total simulation time increases.

dt = 0.1
q = 1.0
sigma_max = 10.0

M = 100
Mmax = 10000
T = 1e5
Tmax = 1e7

Msubs = np.geomspace(5, Mmax / 2, num=100, dtype=int)
residuals_M = {m: [] for m in model_names}
with tqdm(total=len(model_names) * len(Msubs)) as pbar:
    for m in model_names:
        R, sd = firing_rates(
            model=m, q=q, dt=dt, T=T, M=Mmax, sigma_max=sigma_max, return_times=True
        )
        rates = sd.rates("Hz")
        for Msub in Msubs:
            idces = np.linspace(0, Mmax - 1, num=Msub, dtype=int)
            rfit = fitted_curve(softplus_ref, R[idces], rates[idces])(R)
            residuals_M[m].append(norm_err(rates, rfit))
            pbar.update(1)

Tsubs = np.geomspace(1000, Tmax, num=100)
residuals_T = {m: [] for m in model_names}
subset_err_T = {m: [] for m in model_names}
with tqdm(total=len(model_names) * len(Tsubs)) as pbar:
    for m in model_names:
        R, sd = firing_rates(
            model=m, q=q, dt=dt, T=Tmax, M=M, sigma_max=sigma_max, return_times=True
        )
        rates = sd.rates("Hz")
        for Tsub in Tsubs:
            rsub = sd.subtime(0, Tsub).rates("Hz")
            rfit = fitted_curve(softplus_ref, R, rsub)(R)
            residuals_T[m].append(norm_err(rates, rfit))
            subset_err_T[m].append(norm_err(rates, rsub))
            pbar.update(1)

with figure("04 Convergence") as f:
    byM, byT = f.subplots(1, 2)

    for i, m in enumerate(model_names):
        byM.loglog(
            Msubs, residuals_M[m], label=model_names[m], linestyle=model_line_styles[m]
        )
        byT.loglog(
            Tsubs / 1e3,
            residuals_T[m],
            label=model_names[m],
            linestyle=model_line_styles[m],
        )

    byM.set_xlabel("Neurons Considered")
    byT.set_xlabel("Simulation Time (s)")

    byM.set_ylabel("Normalized RMSE")
    byM.set_yticks([2e-2, 0.6e-2, 0.2e-2], ["2\%", "0.6\%", "0.2\%"])
    byM.set_ylim(0.0015, 0.027)
    byT.set_yticks([])
    byT.set_yticks([], minor=True)
    byT.set_ylim(byM.get_ylim())

    byM.legend(loc="lower right")


for m in model_names:
    is_better = np.less(residuals_T[m], subset_err_T[m])
    i = np.nonzero(is_better)[0][-1]
    T = Tsubs[i] / 1e3
    err = residuals_T[m][i]
    worst_rel_err = subset_err_T[m][0] / residuals_T[m][0]
    print(f"{m} crossover at {T=:.1e}s: {err=:.2e}, worst", worst_rel_err)


# %%
# Figure 5
# ========
# This figure demonstrates that Refractory SoftPlus can be fitted to
# a variety of different neuron configurations.

T = 1e5
q = 1.0
dt = 0.1
sigma_max = 10.0
N_samples = 100

errses = generalization_errors(
    softplus_ref, T=T, q=q, dt=dt, sigma_max=sigma_max, N_samples=N_samples
)

with figure(
    "05 Parameter Generalization",
    figsize=[5.1, 3.0],
    save_args=dict(bbox_inches="tight"),
) as f:
    ftop, fbot = f.subfigures(2, 1)
    hist = fbot.subplots()

    tf, err = [], []
    x = 3
    tops = ftop.subfigures(1, 6, width_ratios=[1, x, 1, x, 1, x])[1::2]
    for sf in tops:
        ax = sf.subplots(2, 1, gridspec_kw=dict(hspace=0.1))
        tf.append(ax[0])
        err.append(ax[1])

    for i, model in enumerate(model_names):
        R, rates = firing_rates(T=T, q=q, dt=dt, model=model, sigma_max=sigma_max)
        ratehats = fitted_curve(softplus_ref, R, rates)(R)
        base_err = norm_err(rates, ratehats)

        r = R / 1e3
        tf[i].plot(r, rates, f"C{i}o", ms=1)
        tf[i].plot(r, ratehats, "k:")
        tf[i].set_xticks([])
        tf[i].set_title(model_names[model])

        err[i].plot(r, (rates - ratehats) / rates.max(), f"C{i}o", ms=1)
        err[i].set_xlabel("Input Rate $R$ (kHz)")
        tf[i].set_ylabel("FR (Hz)")
        err[i].set_ylabel("Error")
        err[i].set_ylim(-0.0325, 0.0325)
        err[i].set_yticks([-0.03, 0, 0.03], ["-3\%", "0\%", "3\%"])

        bins = np.linspace(0.004, 0.016, 41)
        hist.hist(
            errses[model],
            alpha=0.75,
            label=model_names[model],
            bins=bins,
            facecolor=f"C{i}",
            histtype="stepfilled",
            edgecolor=f"C{i}",
            ls=model_line_styles[model],
        )
        hist.plot(base_err, 8.5, f"C{i}*")
        hist.legend()

    for sf in tops:
        sf.align_ylabels()

    hist.set_ylabel("Count")
    hist.set_xlabel("Normalized Root-Mean-Square Error")
    hist.set_xticklabels([f"{100*x:.1f}\%" for x in hist.get_xticks()])


# %%
# Figure 6
# ========
# Simulated vs. theoretical fixed points in two different LIF networks.
# First is the best-case scenario, demonstrating only a few percent error in
# a high-σ condition, even for fairly large FR. Second is a case with more
# recurrence, where the feedback behavior is worse because N is a bit lower
# relative to the amount of input being expected from it, so firing rates
# are systematically underestimated.

eta = 0.8
M = 10000
dt = 0.1
T = 2e3
model = "iaf_psc_delta"

N_theo = np.arange(30, 91)
N_sim = np.array([N for N in N_theo if int(N * eta) == N * eta])
conditions = [
    # R_bg, q, annealed_average
    (10e3, 3.0, False),
    (0.1e3, 5.0, False),
    (0.1e3, 5.0, True),
]

fp_theo = []
for Rb, q, aa in conditions:
    fp_theo.append([])
    if aa:
        continue
    R, rates = firing_rates(model, q, eta=eta, dt=dt, T=1e5, M=100, sigma_max=10.0)
    tf = fitted_curve(softplus_ref, R, rates)
    lowers, uppers, unstables = [], [], []
    for N in N_theo:
        F, Finv = parametrized_F_Finv(tf.p, Rb, N, q)
        st, us = find_fps(40, F, Finv)
        lowers.append((N, st[0]))
        if len(us) == 1:
            unstables.append((N, us[0]))
        if len(st) == 2:
            uppers.append((N, st[1]))
    fp_theo[-1].append(np.array(lowers))
    fp_theo[-1].append(np.array(uppers))
    fp_theo[-1].append(np.array(unstables))


fp_sim = []
with tqdm(total=13 * len(N_sim) * len(conditions), desc="Sim") as pbar:
    for Rb, q, aa in conditions:
        fp_sim.append([])
        delay = 1.0 + nest.random.uniform_int(10)
        connmodel = AnnealedAverageConnectivity if aa else RandomConnectivity
        for N in N_sim:
            connectivity = connmodel(N, eta, q, delay)
            same_args = dict(
                model=model,
                q=q,
                eta=eta,
                dt=dt,
                T=T,
                M=M,
                R_max=Rb,
                progress_interval=None,
                uniform_input=True,
                connectivity=connectivity,
                seed=1234,
            )
            # Just do a bunch of short simulations so the FR can be averaged. Fewer
            # for the bottom case because it's usually zero.
            fr_top, fr_bot = [], []
            while len(fr_top) < 10:
                same_args["seed"] += 1
                fr = firing_rates(warmup_time=1e3, **same_args)[1].mean()
                # For the N = 55 case, we often fall off the fixed point
                # (sometimes even within the warmup time) so reject any that
                # have total firing rate below 10 Hz.
                if N == 55 and fr < 10:
                    print("N = 55 exception rejecting FR", fr, "Hz.")
                    continue
                fr_top.append(fr)
                pbar.update(1)
                if len(fr_bot) < 3:
                    fr_bot.append(firing_rates(**same_args)[1].mean())
                    pbar.update(1)
            fp_sim[-1].append((fr_top, fr_bot))


def plotkw(Rb, q, aa):
    qlb = f"$q=\qty{{{q}}}{{mV}}$"
    if Rb > 1e3:
        Rlb = f"$R_\\mathrm{{b}} = \\qty{{{round(Rb/1e3)}}}{{kHz}}$"
    else:
        Rlb = f"$R_\\mathrm{{b}} = \\qty{{{Rb/1000}}}{{kHz}}$"
    label = f"{qlb}, {Rlb}"
    if aa:
        label += ", annealed"
    return dict(label=label, ms=5, fillstyle="none")


theo_markers = ["--", "-"]
sim_markers = ["^", "o", "s"]
with figure("06 Sim Fixed Points") as f:
    ax = f.gca()
    for i, (Rb, q, aa) in enumerate(conditions):
        # We didn't compute the theoretical fixed points for the AA version.
        if i in (0, 1):
            # There will always be a lower stable fixed point.
            lowers, uppers, unstables = fp_theo[i]
            ax.plot(lowers[:, 0], lowers[:, 1], "k" + theo_markers[i], lw=1)
        # Only this condition is bistable.
        if i == 1:
            # Add a fake point at the real bifurcation (which isn't at an
            # integer value of N) so the bifurcation looks nice.
            bifurcation = 51.2, 28
            unstables = np.vstack([bifurcation, unstables])
            uppers = np.vstack([bifurcation, uppers])
            ax.plot(unstables[:, 0], unstables[:, 1], "r" + theo_markers[i], lw=1)
            ax.plot(uppers[:, 0], uppers[:, 1], "k" + theo_markers[i], lw=1)
        mark = f"C{i}" + sim_markers[i]
        ax.plot([], [], mark, **plotkw(Rb, q, aa))
        # Average the runs for the top and bottom cases, then combine the
        # cases whenever the bottom case is less than half the top case.
        fpt = np.mean([fps[0] for fps in fp_sim[i]], 1)
        fpb = np.mean([fps[1] for fps in fp_sim[i]], 1)
        distinct = fpb < 0.5 * fpt
        combined = (fpt[~distinct] + fpb[~distinct]) / 2
        Nc, Nd = N_sim[~distinct], N_sim[distinct]
        fpt, fpb = fpt[distinct], fpb[distinct]
        ax.plot(Nd, fpt, mark, ms=5, fillstyle="none")
        ax.plot(Nd, fpb, mark, ms=5, fillstyle="none")
        ax.plot(Nc, combined, mark, ms=5, fillstyle="none")
        ax.set_xlabel("Number of Recurrent Connections")
        ax.set_ylabel("Firing Rate (Hz)")
        ax.legend()


# %%
# Figure 7
# ========
# Finite size effects on the fixed point of the recurrent network from the
# first example of the bifurcation analysis figure. The location of the
# equilibrium doesn't depend on M, but the variability of the firing rate
# around it certainly does.

eta = 0.8
dt = 0.1
T = 2e3
model = "iaf_psc_delta"
Rb = 10e3
q = 3.0
N = 75

Ms = np.geomspace(100, 100000, num=31, dtype=int)

delay = 1.0 + nest.random.uniform_int(10)
run_args = dict(
    model=model,
    q=q,
    eta=eta,
    dt=dt,
    T=T,
    R_max=Rb,
    # Short warmup to avoid a tiny artefact at the start of trate
    warmup_time=10.0,
    warmup_rate=Rb,
    uniform_input=True,
    progress_interval=None,
    return_times=True,
    connectivity=RandomConnectivity(N, eta, q, delay),
)


def oufit(X, h):
    """
    Perform a maximum-likelihood fit of the parameters of an OU
    process to the increments of a time series X with time step h.
    Returns just the process's sigma because the other ones aren't
    interesting at all.
    """
    dX = np.diff(X)
    a, b = np.polyfit(X[:-1], dX, 1)
    eps = dX - a * X[:-1] - b
    return np.std(eps) * np.sqrt(h)


mean, std = [], []
with tqdm(total=10 * sum(Ms), unit="neuron") as pbar:
    for M in Ms:
        mean.append([])
        std.append([])
        for i in range(10):
            _, sd = firing_rates(**run_args, M=M, seed=1234 + i, cache=M > 10000)
            mean[-1].append(sd.rates("Hz").mean())
            std[-1].append(oufit(sd.binned(1) / M / 1e-3, 1.0))
            pbar.update(M)

mean = np.array(mean)
std = np.array(std)

# The model predicted value of `mean` is `theo`
R, rates = firing_rates(model, q, eta=eta, dt=dt, T=1e5, M=100, sigma_max=10.0)
tf = fitted_curve(softplus_ref, R, rates)
F, Finv = parametrized_F_Finv(tf.p, Rb, N, q)
[theo], () = find_fps(40, F, Finv)

# Hang on to an example run with smallish M for the figure.
_, sd = firing_rates(**run_args, M=1000, cache=False)
trate = sd.binned(1) / 1000 / 1e-3

with figure("07 Finite Size Effects", figsize=[4.5, 3.0]) as f:
    axes = f.subplot_mosaic("AA\nBC", height_ratios=[1, 2])
    axes["A"].plot(trate)
    axes["A"].set_xlabel("Time (ms)")
    axes["A"].set_xlim(0, 1e3)
    axes["A"].set_ylabel("Firing Rate (Hz)")

    axes["B"].axhline(theo, color="grey")
    mm, ms = mean.mean(1), mean.std(1)
    axes["B"].semilogx(Ms, mean.mean(1))
    axes["B"].fill_between(Ms, mm - ms, mm + ms, alpha=0.5)
    axes["B"].set_xlabel("Number of Neurons")
    axes["B"].set_ylabel("Firing Rate (Hz)")

    # Plot the fitted OU sigma, and the best-fit square root law.
    sm, ss = std.mean(1), std.std(1)
    k = np.mean(sm * np.sqrt(Ms))
    axes["C"].semilogx(Ms, sm, label="Simulation")
    axes["C"].fill_between(Ms, sm - ss, sm + ss, alpha=0.5)
    axes["C"].plot([], [], "grey", label="Model")
    axes["C"].plot(Ms, k / np.sqrt(Ms), "k:", label="Square Root Law")
    axes["C"].legend()
    axes["C"].set_xlabel("Number of Neurons")
    axes["C"].set_ylabel("F.R. Variation (Hz)")


# %%
# Table 2
# =======
# Compute the practical stability of the fixed point for the N = 55 case with both
# fixed and annealed-average connectivity.

N = 55
Rb = 0.1e3
q = 5.0

eta = 0.8
dt = 0.1
T = 2e3
model = "iaf_psc_delta"

# You can't construct AA connectivity for very large networks, so stop at 20k
# and assume it's 100% after that.
Ms = [1000, 2000, 5000, 10000, 20000, 50000, 100000]
Ms_aa = Ms[:-2]
N_samples = 50


def practical_stability(M, aa, pbar=None):
    delay = 1.0 + nest.random.uniform_int(10)
    cmodel = AnnealedAverageConnectivity if aa else RandomConnectivity
    common_args = dict(
        model=model,
        M=M,
        q=q,
        dt=dt,
        eta=eta,
        T=T,
        R_max=Rb,
        progress_interval=None,
        uniform_input=True,
        connectivity=cmodel(N, eta, q, delay),
        return_times=True,
        cache=M > 10000,
        warmup_time=1e3,
    )
    count = 0
    for i in range(N_samples):
        _, sd = firing_rates(**common_args, seed=1234 + i)
        trate = sd.binned(500.0) / M / 0.5
        # This is arbitrary, but in this particular case, the fixed point is near 80, so
        # a false negative is extremely unlikely. A false positive probably just means
        # that it fell off the FP _right_ at the end, because the variance of the lower
        # FP is very low if it's actually sitting there.
        if trate[-1] > 20:
            count += 1
        if pbar is not None:
            pbar.update(M)
    return count / N_samples


with tqdm(total=N_samples * (sum(Ms) + sum(Ms_aa)), unit="neurons") as pbar:
    stability = [practical_stability(M, False, pbar=pbar) for M in Ms]
    stability_aa = [practical_stability(M, True, pbar=pbar) for M in Ms_aa]


print("Practical stability of fixed connectivity:")
for M, s, sa in zip_longest(Ms, stability, stability_aa, fillvalue=1):
    print(f"${M = :6d}$ & {s:.0%} & {sa:.0%} \\\\".replace("%", "\%"))


# %%
# Figure 8
# ========
# Demonstrating modeling the dynamics appropriately and inappropriately via
# a few simple examples of the model behavior.

eta = 0.8
q = 3.0
dt = 0.1
tau = 1.0
Rb = 10e3

M = 1000
T = 2e3
bin_size_ms = 10.0
warmup_bins = 10
warmup_ms = warmup_bins * bin_size_ms


def sim_sinusoid(model, *, N, amp=1e3, freq=1.0, **kwargs):
    delay = 1.0 + nest.random.uniform_int(10)
    t = np.arange(0, T, bin_size_ms)
    return t, firing_rates(
        model=model,
        q=q,
        M=M,
        T=T + warmup_ms,
        dt=dt,
        connectivity=RandomConnectivity(N, eta, q, delay=delay),
        return_times=True,
        uniform_input=True,
        R_max=Rb,
        osc_amplitude=amp,
        osc_frequency=freq,
        cache=False,
        progress_interval=None,
        **kwargs,
    )[1].binned(bin_size_ms)[warmup_bins:] / (M / 1e3 * bin_size_ms)


def mf_sinusoid(tf, *, N, amp=1e3, freq=1.0):
    last_r, r_pred = 0.0, []
    t_full = np.arange(-3 * tau, T)
    r_inputs = Rb + amp * np.sin(2e-3 * np.pi * freq * (t_full + warmup_ms))
    for i, r_input in enumerate(r_inputs):
        r_star = tf(r_input + N * last_r)
        rdot = (r_star - last_r) / tau
        last_r += rdot  # here dt is 1 ms
        r_pred.append(last_r)
    return t_full[t_full >= 0], np.array(r_pred)[t_full >= 0]


sin_egs = {model: [(50, 5e3), (50, 10e3)] for model in model_names}

eg_results = {}
for model, conds in sin_egs.items():
    R, rates = firing_rates(
        model=model,
        eta=eta,
        q=q,
        dt=dt,
        T=1e5,
        sigma_max=10.0,
    )
    tf = fitted_curve(softplus_ref, R, rates)
    eg_results[model] = []
    for N, amp in conds:
        tt, true = sim_sinusoid(model, N=N, amp=amp)
        tp, pred = mf_sinusoid(tf, N=N, amp=amp)
        eg_results[model].append((tt, true, tp, pred))


with figure("08 Sinusoid Following", figsize=[4, 3]) as f:
    axes = f.subplots(3, 2)
    for i, model in enumerate(model_names):
        color = f"C{i}"
        for j, ax in enumerate(axes[i, :]):
            tt, true, tp, pred = eg_results[model][j]
            ax.plot(tt, true, color)
            ax.plot(tp, pred, "k")
            ax.set_xlim(0, 2e3)
            if i == 2:
                ax.set_xlabel("Time (ms)")
            else:
                ax.set_xticks([])
        ytop = max(axes[i, j].get_ylim()[1] for j in range(2))
        axes[i, 0].set_ylim(0, ytop)
        axes[i, 1].set_ylim(0, ytop)
        axes[i, 1].set_yticks([])
    axes[0, 0].set_title("10 kHz ± 5 kHz")
    axes[0, 1].set_title("10 kHz ± 10 kHz")
    axes[0, 0].set_ylabel("LIF F.R. (Hz)")
    axes[1, 0].set_ylabel("Izh. F.R. (Hz)")
    axes[2, 0].set_ylabel("HH F.R. (Hz)")
    f.align_ylabels()


# %%
# Figure S1
# ========
# Compare the RS79 analytical solution to simulated firing rates for
# a single neuron to demonstrate that it works in the diffusion limit but
# not away from it.

T = 1e5
model = "iaf_psc_delta"
sigma_max = 10.0
t_refs = [0.0, 2.0]

with figure("S1 LIF Analytical Solutions", save_args=dict(bbox_inches="tight")) as f:
    axes = f.subplots(2, 2)
    for t_ref, axr, axe in zip(t_refs, *axes):
        conditions = {
            c: firing_rates(
                **p,
                T=T,
                sigma_max=sigma_max,
                model_params=dict(t_ref=t_ref),
                model=model,
                dt=0.001,
            )
            for c, p in [
                ("Limiting Behavior", dict(q=0.1)),
                ("$q = \\qty{1.0}{mV}$", dict(q=1.0)),
            ]
        }

        R = conditions["Limiting Behavior"][0]
        ratehats = rs79(R, 0.1, 10, 15, t_ref)

        x = np.linspace(0, 100, num=len(R))
        for i, (_, rates) in enumerate(conditions.values()):
            axr.plot(x, rates, "o^s"[i], ms=1)
            axe.plot(x, (rates - ratehats) / rates.max(), "o^s"[i], ms=1)
        axr.plot(x, ratehats, "k:")
        axe.set_xlabel("Mean Presynaptic Rate $r$ (Hz)")
        axr.set_xticks([])

    axes[1, 0].set_yticks([-0.1, -0.05, 0.0], ["-10\%", "-5\%", "0\%"])
    for ax in axes.ravel():
        ax.set_xlim(0, x.max())
    for l, r in axes:
        r.set_yticks([])
        r.set_ylim(*l.get_ylim())
    axes[0, 0].set_title("Non-Refractory Neuron")
    axes[0, 1].set_title("Refractory Neuron")
    axes[0, 0].set_ylabel("Firing Rate (Hz)")
    axes[1, 0].set_ylabel("Normalized Error")
    f.align_ylabels()

    # Create bogus artists for the legend.
    for i, c in enumerate(conditions):
        axr.plot([], [], f"C{i}" + "o^s"[i], label=c)
    axr.plot([], [], "k:", label="Analytical")
    f.legend(ncol=4, loc=(0.175, 0.0))


# %%
# Figure S2
# ========
# Here we simulate theoretical transfer functions for a grid of values of
# R and q, fit a single transfer function to all of them, and plot both the
# transfer function and its error as a 3D surface.

model = "iaf_psc_delta"
Tmax = 1e7
dt = 0.1
qs = np.geomspace(0.1, 10, num=20)
rateses = {
    q: firing_rates(
        q=q, sigma_max=10.0, M=100, model=model, dt=dt, T=Tmax, progress_interval=None
    )
    for q in tqdm(qs)
}
Rses = {q: rates[0] for q, rates in rateses.items()}
rateses = {q: rates[1] for q, rates in rateses.items()}
Ns = np.int64([Rses[q][-1] / 100 for q in qs])
R = Rses[qs[0]] / Ns[0]

Rs_and_qs = np.hstack([(Rses[q], q * np.ones_like(Rses[q])) for q in qs])
rates = np.hstack([rateses[q] for q in qs])
tf = fitted_curve(softplus_ref_q_dep, Rs_and_qs, rates)
ratehats = {q: tf([Rses[q], q]) for q in qs}

surfargs = dict(color="cyan")

# Disgusting way to avoid the weird off-center plot with a ton of whitespace
# that overflows the columns of my paper.
bb = plt.matplotlib.transforms.Bbox([[0.75, 0], [6, 2.5]])
with figure(
    "S2 FR and Error Landscapes", figsize=(6, 2.5), save_args=dict(bbox_inches=bb)
) as f:
    fr, err = f.subplots(1, 2, subplot_kw=dict(projection="3d", facecolor="#00000000"))

    iv = np.meshgrid(np.log10(qs), R)

    fr.plot_surface(*iv, np.array([rateses[q] for q in qs]).T, **surfargs)
    fr.set_zlabel("Firing Rate (Hz)")
    fr.set_xlabel("$q$ (mV)")
    fr.set_xticks([-1, 0, 1], ["0.1", "1", "10"])
    fr.set_ylabel("$r$ (Hz)")

    err.plot_surface(
        *iv,
        100 * np.array([(rateses[q] - ratehats[q]) / ratehats[q].max() for q in qs]).T,
        **surfargs,
    )
    err.set_zlabel("Rate Error (\%)")
    err.set_ylabel("$r$ (Hz)")
    err.set_xlabel("$q$ (mV)")
    err.zaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(decimals=0))
    err.set_xticks([-1, 0, 1], ["0.1", "1", "10"])


# %%
# Figure S3
# =========
# Variance in the firing rate estimate as a function of the time used to
# calculate that estimate, compared to the theoretical value.

dt = 0.1
q = 1.0
T = 1e7
N_per_input = 50

inputs = np.linspace(1e4, 4e4, 4)
R_in, sd = firing_rates(
    "iaf_psc_delta",
    M=len(inputs) * N_per_input,
    dt=dt,
    q=q,
    R_max=np.repeat(inputs, N_per_input),
    T=T,
    uniform_input=True,
    return_times=True,
)

Tsubs = np.geomspace(1000, T, num=21)
stds = [[] for _ in inputs]
for Tsub in Tsubs:
    rates = sd.subtime(0, Tsub).rates("Hz")
    for i in range(len(inputs)):
        this_rates = rates[i * N_per_input : (i + 1) * N_per_input]
        stds[i].append(this_rates.std())

means = [
    np.mean(rates[i * N_per_input : (i + 1) * N_per_input]) for i in range(len(inputs))
]

Tsec = Tsubs / 1e3
with figure("S3 Estimate Variance") as f:
    ax = f.gca()
    for i, R in enumerate(inputs):
        label = f"$R = \\qty{{{R/1e3:.0f}}}{{kHz}}$"
        ax.loglog(Tsec, stds[i], f"C{i}o", ms=4, label=label)
        ax.loglog(Tsec, np.sqrt(means[i] / Tsec), f"C{i}-")
    ax.set_xlabel("Time Used for Estimate (s)")
    ax.set_ylabel("Standard Deviation (Hz)")
    ax.legend()
