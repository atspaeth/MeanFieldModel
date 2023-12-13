# Final Figures
#
# This script generates all the figures from our new manuscript
# ``Model-agnostic neural mean-field models with the Refractory SoftPlus
# transfer function''
import matplotlib.pyplot as plt
import nest
import numpy as np
from scipy import signal
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
tfs = dict(Sigmoid=sigmoid, ReLU=relu, SoftPlus=softplus_ref)
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
    ax2.set_ylabel("Error (Hz)")
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
    ax.set_xlabel("Input Firing Rate (Hz)")
    ax.set_ylabel("Output Firing Rate (Hz)")

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
with tqdm(total=len(model_names) * len(Tsubs)) as pbar:
    for m in model_names:
        R, sd = firing_rates(
            model=m, q=q, dt=dt, T=Tmax, M=M, sigma_max=sigma_max, return_times=True
        )
        for Tsub in Tsubs:
            rsub = sd.subtime(0, Tsub).rates("Hz")
            rfit = fitted_curve(softplus_ref, R, rsub)(R)
            residuals_T[m].append(norm_err(sd.rates("Hz"), rfit))
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

R, rates = firing_rates(
    model=model,
    q=q,
    eta=eta,
    dt=dt,
    T=1e5,
    M=100,
    sigma_max=10.0,
    progress_interval=10.0,
)
tf = fitted_curve(softplus_ref, R, rates)


def mean_field_fixed_points(N, R_background, q):
    F, Finv = parametrized_F_Finv(tf.p, R_background, N, q)
    stables = find_fps(80, F, Finv)[0]
    if len(stables) == 1:
        return np.array(stables * 2)
    else:
        return np.array(stables)


N_theo = np.arange(30, 91)
N_sim = np.array([N for N in N_theo if int(N * eta) == N * eta])
conditions = [
    # R_bg, q, annealed_average
    (10e3, 3.0, False),
    (0.1e3, 5.0, False),
    (0.1e3, 5.0, True),
]

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
                _, sd_top = firing_rates(
                    warmup_time=1e3, return_times=True, **same_args
                )
                # Ignore runs where we lose the fixed point, detected by checking if
                # the firing rate starts out above 5 Hz and ends up below 5 Hz.
                rate_Hz = sd_top.binned(500) / M / 0.5
                if rate_Hz[0] > 5.0 > rate_Hz[-1]:
                    print(f"{N = }, discarding run that fell off of FP.")
                    continue
                fr_top.append(sd_top.rates("Hz").mean())
                pbar.update(1)
                if len(fr_bot) < 3:
                    fr_bot.append(firing_rates(**same_args)[1].mean())
                    pbar.update(1)
            fp_sim[-1].append((fr_top, fr_bot))

fp_theo = [
    [mean_field_fixed_points(N, Rb, q) for N in tqdm(N_theo, desc=f"Theo {i+1}")]
    for i, (Rb, q, aa) in enumerate(conditions)
    if not aa
]


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


theo_markers = ["k--", "k-"]
sim_markers = ["^", "o", "s"]
with figure("06 Sim Fixed Points") as f:
    ax = f.gca()
    for i, (Rb, q, aa) in enumerate(conditions):
        if not aa:
            ax.plot(N_theo, fp_theo[i], theo_markers[i], lw=1)
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

mean, std = [], []
with tqdm(total=10 * sum(Ms), unit="neuron") as pbar:
    for M in Ms:
        mean.append([])
        std.append([])
        for i in range(10):
            _, sd = firing_rates(**run_args, M=M, seed=1234 + i, cache=M > 10000)
            mean[-1].append(sd.rates("Hz").mean())
            trate = sd.binned(1) / M / 1e-3
            std[-1].append(trate.std())
            pbar.update(M)

mean = np.array(mean)
std = np.array(std)
theo = mean_field_fixed_points(N, Rb, q).mean()

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

    sm, ss = std.mean(1), std.std(1)
    axes["C"].semilogx(Ms, sm, label="Simulation")
    axes["C"].fill_between(Ms, sm - ss, sm + ss, alpha=0.5)
    axes["C"].plot([], [], "grey", label="Model")
    axes["C"].plot(Ms, 500 / np.sqrt(Ms), "k:", label="Square Root Law")
    axes["C"].legend()
    axes["C"].set_xlabel("Number of Neurons")
    axes["C"].set_ylabel("F.R. Variation (Hz)")


# %%
# Figure 8
# ========
# Demonstrating modeling the dynamics appropriately and inappropriately via
# a few examples of the model behavior, and a graph of diverging error with
# increasing fraction of the input which is recurrent.

eta = 0.8
q = 3.0
dt = 0.1
tau = 1.0
Rb = 10e3

M = 10000
T = 2e3
bin_size_ms = 10.0
warmup_bins = 10
warmup_ms = warmup_bins * bin_size_ms


def sim_sinusoid(model, N, seed=1234):
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
        osc_amplitude=3e3,
        osc_frequency=1.0,
        seed=seed,
        cache=False,
        progress_interval=None,
    )[1].binned(bin_size_ms)[warmup_bins:] / (M / 1e3 * bin_size_ms)


def mf_sinusoid(N, tf):
    last_r, r_pred = 0.0, []
    t_full = np.arange(-3 * tau, T)
    r_inputs = 10e3 + 3e3 * np.sin(2 * np.pi * (t_full + warmup_ms) / 1e3)
    for i, r_input in enumerate(r_inputs):
        r_star = tf(r_input + N * last_r)
        rdot = (r_star - last_r) / tau
        last_r += rdot  # here dt is 1 ms
        r_pred.append(last_r)
    return t_full[t_full >= 0], np.array(r_pred)[t_full >= 0]


sin_egs = [
    ("iaf_psc_delta", 10),
    ("iaf_psc_delta", 100),
    ("izhikevich", 40),
    ("hh_psc_alpha", 60),
]

eg_results = []
for model, N in sin_egs:
    R, rates = firing_rates(
        model=model, eta=eta, q=q, dt=dt, T=1e5, M=100, sigma_max=10.0
    )
    tf = fitted_curve(softplus_ref, R, rates)
    true = sim_sinusoid(model, N)
    pred = mf_sinusoid(N, tf)
    # Awkwardly combine both results tuples into one thing.
    eg_results.append(true + pred)


model = "iaf_psc_delta"
n_seeds = 10
Ns = np.geomspace(3, 300, num=21, dtype=int)

R, rates = firing_rates(model=model, eta=eta, q=q, dt=dt, T=1e5, M=100, sigma_max=10.0)
tf = fitted_curve(softplus_ref, R, rates)

sweep_results = []
with tqdm(total=len(Ns) * n_seeds) as pbar:
    for N in Ns:
        tp, pred = mf_sinusoid(N, tf)
        pred = signal.decimate(pred, 10)
        sweep_results.append([])
        for i in range(n_seeds):
            t, true = sim_sinusoid(model, N, seed=1234 + i)
            sweep_results[-1].append(norm_err(true, pred))
            pbar.update(1)
sweep_results = np.array(sweep_results)


input_fracs = []
for N in Ns:
    F, Finv = parametrized_F_Finv(tf.p, Rb, N)
    (r,), () = find_fps(80, F, Finv)
    input_fracs.append(r * N / (r * N + Rb))


with figure("08 Sinusoid Following", figsize=[5.0, 3.0]) as f:
    axes = f.subplot_mosaic("AE\nBE\nCE\nDE", width_ratios=[1, 2])
    A, B, C, D, E = [axes[c] for c in "ABCDE"]

    for ax, (tt, true, tp, pred) in zip([A, B, C, D], eg_results):
        ax.plot(tt / 1e3, true)
        ax.plot(tp / 1e3, pred, "k-")
        ax.set_ylabel("FR (Hz)")
        if ax is D:
            ax.set_xlabel("Time (s)")
        else:
            ax.set_xticks([])
    f.align_ylabels([A, B, C, D])

    emean = sweep_results.mean(1)
    estd = sweep_results.std(1)
    E.plot(input_fracs, emean)
    E.fill_between(input_fracs, emean - estd, emean + estd, alpha=0.5)
    E.set_xlabel("Input From Recurrent Neurons (\%)")
    E.set_ylabel("Error (\%)")
    percents = plt.matplotlib.ticker.PercentFormatter(decimals=0, xmax=1)
    E.xaxis.set_major_formatter(percents)
    E.yaxis.set_major_formatter(percents)


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
