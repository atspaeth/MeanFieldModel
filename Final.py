# Final Figures
#
# This script generates all the figures from our new manuscript
# ``Model-agnostic neural mean-field models with the Refractory SoftPlus
# transfer function''
import functools
import itertools
import collections
import numpy as np
from scipy import stats, signal, optimize, special, sparse
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle

import mfsupport
from mfsupport import softplus_ref, rs79, softplus, zerlaut_erf, \
    parametrized_F_Finv, find_fps, firing_rates, softplus_ref_q_dep, \
    BiasedPoisson, BalancedPoisson, RandomConnectivity, \
    BernoulliAllToAllConnectivity, figure
# mfsupport.figdir('~/Dropbox/Apps/Overleaf/MeanField/Figures/')

plt.ion()
plt.rcParams['figure.dpi'] = 300
if 'elsevier' in plt.style.available:
    plt.style.use('elsevier')

def fitted_curve(f, x, y):
    return f(x, *optimize.curve_fit(f, x, y, method='trf')[0])

def norm_err(true, est):
    return np.sqrt(np.mean((true - est)**2)) / true.max()

model_names = {'iaf_psc_delta': 'Leaky Integrate-and-Fire',
               'izhikevich': 'Izhikevich',
               'hh_psc_alpha': 'Hodgkin-Huxley'}
model_line_styles = {'iaf_psc_delta': (1, (1,0.5)),
                     'izhikevich': (3, (6,1)),
                     'hh_psc_alpha': (2.5, (5,1,1,1))}
figure_width, figure_height = plt.rcParams['figure.figsize']


# %%
# Figure 2
# ========
# This figure demonstrates that Refractory SoftPlus can be fitted to
# a variety of different neuron configurations.

T = 1e5
q = 1.0
dt = 0.1
sigma_max = 10.0
N_samples = 100


def sample_params(model):
    match model:
        case 'iaf_psc_delta':
            return dict(t_ref=np.random.exponential(2),
                        C_m=np.random.normal(250, 50))
        case 'izhikevich':
            return dict(a=np.random.uniform(0.02, 0.1),
                        b=np.random.uniform(0.2, 0.25),
                        c=np.random.uniform(-65, -50),
                        d=np.random.uniform(2, 8))
        case 'hh_psc_alpha':
            return dict(C_m=np.random.normal(100, 10),
                        t_ref=np.random.exponential(2),
                        tau_syn_ex=np.random.exponential(1),
                        tau_syn_in=np.random.exponential(1))

# Explicitly cache the actual error values in a single pickle file, because
# we can't cache the simulation runs themselves due to their random params.
try:
    with open('fig2_errors.pickle', 'rb') as f:
        errses = pickle.load(f)
except FileNotFoundError:
    errses = {model: [] for model in model_names}

try:
    for model in model_names:
        to_run = N_samples - len(errses[model])
        if not to_run:
            continue
        for _ in tqdm(range(to_run)):
            R, rates = firing_rates(
                T=T, q=q, dt=dt, model=model, sigma_max=sigma_max,
                cache=False, model_params=sample_params(model),
                progress_interval=None)
            ratehats = fitted_curve(softplus_ref, R, rates)
            errses[model].append(
                norm_err(rates, ratehats))
finally:
    with open('fig2_errors.pickle', 'wb') as f:
        pickle.dump(errses, f)

with figure('02 Parameter Generalization', figsize=[5.1, 3.0],
            save_args=dict(bbox_inches='tight')) as f:
    ftop, fbot = f.subfigures(2, 1)
    hist = fbot.subplots()

    tf, err = [], []
    x = 3
    tops = ftop.subfigures(1, 6, width_ratios=[1,x,1,x,1,x])[1::2]
    for sf in tops:
        ax = sf.subplots(2, 1, gridspec_kw=dict(hspace=0.1))
        tf.append(ax[0])
        err.append(ax[1])

    for i, model in enumerate(model_names):
        R, rates = firing_rates(T=T, q=q, dt=dt, model=model,
                                sigma_max=sigma_max)
        ratehats = fitted_curve(softplus_ref, R, rates)
        base_err = norm_err(rates, ratehats)

        r = R/1e3
        tf[i].plot(r, rates, f'C{i}o', ms=1)
        tf[i].plot(r, ratehats, 'k:')
        tf[i].set_xticks([])
        tf[i].set_title(model_names[model])

        err[i].plot(r, (rates - ratehats) / rates.max(), f'C{i}o', ms=1)
        err[i].set_xlabel('Input Rate $R$ (kHz)')
        tf[i].set_ylabel('FR (Hz)')
        err[i].set_ylabel('Error')
        err[i].set_ylim(-0.0325, 0.0325)
        err[i].set_yticks([-0.03, 0, 0.03], ['-3\%', '0\%', '3\%'])

        bins = np.linspace(0.004, 0.016, 41)
        hist.hist(errses[model], alpha=0.75, label=model_names[model],
                  bins=bins, facecolor=f'C{i}', histtype='stepfilled',
                  edgecolor=f'C{i}', ls=model_line_styles[model])
        hist.plot(base_err, 8.5, f'C{i}*')
        hist.legend()

    for sf in tops:
        sf.align_ylabels()

    hist.set_ylabel('Count')
    hist.set_xlabel('Normalized Root-Mean-Square Error')
    hist.set_xticklabels([f'{100*x:.1f}\%' for x in hist.get_xticks()])

# %%
# Figure 3
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

def subfit_err(model, Msub):
    idces = np.linspace(0, Mmax-1, num=Msub, dtype=int)
    rsub = sds[model].subset(idces).rates('Hz')
    mu = optimize.curve_fit(softplus_ref, R[idces], rsub, method='trf')[0]
    return norm_err(sds[model].rates('Hz'), softplus_ref(R, *mu))

Msubs = np.geomspace(5, Mmax/2, num=100, dtype=int)
residuals_M = {m: [] for m in model_names}
with tqdm(total=len(model_names)*len(Msubs)) as pbar:
    for m in model_names:
        R, sd = firing_rates(model=m, q=q, dt=dt, T=T, M=Mmax,
                             sigma_max=sigma_max, return_times=True)
        rates = sd.rates('Hz')
        for Msub in Msubs:
            idces = np.linspace(0, Mmax-1, num=Msub, dtype=int)
            mu = optimize.curve_fit(softplus_ref, R[idces], rates[idces],
                                    method='trf')[0]
            rfit = softplus_ref(R, *mu)
            residuals_M[m].append(norm_err(rates, rfit))
            pbar.update(1)

Tsubs = np.geomspace(1000, Tmax, num=100)
residuals_T = {m: [] for m in model_names}
with tqdm(total=len(model_names)*len(Tsubs)) as pbar:
    for m in model_names:
        R, sd = firing_rates(model=m, q=q, dt=dt, T=Tmax, M=M,
                             sigma_max=sigma_max, return_times=True)
        for Tsub in Tsubs:
            rsub = sd.subtime(0, Tsub).rates('Hz')
            mu = optimize.curve_fit(softplus_ref, R, rsub, method='trf')[0]
            rfit = softplus_ref(R, *mu)
            residuals_T[m].append(norm_err(sd.rates('Hz'), rfit))
            pbar.update(1)

with figure('03 Convergence') as f:
    byM, byT = f.subplots(1, 2)

    for i,m in enumerate(model_names):
        byM.loglog(Msubs, residuals_M[m], label=model_names[m],
                   linestyle=model_line_styles[i])
        byT.loglog(Tsubs / 1e3, residuals_T[m], label=model_names[m],
                   linestyle=model_line_styles[i])

    byM.set_xlabel('Neurons Considered')
    byT.set_xlabel('Simulation Time (s)')

    byM.set_ylabel('Normalized RMSE')
    byM.set_yticks([2e-2, 0.6e-2, 0.2e-2], ['2\%', '0.6\%', '0.2\%'])
    byM.set_ylim(0.0015, 0.027)
    byT.set_yticks([])
    byT.set_yticks([], minor=True)
    byT.set_ylim(byM.get_ylim())

    byM.legend(loc='lower right')



# %%
# Figure 4
# ========
# Three consistency curves for the LIF neuron, demonstrating that the model
# predicts a saddle-node bifurcation that we're going to be able to observe
# in the next set of simulations...

Ns = [25, 42, 50]
upper_Ns = [50, 59, 100]

dt = 0.1
q = 5.0
Rb = 0.1e3
rs = np.linspace(0, 50, num=1000)

R, rates = firing_rates(model='iaf_psc_delta', q=q, dt=dt, T=1e5, M=100,
                        sigma_max=10.0)
p = optimize.curve_fit(softplus_ref, R, rates, method='trf')[0]

lses = [(0, (3,1)), (3, (6,1)), (0, (1,0)),
        (2.5, (5, 1, 1, 1)), (2, (4, 1, 1, 1, 1, 1))]

with figure('04 Consistency Condition',
            figsize=(2*figure_height, figure_height)) as f:
    axes = f.subplots(1, 2)

    cmax = 0
    def plot_Ns(ax, Ns, r_max):
        r_in = np.linspace(0, r_max, num=1000)
        for i,N in enumerate(Ns):
            F, Finv = parametrized_F_Finv(p, Rb, N, q)
            all_lines.append(ax.plot(r_in, F(r_in),
                                     f'C{cmax+i}', ls=lses[cmax+i],
                                     label=f'$N = {N}$')[0])
            ax.plot(r_in, r_in, 'k:')

            stables, unstables = find_fps(50, F, Finv)

            ax.plot(stables, stables, 'ko', fillstyle='full')
            if len(unstables) > 0:
                ax.plot(unstables, unstables, 'ko', fillstyle='none')

    all_lines = []
    plot_Ns(axes[0], Ns, 50)
    all_lines.pop(-1)
    cmax += 2
    plot_Ns(axes[1], upper_Ns, 100)

    axes[0].set_xticks([0, 25, 50])
    axes[1].set_xticks([0, 50, 100])
    for ax in axes:
        ax.set_aspect('equal')
        ax.set_yticks(ax.get_xticks())
        ax.set_xlabel('Input Firing Rate (Hz)')
        ax.set_ylabel('Output Firing Rate (Hz)')

    all_lines.append(
        ax.plot([], [], 'ko', fillstyle='right', label='Fixed Point')[0])
    plt.figlegend(all_lines, [ln.get_label() for ln in all_lines],
                  loc=(0.1, 0.6), ncol=1, fontsize='small')

    axins = axes[1].inset_axes([0.52, -0.01, 0.4, 0.4])
    plot_Ns(axins, upper_Ns, 150)
    axins.set_xlim(-0.5, 3.5)
    axins.set_ylim(*axins.get_xlim())
    axins.set_xticks([0, 3])
    axins.set_yticks([0, 3])
    axins.tick_params(axis='both', which='both', length=2, pad=1.5,
                      labelsize='x-small', direction='out')
    axes[1].indicate_inset_zoom(axins, edgecolor='k', alpha=1)

    # Add the half-stable fixed points manually...
    for ax, hsfp in zip([axes[0], axins], [21.4, 2]):
        ax.plot(hsfp, hsfp, 'ko', fillstyle='right')


def predicate(N):
    F, Finv = parametrized_F_Finv(p, Rb, N, q)
    try:
        stables, _ = find_fps(50, F, Finv)
    # We'll asssume that failing to find a fixed point means that
    # the system isn't bistable anymore.
    except RuntimeError:
        return False
    return len(stables) == 2


def bifurcate_bistability(lower, upper, rtol=1e-3):

    p_lo, p_hi = predicate(lower), predicate(upper)
    if p_lo == p_hi:
        raise ValueError('Binary search requires predicate to differ!')

    while (upper - lower) > abs(upper + lower) * rtol/2:
        mid = (upper + lower) / 2
        p_mid = predicate(mid)
        if p_mid == p_lo:
            lower = mid
        else:
            upper = mid

    return (upper + lower) / 2


N_star_lower = bifurcate_bistability(Ns[1], Ns[2])
print('Bifurcation to bistability at N =', N_star_lower)
print('Lower bifurcation has 0 Hz stable, saddle node at FR =',
      find_fps(
          80, *parametrized_F_Finv(p, Rb,
                                   int(N_star_lower+1), q))[0][-1])

N_star_upper = bifurcate_bistability(Ns[-1], 100.0)
print('Bifurcation to monostability at N =', N_star_upper)
print('Upper bifurcation has stable FR =',
      find_fps(
          80, *parametrized_F_Finv(p, Rb,
                                   int(N_star_upper), q))[0][-1])


# %%
# Figure 5
# ========
# Simulated vs. theoretical fixed points in two different LIF networks.
# First is the best-case scenario, demonstrating only a few percent error in
# a high-σ condition, even for fairly large FR. Second is a case with more
# recurrence, where the feedback behavior is worse because N is a bit lower
# relative to the amount of input being expected from it, so firing rates
# are systematically underestimated.

M = 10000
dt = 0.1
T = 3e3
model = 'LIF'

def mean_field_fixed_points(N, R_background, q):
    R, rates = firing_rates(model=model, q=q, dt=dt, T=1e5, M=100,
                            sigma_max=10.0)
    p = optimize.curve_fit(softplus_ref, R, rates, method='trf')[0]

    F, Finv = parametrized_F_Finv(p, R_background, N, q)
    stables = find_fps(80, F, Finv)[0]
    if len(stables) == 1:
        return np.array(stables*2)
    else:
        return np.array(stables)

def sim_fixed_points(N, R_background, q, annealed_average=False):
    if annealed_average:
        connectivity = BernoulliAllToAllConnectivity(N/M, q)
    else:
        connectivity = RandomConnectivity(
            N, q, synapse_params=dict(delay=5.0))
    same_args = dict(model=model, q=q, dt=dt, T=T, M=M, R_max=R_background,
                     progress_interval=10.0, uniform_input=True,
                     cache=False, connectivity=connectivity,
                     return_times=True)
    R, sd_top = firing_rates(warmup_time=0.5e3, **same_args)
    R, sd_bot = firing_rates(**same_args)
    return np.array([sd_bot.subtime(1e3, ...).rates('Hz').mean(),
                     sd_top.subtime(1e3, ...).rates('Hz').mean()])

N_theo = np.arange(30, 91)
N_sim = np.linspace(N_theo[0], N_theo[-1], 8).astype(int)
conditions = [
    # R_bg, q, annealed_average
    (0.1e3, 5.0, False),
    (10e3, 3.0, False)]

fp_theo, fp_sim = [], []
with tqdm(total=len(N_sim) * len(conditions)) as pbar:
    for Rb, q, aa in conditions:
        fp_theo.append([mean_field_fixed_points(N, Rb, q)
                        for N in N_theo])
        fp_sim.append([])
        for N in N_sim:
            fp_sim[-1].append(sim_fixed_points(N, Rb, q, aa))
            pbar.update(1)

theo_markers = ['k--', 'k-', None]
sim_markers = ['^', 'o', 's']
with figure('05 Sim Fixed Points', save_exts=[]) as f:
    ax = f.gca()
    for i, (Rb,q,aa) in enumerate(conditions):
        qlb = f'$q=\qty{{{q}}}{{mV}}$'
        if Rb > 1e3:
            Rlb = f'$R_\\mathrm{{b}} = \\qty{{{int(Rb/1e3)}}}{{kHz}}$'
        else:
            Rlb = f'$R_\\mathrm{{b}} = \\qty{{{int(Rb)}}}{{Hz}}$'
        label = f'{qlb}, {Rlb}'
        if aa:
            label += ', annealed'
        ax.plot([], [], f'C{i}' + sim_markers[i],
                label=label, ms=5, fillstyle='none')
        if theo_markers[i]:
            ax.plot(N_theo, fp_theo[i], theo_markers[i], lw=1)
        ax.plot(N_sim, fp_sim[i], f'C{i}' + sim_markers[i], ms=5,
                fillstyle='none')
        ax.set_xlabel('Number of Presynaptic Neurons')
        ax.set_ylabel('Firing Rate (Hz)')
        ax.legend()


# %%
# Figure 6
# ========
# Compare the RS79 analytical solution to simulated firing rates for
# a single neuron to demonstrate that it works in the diffusion limit but
# not away from it.

T = 1e5
model = 'iaf_psc_delta'
sigma_max = 10.0
t_refs = [0.0, 2.0]

with figure('06 LIF Analytical Solutions',
            save_args=dict(bbox_inches='tight')) as f:
    axes = f.subplots(2, 2)
    for t_ref, axr, axe in zip(t_refs, *axes):
        conditions = {c: firing_rates(**p, T=T, sigma_max=sigma_max,
                                      model_params=dict(t_ref=t_ref),
                                      model=model, dt=0.001)
                      for c,p in [
                          ('Limiting Behavior', dict(q=0.1)),
                          ('$q = \\qty{1.0}{mV}$', dict(q=1.0)),
                      ]}

        R = conditions['Limiting Behavior'][0]
        ratehats = rs79(R, 0.1, 10, 15, t_ref)

        x = np.linspace(0, 100, num=len(R))
        for i, (_, rates) in enumerate(conditions.values()):
            axr.plot(x, rates, 'o^s'[i], ms=1)
            axe.plot(x, (rates - ratehats) / rates.max(), 'o^s'[i], ms=1)
        axr.plot(x, ratehats, 'k:')
        axe.set_xlabel('Mean Presynaptic Rate $r$ (Hz)')
        axr.set_xticks([])

    axes[1,0].set_yticks([-0.1, -0.05, 0.0], ['-10\%', '-5\%', '0\%'])
    for ax in axes.ravel():
        ax.set_xlim(0, x.max())
    for l,r in axes:
        r.set_yticks([])
        r.set_ylim(*l.get_ylim())
    axes[0,0].set_title('Non-Refractory Neuron')
    axes[0,1].set_title('Refractory Neuron')
    axes[0,0].set_ylabel('Firing Rate (Hz)')
    axes[1,0].set_ylabel('Normalized Error')
    f.align_ylabels()

    # Create bogus artists for the legend.
    for i,c in enumerate(conditions):
        axr.plot([], [], f'C{i}' + 'o^s'[i], label=c)
    axr.plot([], [], 'k:', label='Analytical')
    f.legend(ncol=4, loc=(0.175, 0.0))


# %%
# Figure 7
# ========
# Here we simulate theoretical transfer functions for a grid of values of
# R and q, fit a single transfer function to all of them, and plot both the
# transfer function and its error as a 3D surface.

model = 'iaf_psc_delta'
Tmax = 1e7
dt = 0.1
qs = np.geomspace(0.1, 10, num=20)
rateses = {q: firing_rates(q=q, sigma_max=10.0, M=100, model=model,
                           dt=dt, T=Tmax, progress_interval=None)
           for q in tqdm(qs)}
Rses = {q: rates[0] for q, rates in rateses.items()}
rateses = {q: rates[1] for q, rates in rateses.items()}
Ns = np.int64([Rses[q][-1] / 100 for q in qs])
R = Rses[qs[0]] / Ns[0]

Rs_and_qs = np.hstack([(Rses[q], q*np.ones_like(Rses[q])) for q in qs])
rates = np.hstack([rateses[q] for q in qs])
mu = optimize.curve_fit(softplus_ref_q_dep, Rs_and_qs, rates, method='trf')[0]
ratehats = {q: softplus_ref_q_dep([Rses[q], q*np.ones_like(Rses[q])], *mu)
            for q in qs}

surfargs = dict(color='cyan')

# Disgusting way to avoid the weird off-center plot with a ton of whitespace
# that overflows the columns of my paper.
bb = plt.matplotlib.transforms.Bbox([[0.75,0], [6,2.5]])
with figure('07 FR and Error Landscapes', figsize=(6,2.5),
            save_args=dict(bbox_inches=bb)) as f:
    fr, err = f.subplots(1, 2, subplot_kw=dict(projection='3d',
                                               facecolor='#00000000'))

    iv = np.meshgrid(np.log10(qs), R)

    fr.plot_surface(*iv, np.array([rateses[q] for q in qs]).T, **surfargs)
    fr.set_zlabel('Firing Rate (Hz)')
    fr.set_xlabel('$q$ (mV)')
    fr.set_xticks([-1, 0, 1], ['0.1', '1', '10'])
    fr.set_ylabel('$r$ (Hz)')

    err.plot_surface(*iv, 100*np.array(
        [(rateses[q] - ratehats[q]) / ratehats[q].max()
         for q in qs]).T, **surfargs)
    err.set_zlabel('Rate Error (\%)')
    err.set_ylabel('$r$ (Hz)')
    err.set_xlabel('$q$ (mV)')
    err.set_zticklabels([f'{z:0.0f}\%' for z in err.get_zticks()])
    err.set_xticks([-1, 0, 1], ['0.1', '1', '10'])
