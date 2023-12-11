import os
from contextlib import contextmanager

import braingeneers.analysis as ba
import matplotlib.pyplot as plt
import nest
import numpy as np
from braingeneers.utils.memoize_s3 import memoize
from scipy import optimize, special
from tqdm import tqdm

nest.set_verbosity("M_WARNING")


def _softplus(arg):
    """
    Just the freaky nonlinear part of softplus, implementing
    a large-argument correction.
    """
    ret = np.array(arg, float)
    if len(ret.shape) > 0:
        nonlinear = ret < 20
        ret[nonlinear] = np.log1p(np.exp(ret[nonlinear]))
        return ret
    else:
        return _softplus([arg])[0]


def _softplus_inv(ret):
    "Inverse of _softplus()"
    arg = np.array(ret, float)
    if len(arg.shape) > 0:
        nonlinear = (0 < arg) & (arg < 20)
        arg[nonlinear] = np.log(np.expm1(arg[nonlinear]))
        arg[arg <= 0] = np.nan
        return arg
    else:
        return _softplus_inv([ret])[0]


def softplus(Rs, x0, b, c):
    "The SoftPlus transfer function."
    return c / b * _softplus(b * (np.sqrt(Rs) - x0))


def softplus_inv(rs, x0, b, c):
    "Inverse of SoftPlus transfer function."
    return (x0 + _softplus_inv(b * rs / c) / b) ** 2


def _refractory(r, t_ref):
    "Correct a rate r in Hz to add a refractory period t in ms."
    return 1 / (1 / r + t_ref * 1e-3)


def _refractory_inv(r, t_ref):
    "Inverse of _refractory."
    return 1 / (1 / r - t_ref * 1e-3)


def softplus_ref(Rs, x0, b, c, t_ref):
    "Empirical functional form for firing rate as a function of noise."
    return _refractory(softplus(Rs, x0, b, c), t_ref)


def softplus_ref_q_dep(Rs_and_qs, x0, b, c, d, t_ref):
    "Refractory Softplus with q dependence incorporated in beta."
    Rs, qs = Rs_and_qs
    fr = c / b * _softplus(b * (1 + d * qs) * (qs * np.sqrt(Rs) - x0))
    return _refractory(fr, t_ref)


def softplus_ref_q_dep_inv(rs, q, x0, b, c, d, t_ref):
    "No single inverse exists, but use this for a given q."
    sp_arg = _softplus_inv(b / c * _refractory_inv(rs, t_ref))
    R_term = sp_arg / b / (1 + d * q) + x0
    return (R_term / q) ** 2


def softplus_ref_inv(rs, x0, b, c, t_ref):
    "Inverse of refractory SoftPlus function."
    return softplus_inv(_refractory_inv(rs, t_ref), x0, b, c)


def relu(R, a, b, R0):
    return a / b**2 * np.maximum(0, b**2 * (np.sqrt(R) - R0))


def sigmoid(R, a, b, R0):
    return a / b**2 * special.expit(b**2 * (np.sqrt(R / 1e3) - R0))


def parametrized_F_Finv(μ_softplus, R_background, N, q=None):
    """
    Return the transfer function for a recurrently connected network whose
    SoftPlus or Refractory SoftPlus transfer function is parameterized by the
    provided μ_softplus, with given noise level and connectivity parameters.
    """
    match len(μ_softplus):
        case 3:
            tf, tfinv = softplus, softplus_inv
        case 4:
            tf, tfinv = softplus_ref, softplus_ref_inv
        case 5:
            if q is None:
                raise ValueError("Must provide q for q-dependent transfer function")

            def tf(R, *a):
                return softplus_ref_q_dep((R, q * np.ones_like(R)), *a)

            def tfinv(R, *a):
                return softplus_ref_q_dep_inv(R, q, *a)

        case _:
            raise ValueError("Invalid number of parameters for transfer function.")

    def F(r):
        return tf(R_background + N * r, *μ_softplus)

    def Finv(r):
        return (tfinv(r, *μ_softplus) - R_background) / N

    return F, Finv


@np.vectorize
def _rs79_isi(sigma, theta, S):
    if sigma == 0.0:
        return np.inf
    y = S / sigma * np.sqrt(2 / theta)
    term2 = np.pi * special.erfi(y / np.sqrt(2))
    n = np.arange(25) + 1
    term1s = y ** (2 * n) / n / special.factorial2(2 * n - 1)
    return theta * (np.sum(term1s) + term2) / 2


def rs79(R, q, theta, S, t_ref):
    "Mean firing rate from Ricciardi & Sacerdote 1979."
    sigma = q * np.sqrt(R / 1e3)
    return _refractory(1e3 / _rs79_isi(sigma, theta, S), t_ref)


def zerlaut_erf(R, b, c, k, V0):
    "Transfer function curve fit by Zerlaut et al. 2018, part-manual here."
    Vx = V0 + b * np.sqrt(R) - c * R
    return k * special.erfc(Vx / np.sqrt(2 * R))


def find_fps(r_top, F, Finv, atol=0.1):
    """
    Find fixed points of the SoftPlus consistency equation, starting with
    one guess at zero and one at r_top. The system undergoes two
    complementary saddle-node bifurcations, so there is a parameter region
    with two stable and one unstable fixed point surrounded by a region with
    only one stable fixed point.

    Returns stable_fps and unstable_fps as lists, either length 2 and 1 or
    length 1 and 0 depending on the location in parameter space.
    """
    # First look for two stable FPs since we know at least one exists.
    stable_fps = [
        optimize.fixed_point(F, x, method="iteration", maxiter=5000) for x in [0, r_top]
    ]

    # Then if they are far enough apart to be considered distinct,
    # use iteration on the inverse dynamics to find the unstable FP.
    if stable_fps[-1] > atol + stable_fps[0]:
        try:
            unstable_fp = optimize.fixed_point(Finv, r_top / 2, method="iteration")
            return stable_fps, [unstable_fp]
        except RuntimeError:
            pass
    else:
        stable_fps = [stable_fps[0]]
    return stable_fps, []


def reset_nest(dt, seed):
    nest.ResetKernel()
    if n_threads := os.environ.get("NEST_NUM_THREADS"):
        nest.local_num_threads = int(n_threads)
    nest.resolution = dt
    nest.rng_seed = seed


def firing_rates(
    model,
    q,
    M=500,
    sigma_max=None,
    R_max=None,
    cache=True,
    return_times=False,
    uniform_input=False,
    seed=42,
    model_params={},
    **kwargs,
):
    if R_max is None and sigma_max is not None:
        R_max = 1e3 * (sigma_max / q) ** 2
    elif (R_max is None) == (sigma_max is None):
        raise ValueError("Either R_max or sigma_max must be given!")
    R = R_max if uniform_input else np.linspace(0, R_max, num=M)

    sim = sim_neurons_nest_eta if cache else sim_neurons_nest_eta.func
    sd = sim(model=model, q=q, R=R, M=M, seed=seed, model_params=model_params, **kwargs)
    return R, (sd if return_times else sd.rates("Hz"))


def sim_progress(total_time, interval):
    """
    If there is a progress interval, use a progress bar. If it is zero, the
    chunk size for simulations should be 100ms.
    """
    pbar = tqdm(
        total=total_time, unit="sim sec", unit_scale=1e-3, disable=interval is None
    )
    pbar.interval = 1e2 if interval is None or interval <= 0 else interval
    return pbar


def sim_step_lengths(pbar, total_time, dt):
    """
    Generate step lengths and update a progress bar for a simulation of
    given total length and dt.
    """
    for _ in range(int(total_time / pbar.interval)):
        yield pbar.interval
        pbar.update(pbar.interval)
    residue = total_time % pbar.interval
    if residue > dt:
        yield residue
        pbar.update(residue)


NoConnectivity = []


@memoize(ignore=["progress_interval"])
def sim_neurons_nest_eta(
    model,
    q,
    R,
    dt,
    T,
    M=None,
    eta=0.8,
    model_params=None,
    warmup_time=0.0,
    warmup_rate=None,
    warmup_steps=10,
    osc_amplitude=0.0,
    osc_frequency=0.0,
    connectivity=NoConnectivity,
    seed=42,
    recordables=None,
    progress_interval=1e3,
):
    """
    Simulate M Izhikevich neurons using NEST. They are receiving Poisson
    inputs with connection strength q and rate R, and optionally connected
    to each other by calling the connect() method of the given connectivity
    object(s) on the neurons after initialization. The amplitude and
    frequency (both in Hz) of fluctuations in the Poisson input can also
    be specified.
    """
    R = np.atleast_1d(R)
    if M is None:
        M = len(R)
    if len(R) not in (1, M):
        raise ValueError("R must be a scalar or a vector of length M.")

    reset_nest(dt=dt, seed=seed)

    # Create the neurons and attach them to a spike recorder.
    neurons = nest.Create(model, n=M, params=model_params)
    rec = nest.Create("spike_recorder")
    nest.Connect(neurons, rec)

    # Create separate excitatory and inhibitory noise, sharing the rate
    # according to eta, with weights chosen to maintain EI balance.
    noise_e, noise_i = [
        nest.Create(
            "sinusoidal_poisson_generator",
            len(R),
            params=dict(
                rate=R * frac, frequency=osc_frequency, amplitude=osc_amplitude * frac
            ),
        )
        for frac in [eta, 1 - eta]
    ]
    conn = "all_to_all" if len(R) == 1 else "one_to_one"
    for noise, q in zip([noise_e, noise_i], split_q(eta, q)):
        w = psp_corrected_weight(neurons[0], q, model)
        nest.Connect(noise, neurons, conn, dict(weight=w))

    for c in connectivity:
        c.connect(neurons, model)

    if recordables:
        params = dict(record_from=recordables, interval=dt)
        nest.Connect(nest.Create("multimeter", params=params), neurons)

    with sim_progress(T + warmup_time, progress_interval) as pbar:
        # During warmup time, ramp the rate of the excitatory noise from
        # the warmup value down to the base value, while keeping the
        # inhibitory rate the same.
        if warmup_time > 0:
            base_rate = noise_e.rate
            if warmup_rate is None:
                warmup_rate = 5 * base_rate
            for i in range(warmup_steps):
                # Ramp the rate downwards from warmup_rate to base_rate.
                # The last warmup step must be at the base rate to avoid
                # artefacts in the actual returned data.
                noise_e.rate = np.interp(
                    i, [0, warmup_steps - 1], [warmup_rate, base_rate]
                )
                nest.Simulate(warmup_time / warmup_steps)
                pbar.update(warmup_time / warmup_steps)
            noise_e.rate = base_rate

        with nest.RunManager():
            for step in sim_step_lengths(pbar, T, dt):
                nest.Run(step)

    # Create SpikeData and trim off the warmup time.
    return ba.SpikeData.from_nest(
        rec, neurons, length=T + warmup_time, metadata=rec.events if recordables else {}
    ).subtime(warmup_time, ...)


def voltage_slew_to_current(neuron, slew):
    """
    Take a neuron and multiply a voltage slew by the membrane capacitance in
    order to turn it into the current that would have produced that slew.
    """
    return slew * nest.GetStatus(neuron[0])[0].get("C_m", 1.0)


def split_q(eta, q):
    """
    Calculate the values of q_e and q_i corresponding to a desired effective
    synaptic weight q, which depends on the excitatory fraction eta.

    For example, for eta = 0.5, q_e = -q_i = q, and for eta = 0.8 (so the
    excitatory-inhibitory ratio is 4:1), q_e = q/2 and q_i = -2q.
    """
    sqrt_γ = np.sqrt(1 / eta - 1)
    return q * sqrt_γ, -q / sqrt_γ


def psp_corrected_weight(neuron, q, model_name=None):
    """
    Take a neuron and a desired synaptic weight for a delta PSP and return
    the synaptic weight which should be used instead so that this neuron
    will receive an equivalent voltage injection from its PSCs.
    """
    # Get the base model name, special-casing the builtin Izhikevich because
    # it doesn't specify its synapse type.
    model_name = neuron[0].model if model_name is None else model_name
    if model_name == "UnknownNode":
        raise ValueError("Must provide model name for custom neuron types.")
    elif model_name == "izhikevich":
        model = ["izhikevich", "psc", "delta"]
    else:
        model = model_name.split("_")

    # For PSC neurons, it doesn't matter whether they're HH or I&F, but the
    # shape of the PSC does matter. Can assume that all such neurons have
    # membrane capacitance C_m as well as synaptic time constants, except
    # the ones with delta synapses, which inject voltage like Izhikevich.
    postfixes = ["ex", "exc"] if q > 0 else ["in", "inh"]
    params = nest.GetStatus(neuron)[0]
    tau = np.array(
        [params.get("tau_syn_" + postfix, -np.inf) for postfix in postfixes]
    ).max()
    match model:
        case [_, "psc", "delta"]:
            return q
        case [_, "psc", "exp"]:
            return q / tau * params.get("C_m", 1.0)
        case [_, "psc", "alpha"]:
            return q / tau / np.e * params.get("C_m", 1.0)
    raise NotImplementedError(f"Model {model_name} not supported.")


class Connectivity:
    def connect(self, neurons, model_name=None):
        pass

    def __iter__(self):
        yield self


class RandomConnectivity(Connectivity):
    def __init__(self, N, eta, q, delay=5.0):
        self.eta = eta
        Nexc = round(N * self.eta)
        Ninh = N - Nexc
        self.Ns = Nexc, Ninh
        self.qs = split_q(eta, q)
        self.delay = delay

    def connect(self, neurons, model_name):
        Mexc = round(len(neurons) * self.eta)
        pops = neurons[:Mexc], neurons[Mexc:]
        for pop, N, q in zip(pops, self.Ns, self.qs):
            weight = psp_corrected_weight(neurons[0], q, model_name)
            nest.Connect(
                pop,
                neurons,
                dict(rule="fixed_indegree", indegree=N),
                dict(synapse_model="static_synapse", weight=weight, delay=self.delay),
            )


class BernoulliConnectivity(Connectivity):
    def __init__(self, N, eta, q, delay=5.0):
        self.N = N
        self.eta = eta
        self.q = q
        self.delay = delay

    def connect(self, neurons, model_name=None):
        M = len(neurons)
        Mexc = round(M * self.eta)
        pops = neurons[:Mexc], neurons[Mexc:]
        for pop, q in zip(pops, split_q(self.eta, self.q)):
            w = psp_corrected_weight(neurons[0], q, model_name)
            nest.Connect(
                pop,
                neurons,
                dict(rule="pairwise_bernoulli", p=self.N / M),
                dict(synapse_model="static_synapse", weight=w, delay=self.delay),
            )


class AnnealedAverageConnectivity(Connectivity):
    def __init__(self, N, eta, q, delay=5.0):
        self.N = N
        self.eta = eta
        self.q = q
        self.delay = delay

    def connect(self, neurons, model_name=None):
        M = len(neurons)
        Mexc = round(M * self.eta)
        pops = neurons[:Mexc], neurons[Mexc:]
        for pop, q in zip(pops, split_q(self.eta, self.q)):
            w = psp_corrected_weight(neurons[0], q, model_name)
            nest.Connect(
                pop,
                neurons,
                "all_to_all",
                dict(
                    synapse_model="bernoulli_synapse",
                    weight=w,
                    p_transmit=self.N / M,
                    delay=self.delay,
                ),
            )


@contextmanager
def figure(name, save_args={}, save_exts=["png"], **kwargs):
    "Create a named figure and save it when done."
    f = plt.figure(name, **kwargs)
    try:
        f.clf()
    except Exception:
        plt.close()
        f = plt.figure(name, **kwargs)

    yield f

    fname = name.lower().strip().replace(" ", "-")
    for ext in save_exts:
        if ext[0] != ".":
            ext = "." + ext
        path = os.path.join("figures", fname + ext)
        f.savefig(path, **save_args)


def fitted_curve(f, x, y):
    ret = lambda xstar: f(xstar, *p)
    ret.p = p = optimize.curve_fit(f, x, y, method="trf")[0]
    return ret


def norm_err(true, est):
    return np.sqrt(np.mean((true - est) ** 2)) / true.max()


@memoize(ignore=["progress"])
def generalization_errors(
    transfer_function, *, T, q, dt, sigma_max, N_samples, progress=True
):
    """
    Generate N_samples random parameter sets for each of the three models,
    using a random seed to make sure the same parameters will always be
    requested so the memoization is useful. The function as a whole is *also*
    memoized because it would take so long to load all of these runs.
    """
    rng = np.random.default_rng(42)
    sample_params = dict(
        iaf_psc_delta=lambda: dict(t_ref=rng.exponential(2), C_m=rng.normal(250, 50)),
        izhikevich=lambda: dict(
            a=rng.uniform(0.02, 0.1),
            b=rng.uniform(0.2, 0.25),
            c=rng.uniform(-65, -50),
            d=rng.uniform(2, 8),
        ),
        hh_psc_alpha=lambda: dict(
            C_m=rng.normal(100, 10),
            t_ref=rng.exponential(2),
            tau_syn_ex=rng.exponential(1),
            tau_syn_in=rng.exponential(1),
        ),
    )

    # Actually generate all the fit results.
    errses = {}
    with tqdm(sample_params, disable=not progress, total=3 * N_samples) as pbar:
        for model in sample_params:
            errses[model] = []
            for _ in range(N_samples):
                R, rates = firing_rates(
                    T=T,
                    q=q,
                    dt=dt,
                    model=model,
                    sigma_max=sigma_max,
                    model_params=sample_params[model](),
                    progress_interval=None,
                )
                ratehats = fitted_curve(transfer_function, R, rates)(R)
                errses[model].append(norm_err(rates, ratehats))
                pbar.update()
    return errses
