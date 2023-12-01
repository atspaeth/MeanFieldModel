import functools
import os
import queue
from contextlib import contextmanager

import braingeneers.analysis as ba
import matplotlib.pyplot as plt
import numpy as np
from braingeneers.iot.messaging import MessageBroker
from braingeneers.utils.memoize_s3 import memoize
from scipy import optimize, special
from tqdm import tqdm


def lazy_package(full_name):
    if callable(full_name):
        return lazy_package(full_name.__name__)(full_name)

    def wrapper(func):
        # If the module is already loaded, don't stub it back out!
        if func.__name__ in globals():
            return globals()[func.__name__]

        # If the module isn't loaded, just put this stub in place for now.
        class stub:
            def __getattr__(self, attr):
                from importlib import import_module

                globals()[func.__name__] = ret = import_module(full_name)
                try:
                    func()
                except Exception as e:
                    globals()[func.__name__] = self
                    raise e
                return getattr(ret, attr)

        return stub()

    return wrapper


@lazy_package
def nest():
    nest.set_verbosity("M_WARNING")
    from pynestml.frontend.pynestml_frontend import generate_nest_target

    generate_nest_target("models/", "/tmp/nestml-mfsupport/", module_name="mfmodule")
    nest.Install("mfmodule")


@lazy_package("bindsnet.network")
def bn():
    pass


@lazy_package
def torch():
    pass


@lazy_package("brian2")
def br():
    pass


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


def relu_ref(R, a, b, R0, t_ref):
    ret = a / b**2 * np.maximum(0, b**2 * (R - R0))
    return _refractory(ret, t_ref)


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
    backend="default",
    model_params={},
    **kwargs,
):
    if R_max is None and sigma_max is not None:
        R_max = 1e3 * (sigma_max / q) ** 2
    elif (R_max is None) == (sigma_max is None):
        raise ValueError("Either R_max or sigma_max must be given!")
    R = R_max if uniform_input else np.linspace(0, R_max, num=M)

    if hasattr(model, "model"):
        model = model(backend)
        model_params = model_params or model.params
        model = model.model
    sim = SIM_BACKENDS[backend] if cache else SIM_BACKENDS[backend].func
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


@memoize(ignore=["progress_interval"])
def sim_neurons_nest(
    model,
    q,
    R,
    dt,
    T,
    M=None,
    I_ext=None,
    model_params=None,
    warmup_time=0.0,
    warmup_rate=None,
    warmup_steps=10,
    connectivity=None,
    seed=42,
    recordables=None,
    progress_interval=1e3,
):
    """
    Simulate M Izhikevich neurons using NEST. They are receiving Poisson
    inputs with connection strength q and rate R, and optionally connected
    to each other by calling the given connectivity object's connect()
    method on the neurons after initialization.
    """
    R = np.atleast_1d(R)
    if M is None:
        M = len(R)

    reset_nest(dt=dt, seed=seed)

    neurons = nest.Create(model, n=M, params=model_params)
    if I_ext is not None:
        neurons.I_e = voltage_slew_to_current(neurons, I_ext)

    noise = nest.Create("poisson_generator", n=len(R), params=dict(rate=R / 2))

    if len(R) == 1:
        conn = "all_to_all"
    elif len(R) == M:
        conn = "one_to_one"
    else:
        raise ValueError("R must be a scalar or a vector of length M.")

    nest.Connect(
        noise, neurons, conn, dict(weight=psp_corrected_weight(neurons[0], q, model))
    )
    nest.Connect(
        noise, neurons, conn, dict(weight=psp_corrected_weight(neurons[0], -q, model))
    )

    if connectivity is not None:
        connectivity.connect_nest(neurons, model)

    rec = nest.Create("spike_recorder")
    nest.Connect(neurons, rec)

    if recordables:
        rec = nest.Create(
            "multimeter", params=dict(record_from=recordables, interval=dt)
        )
        nest.Connect(rec, neurons)

    with sim_progress(T + warmup_time, progress_interval) as pbar:
        if warmup_time > 0:
            base_rate = noise.rate
            if warmup_rate is None:
                warmup_rate = 10 * base_rate
            for i in range(warmup_steps):
                noise.rate = np.interp(i, [0, warmup_steps], [warmup_rate, base_rate])
                nest.Simulate(warmup_time / warmup_steps)
                pbar.update(warmup_time / warmup_steps)
            noise.rate = base_rate

        with nest.RunManager():
            for step in sim_step_lengths(pbar, T, dt):
                nest.Run(step)

    # Create SpikeData and trim off the warmup time.
    return ba.SpikeData.from_nest(
        rec, neurons, length=T + warmup_time, metadata=rec.events if recordables else {}
    ).subtime(warmup_time, ...)


def run_net_with_rates(net, R, T, Npre):
    """
    Given a BindsNET network whose input layer is called 'source' and a set
    of input rates R, generate balanced Poisson inputs and run the network.
    """
    # We have to divide R by Npre because it's split into the Npre balanced
    # parts, but also change units from Hz to spikes per time step.
    steps = int(T / net.dt)
    R = R.repeat_interleave(Npre) / Npre
    rates = torch.vstack([R * net.dt / 1e3] * steps)
    input_data = torch.poisson(rates).byte()
    net.run(inputs={"source": input_data}, time=T)


@memoize(ignore=["device", "progress_interval"])
def sim_neurons_bindsnet(
    model,
    q,
    R,
    dt,
    T,
    M=None,
    seed=42,
    model_params={},
    connectivity=None,
    warmup_time=0.0,
    warmup_steps=10,
    progress_interval=None,
    device="cuda",
):
    """
    Simulate M neurons of the given model using BindsNET. They receive
    balanced Poisson inputs with connection strength q and rate R.
    """
    with torch.device(device), torch.no_grad():
        torch.random.manual_seed(seed)
        R = torch.atleast_1d(torch.as_tensor(R))
        if M is None:
            M = len(R)
        elif len(R) == 1:
            R = R.repeat(M)
        elif len(R) != M:
            raise ValueError("R must be scalar or length M.")

        # Build the base network and its layers. Split each conceptual
        # Poisson input neuron into Npre separate neurons so weights can be
        # well-defined and no neuron has to fire more than once per step.
        Npre = 42
        net = bn.Network(dt=dt)
        net.to(device)
        source = bn.nodes.Input(n=Npre * M)
        source.to(device)
        neurons = model(n=M, **model_params)
        neurons.to(device)
        net.add_layer(source, name="source")
        net.add_layer(neurons, name="neurons")
        if connectivity is not None:
            connectivity.connect_bindsnet(net)

        # Connect the input to the neurons Npre-to-one with weight ±q,
        # alternating so the odd indices are negative.
        w = torch.sparse_csc_tensor(
            ccol_indices=Npre * torch.arange(M + 1),
            row_indices=torch.arange(M * Npre),
            values=q * (-1) ** torch.arange(M * Npre),
            size=(M * Npre, M),
        )
        net.add_connection(
            source="source",
            target="neurons",
            connection=bn.topology.Connection(source=source, target=neurons, w=w),
        )

        with sim_progress(T + warmup_time, progress_interval) as pbar:
            # Do the warmup simulation without recording.
            if warmup_time > 0:
                warmup_T = warmup_time / warmup_steps
                for i in range(warmup_steps):
                    warmup_ramp = np.interp(i, [0, warmup_steps], [10, 1])
                    run_net_with_rates(net, warmup_ramp * R, warmup_T, Npre)
                    pbar.update(warmup_T)

            # Add recording before finishing the simulation.
            monitor = bn.monitors.Monitor(
                obj=neurons, state_vars=["s"], time=int(T / net.dt)
            )
            net.add_monitor(monitor=monitor, name="neurons")

            # Run the simulation broken up into chunks.
            for step in sim_step_lengths(pbar, T, dt):
                run_net_with_rates(net, R, step, Npre)

        # Grab the monitor's spike matrix and turn it into SpikeData.
        times, _, idces = torch.nonzero(monitor.get("s"), as_tuple=True)
        sd = ba.SpikeData(idces, times * dt, length=float(T), N=M)

    # BindsNET simulations need to be cleared from GPU memory to allow other
    # simulations to follow them. This is weird because references to the
    # tensors on GPU are still in Python memory, so GC them first.
    import gc

    gc.collect()
    torch.cuda.empty_cache()
    return sd


@memoize(ignore=["progress_interval"])
def sim_neurons_brian2(
    model,
    q,
    R,
    dt,
    T,
    M=None,
    connectivity=None,
    warmup_time=0.0,
    warmup_steps=10,
    model_params={},
    progress_interval=None,
    seed=42,
):
    """
    Simulate M neurons using Brian2. They receive balanced Poisson inputs
    with connection strength q and rate R.

    Since Brian2 has no built-in models, a model must be specified as an
    entire dictionary where 'model' contains a multi-line string with its
    dynamic equations, 'threshold' contains the spike threshold condition as
    a boolean expression, etc. A model parameter dictionary is accepted for
    compatibility with other solvers but is ignored.
    """
    R = np.atleast_1d(R)
    if M is None:
        M = len(R)
    elif len(R) not in (1, M):
        raise ValueError("R must be scalar or length M.")

    br.defaultclock.dt = dt * br.ms
    neurons = br.NeuronGroup(M, **model)

    # All constructed objects must be explicitly named and in scope so Brian
    # can extract them for the run() call.
    if connectivity is not None:
        syn_recurrent = connectivity.connect_brian2(neurons)  # noqa: F841

    Npre = 150
    if len(R) == 1:
        input_pos = br.PoissonInput(neurons, "v", Npre // 2, R[0] / Npre * br.Hz, "q")
        input_neg = br.PoissonInput(neurons, "v", Npre // 2, R[0] / Npre * br.Hz, "-q")
    else:
        # For each postsynaptic neuron, create Npre separate presynaptic
        # Poisson inputs. The odd ones have negative weights, and the split
        # must be significantly greater than 2 because it's impossible for
        # a single neuron to spike multiple times per step.
        source = br.PoissonGroup(Npre * M, R.repeat(Npre) / Npre * br.Hz)
        syn = br.Synapses(source, neurons, on_pre="v += (-1)**i * q")
        syn.connect(j=f"int(i/{Npre})")

    with sim_progress(T + warmup_time, progress_interval) as pbar:
        # Create the namespace for all simulations. Note that this R is only
        # relevant for the case where there is only one rate, and is ignored
        # in the case of a range of values.
        namespace = dict(q=q * br.mV)

        # Run the warmup simulation.
        if warmup_time > 0:
            if len(R) != 1:
                raise ValueError("Warmup not supported for multiple rates.")
            step_length = warmup_time / warmup_steps
            for i in range(warmup_steps, 0, -1):
                input_pos.R = input_neg.R = i * R[0] / 2 * br.Hz
                br.run(step_length * br.ms, namespace=namespace)
                pbar.update(step_length)

        # Add a spike monitor and run the proper simulation.
        monitor = br.SpikeMonitor(neurons)
        for step in sim_step_lengths(pbar, T, dt):
            br.run(step * br.ms, namespace=namespace)

    # Translate the spike monitor into a SpikeData object.
    return ba.SpikeData(monitor.i, monitor.t / br.ms - warmup_time, length=T, N=M)


def backend_unimplemented(*args, **kwargs):
    raise NotImplementedError("Backend not yet supported.")


SIM_BACKENDS = {
    "default": sim_neurons_nest,
    "nest": sim_neurons_nest,
    "bindsnet": sim_neurons_bindsnet,
    "bindsnet:cpu": functools.partial(sim_neurons_bindsnet, device="cpu"),
    "bindsnet:gpu": functools.partial(sim_neurons_bindsnet, device="cuda"),
    "norse": backend_unimplemented,
    "brian2": sim_neurons_brian2,
}


class LIF:
    def __init__(self, backend="default"):
        self.backend = backend.split(":", 1)[0]

    @property
    def model(self):
        match self.backend:
            case "default" | "nest":
                return "iaf_psc_delta"
            case "bindsnet":
                return bn.nodes.LIFNodes
            case "brian2":
                return {
                    "model": """
                        dv/dt = -v/tau : volt (unless refractory)
                    """,
                    "threshold": "v > vt",
                    "reset": "v=0 * mV",
                    "namespace": {"tau": 10 * br.ms, "vt": 15 * br.mV},
                    "refractory": 2 * br.ms,
                    "method": "euler",
                }

    @property
    def params(self):
        if self.backend == "bindsnet":
            return dict(thresh=-55.0, rest=-70.0, reset=-70.0, refrac=2, tc_decay=10.0)
        else:
            return {}


def voltage_slew_to_current(neuron, slew):
    """
    Take a neuron and multiply a voltage slew by the membrane capacitance in
    order to turn it into the current that would have produced that slew.
    """
    return slew * nest.GetStatus(neuron[0])[0].get("C_m", 1.0)


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
    def connect_bindsnet(self, neurons):
        name = self.__class__.__name__
        raise NotImplementedError(f"{name} does not support BindsNET.")

    def connect_nest(self, neurons, model_name=None):
        name = self.__class__.__name__
        raise NotImplementedError(f"{name} does not support NEST.")

    def connect_brian2(self, neurons):
        name = self.__class__.__name__
        raise NotImplementedError(f"{name} does not support Brian2.")


class CombinedConnectivity(Connectivity):
    def __init__(self, *connectivities):
        self.connectivities = connectivities

    def connect_bindsnet(self, neurons):
        for conn in self.connectivities:
            conn.connect_bindsnet(neurons)

    def connect_nest(self, neurons, model_name=None):
        for conn in self.connectivities:
            conn.connect_nest(neurons, model_name)

    def connect_brian2(self, neurons):
        for conn in self.connectivities:
            conn.connect_brian2(neurons)


class RandomConnectivity(Connectivity):
    def __init__(self, N, q, delay=5.0):
        self.N = N
        self.q = q
        self.delay = delay

    def connect_bindsnet(self, net):
        M = net.layers["neurons"].n
        topology = torch.zeros(M, M)
        for i in range(M):
            idces = torch.randperm(M)[: self.N]
            topology[idces[: self.N // 2], i] = self.q
            topology[idces[self.N // 2 :], i] = -self.q
        conn = bn.topology.Connection(
            net.layers["neurons"], net.layers["neurons"], w=topology, delay=self.delay
        )
        net.add_connection(conn, "neurons", "neurons")

    def connect_nest(self, neurons, model_name):
        for q in (self.q, -self.q):
            weight = psp_corrected_weight(neurons[0], q, model_name)
            nest.Connect(
                neurons,
                neurons,
                dict(rule="fixed_indegree", indegree=self.N // 2),
                dict(synapse_model="static_synapse", weight=weight, delay=self.delay),
            )

    def connect_brian2(self, grp):
        syn = br.Synapses(
            grp,
            grp,
            "w : volt",
            on_pre="v_post += w",
            namespace=dict(q=self.q * br.mV),
            delay=self.delay * br.ms,
        )
        i = np.hstack(
            [np.random.choice(len(grp), self.N, False) for _ in range(len(grp))]
        )
        j = np.repeat(np.arange(len(grp)), self.N)
        syn.connect(i=i, j=j)
        syn.w = "q * (-1)**int(rand() < 0.5)"
        return syn


class BernoulliAllToAllConnectivity(Connectivity):
    def __init__(self, p, q):
        self.p = p
        self.q = q

    def connect_nest(self, neurons, model_name=None):
        for q in (self.q, -self.q):
            w = psp_corrected_weight(neurons[0], q, model_name)
            nest.Connect(
                neurons,
                neurons,
                "all_to_all",
                dict(
                    synapse_model="bernoulli_synapse", weight=w, p_transmit=self.p / 2
                ),
            )

    def connect_brian2(self, grp):
        # Suppress code generation errors because there's no other way to
        # avoid a warning literally every timestep apparently.
        br.codegen.generators.base.logger.log_level_error()
        syn = br.Synapses(
            grp,
            grp,
            "",
            namespace=dict(q_syn=self.q * br.mV, p=self.p),
            on_pre="v_post += q_syn * (-1)**int(rand() < 0.5) * (rand() < p)",
        )
        syn.connect(condition="i!=j")
        return syn


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
        path = os.path.join(figdir(), fname + ext)
        f.savefig(path, **save_args)


try:
    _figdir_before_reload = figdir.dir
except NameError:
    _figdir_before_reload = "figures"


def figdir(path=None):
    if path is not None:
        path = os.path.expanduser(path.strip())
        if path[0] == "/":
            figdir.dir = path
        else:
            figdir.dir = os.path.join("figures", path)
        if not os.path.exists(figdir.dir):
            os.makedirs(figdir.dir)
    return os.path.abspath(figdir.dir)


figdir.dir = _figdir_before_reload


def fitted_curve(f, x, y):
    return f(x, *optimize.curve_fit(f, x, y, method="trf")[0])


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
                ratehats = fitted_curve(transfer_function, R, rates)
                errses[model].append(norm_err(rates, ratehats))
                pbar.update()
    return errses


class Job:
    def __init__(self, q, item):
        self._q = q
        self._item = item
        self.params = item["params"]
        self.retries_allowed = item.get("retries_allowed", 3)

    def requeue(self):
        if self.retries_allowed > 0:
            self._item["retries_allowed"] = self.retries_allowed - 1
            self.q.put(self._item)
            return True
        else:
            return False


def become_worker(what, how):
    q = MessageBroker().get_queue(f"{os.environ['S3_USER']}/{what}-job-queue")

    try:
        while True:
            # Keep popping queue items and fitting HMMs with those parameters.
            job = Job(q, q.get())

            try:
                how(job)
            finally:
                # Always issue task_done, even if the worker failed. If the
                # task counts are misaligned, log it but continue.
                try:
                    q.task_done()
                except ValueError as e:
                    print("Queue misaligned:", e)

    # If there are no more jobs, let the worker quit.
    except queue.Empty:
        print("No more jobs in queue.")

    # Any other exception is a problem with the worker, so put the job
    # back in the queue unaltered and quit. Also issue task_done because we
    # are not going to process the original job.
    except BaseException as e:
        print(f"Worker terminated with exception {e}.")
        q.put(job._item)
        print("Job requeued.")
