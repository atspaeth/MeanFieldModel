import os
import itertools
import pickle
import numpy as np
import matplotlib.pyplot as plt
import braingeneers.analysis as ba
from scipy import optimize, special
from tqdm import tqdm
from contextlib import contextmanager
from joblib import Memory

os.environ['PYNEST_QUIET'] = '1'
memory = Memory(location='.cache', verbose=0)

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
                globals()[func.__name__] = import_module(full_name)
                func()
                return getattr(globals()[func.__name__], attr)
        return stub()
    return wrapper

@lazy_package
def nest():
    nest.set_verbosity('M_WARNING')
    from pynestml.frontend.pynestml_frontend import generate_nest_target
    generate_nest_target('models/', '/tmp/nestml-mfsupport/',
                         module_name='mfmodule')
    nest.Install('mfmodule')

@lazy_package('bindsnet.network')
def bn(): pass

@lazy_package
def torch(): pass

@lazy_package('brian2')
def br(): pass


def _softplus(arg):
    '''
    Just the freaky nonlinear part of softplus, implementing
    a large-argument correction.
    '''
    arg = np.asarray(arg)
    if len(arg.shape) > 0:
        out = arg * 1.0
        nonlinear = arg < 20
        out[nonlinear] = np.log1p(np.exp(arg[nonlinear]))
        return out
    else:
        return _softplus([arg])[0]


def _softplus_inv(ret):
    'Inverse of _softplus()'
    ret = np.asarray(ret)
    if len(ret.shape) > 0:
        arg = ret * 1.0
        nonlinear = ret < 20
        arg[nonlinear] = np.log(np.expm1(ret[nonlinear]))
        return arg
    else:
        return _softplus_inv([ret])[0]


def softplus(Rs, x0, b, c):
    'The SoftPlus transfer function.'
    return c/b * _softplus(b*(np.sqrt(Rs) - x0))


def softplus_inv(rs, x0, b, c):
    'Inverse of SoftPlus transfer function.'
    return (x0 + _softplus_inv(b*rs/c)/b)**2


def _refractory(r, t_ref):
    'Correct a rate r in Hz to add a refractory period t in ms.'
    return 1/(1/r + t_ref*1e-3)


def _refractory_inv(r, t_ref):
    'Inverse of _refractory.'
    return 1/(1/r - t_ref*1e-3)


def softplus_ref(Rs, x0, b, c, t_ref):
    'Empirical functional form for firing rate as a function of noise.'
    return _refractory(softplus(Rs, x0, b, c), t_ref)


def softplus_ref_q_dep(Rs_and_qs, x0, b, c, d, t_ref):
    'Refractory Softplus with q dependence incorporated in beta.'
    Rs, qs = Rs_and_qs
    fr = c/b * _softplus(b*(1 + d*qs)*(qs * np.sqrt(Rs) - x0))
    return _refractory(fr, t_ref)


def softplus_ref_q_dep_inv(rs, q, x0, b, c, d, t_ref):
    'No single inverse exists, but use this for a given q.'
    sp_arg = _softplus_inv(b/c * _refractory_inv(rs, t_ref))
    R_term = sp_arg / b / (1 + d*q) + x0
    return (R_term / q)**2


def softplus_ref_inv(rs, x0, b, c, t_ref):
    'Inverse of refractory SoftPlus function.'
    return softplus_inv(_refractory_inv(rs, t_ref), x0, b, c)


def parametrized_F_Finv(μ_softplus, R_background, N, q=None):
    '''
    Return the transfer function for a recurrently connected network whose
    SoftPlus or Refractory SoftPlus transfer function is parameterized by the
    provided μ_softplus, with given noise level and connectivity parameters.
    '''
    match len(μ_softplus):
        case 3:
            tf, tfinv = softplus, softplus_inv
        case 4:
            tf, tfinv = softplus_ref, softplus_ref_inv
        case 5:
            if q is None:
                raise ValueError(
                    'Must provide q for q-dependent transfer function')
            def tf(R, *a):
                return softplus_ref_q_dep((R, q*np.ones_like(R)), *a)
            def tfinv(R, *a):
                return softplus_ref_q_dep_inv(R, q, *a)
        case _:
            raise ValueError(
                'Invalid number of parameters for transfer function.')

    def F(r):
        return tf(R_background + N*r, *μ_softplus)

    def Finv(r):
        return (tfinv(r, *μ_softplus) - R_background) / N

    return F, Finv


@np.vectorize
def _rs79_isi(sigma, theta, S):
    if sigma == 0.0:
        return np.inf
    y = S/sigma * np.sqrt(2/theta)
    term2 = np.pi * special.erfi(y / np.sqrt(2))
    n = np.arange(25)+1
    term1s = y**(2*n) / n / special.factorial2(2*n-1)
    return theta * (np.sum(term1s) + term2) / 2


def rs79(R, q, theta, S, t_ref):
    'Mean firing rate from Ricciardi & Sacerdote 1979.'
    sigma = q * np.sqrt(R/1e3)
    return _refractory(1e3/_rs79_isi(sigma, theta, S), t_ref)


def zerlaut_erf(R, b, c, k, V0):
    'Transfer function curve fit by Zerlaut et al. 2018, part-manual here.'
    Vx = V0 + b*np.sqrt(R) - c*R
    return k * special.erfc(Vx/np.sqrt(2*R))


def find_fps(r_top, F, Finv, atol=0.1):
    '''
    Find fixed points of the SoftPlus consistency equation, starting with
    one guess at zero and one at r_top. The system undergoes two
    complementary saddle-node bifurcations, so there is a parameter region
    with two stable and one unstable fixed point surrounded by a region with
    only one stable fixed point.

    Returns stable_fps and unstable_fps as lists, either length 2 and 1 or
    length 1 and 0 depending on the location in parameter space.
    '''
    # First look for two stable FPs since we know at least one exists.
    stable_fps = [optimize.fixed_point(F, x, method='iteration',
                                       maxiter=5000)
                  for x in [0, r_top]]

    # Then if they are far enough apart to be considered distinct,
    # use iteration on the inverse dynamics to find the unstable FP.
    if stable_fps[-1] > atol + stable_fps[0]:
        try:
            unstable_fp = optimize.fixed_point(Finv, r_top/2,
                                               method='iteration')
            return stable_fps, [unstable_fp]
        except RuntimeError:
            pass
    else:
        stable_fps = [stable_fps[0]]
    return stable_fps, []


def reset_nest(dt, seed):
    nest.ResetKernel()
    nest.local_num_threads = 10
    nest.resolution = dt
    nest.rng_seed = seed


def firing_rates(model, q, M=500, sigma_max=None, R_max=None, cache=True,
                 return_times=False, uniform_input=False, seed=42,
                 backend='default', **kwargs):
    if R_max is None and sigma_max is not None:
        R_max = 1e3 * (sigma_max / q)**2
    elif (R_max is None) == (sigma_max is None):
        raise ValueError('Either R_max or sigma_max must be given!')

    R = R_max if uniform_input else np.linspace(0, R_max, num=M)
    sim = SIM_BACKENDS[backend] if cache else SIM_BACKENDS[backend].func
    sd = sim(model=model, q=q, R=R, M=M, seed=seed, **kwargs)
    return R, (sd if return_times else sd.rates('Hz'))


@memory.cache(ignore=['progress_interval'])
def sim_neurons_nest(model, q, R, dt, T, M=None, I_ext=None, model_params=None,
                warmup_time=0.0, warmup_rate=None, warmup_steps=10,
                connectivity=None, seed=42, recordables=None,
                progress_interval=1e3):
    '''
    Simulate M Izhikevich neurons using NEST. They are receiving Poisson
    inputs with connection strength q and rate R, and optionally connected
    to each other by calling the given connectivity object's connect()
    method on the neurons after initialization.
    '''
    R = np.atleast_1d(R)
    if M is None:
        M = len(R)

    reset_nest(dt=dt, seed=seed)

    neurons = nest.Create(model, n=M, params=model_params)
    if I_ext is not None:
        neurons.I_e = voltage_slew_to_current(neurons, I_ext)

    noise = nest.Create('poisson_generator', n=len(R),
                        params=dict(rate=R/2))

    if len(R) == 1:
        conn = 'all_to_all'
    elif len(R) == M:
        conn = 'one_to_one'
    else:
        raise ValueError('R must be a scalar or a vector of length M.')

    nest.Connect(noise, neurons, conn,
                 dict(weight=psp_corrected_weight(neurons[0], q)))
    nest.Connect(noise, neurons, conn,
                 dict(weight=psp_corrected_weight(neurons[0], -q)))

    if connectivity is not None:
        connectivity.connect(neurons)

    rec = nest.Create('spike_recorder')
    nest.Connect(neurons, rec)

    if progress_interval is not None:
        pbar = tqdm(total=T + warmup_time, unit='sim sec', unit_scale=1e-3)

    if recordables:
        rec = nest.Create('multimeter',
                          params=dict(record_from=recordables, interval=dt))
        nest.Connect(rec, neurons)

    if warmup_time > 0:
        base_rate = noise.rate
        if warmup_rate is None:
            warmup_rate = 10*base_rate
        for i in range(warmup_steps):
            noise.rate = np.interp(i, [0,warmup_steps],
                                   [warmup_rate, base_rate])
            nest.Simulate(warmup_time / warmup_steps)
            if progress_interval is not None:
                pbar.update(warmup_time / warmup_steps)
        noise.rate = base_rate

    if progress_interval is None:
        nest.Simulate(T)
    else:
        with nest.RunManager():
            nest.Run(T % progress_interval)
            pbar.update(T % progress_interval)
            for _ in range(int(T//progress_interval)):
                nest.Run(progress_interval)
                pbar.update(progress_interval)
        pbar.close()

    # Create SpikeData and trim off the warmup time.
    return ba.SpikeData(rec, neurons, length=T+warmup_time,
                        metadata=rec.events if recordables else {}
                        ).subtime(warmup_time, ...)


def run_net_with_rates(net, R, T, Npre):
    '''
    Given a BindsNET network whose input layer is called 'source' and a set
    of input rates R, generate balanced Poisson inputs and run the network.
    '''
    # We have to divide R by Npre because it's split into the Npre balanced
    # parts, but also change units from Hz to spikes per time step.
    steps = int(T / net.dt)
    R = R.repeat_interleave(Npre) / Npre
    rates = torch.vstack([R * net.dt/1e3]*steps)
    input_data = torch.poisson(rates).byte()
    net.run(inputs={'source': input_data}, time=T)


@memory.cache(ignore=['device', 'progress_interval'])
def sim_neurons_bindsnet(model, q, R, dt, T, M=None, seed=42,
                         model_params={}, connectivity=None,
                         warmup_time=0.0, warmup_steps=10,
                         progress_interval=None, device='cuda'):
    '''
    Simulate M neurons of the given model using BindsNET. They receive
    balanced Poisson inputs with connection strength q and rate R.
    '''
    with torch.device(device), torch.no_grad():
        torch.random.manual_seed(seed)
        R = torch.atleast_1d(torch.as_tensor(R))
        if M is None:
            M = len(R)
        elif len(R) == 1:
            R = R.repeat(M)
        elif len(R) != M:
            raise ValueError('R must be scalar or length M.')

        # Build the base network and its layers. Split each conceptual
        # Poisson input neuron into Npre separate neurons so weights can be
        # well-defined and no neuron has to fire more than once per step.
        Npre = 42
        net = bn.Network(dt=dt)
        net.to(device)
        source = bn.nodes.Input(n=Npre*M)
        source.to(device)
        neurons = model(n=M, **model_params)
        neurons.to(device)
        net.add_layer(source, name='source')
        net.add_layer(neurons, name='neurons')
        if connectivity is not None:
            connectivity.connect(net)

        # Connect the input to the neurons Npre-to-one with weight ±q,
        # alternating so the odd indices are negative.
        w = torch.eye(M).repeat_interleave(Npre, dim=1)
        w *= (-1)**torch.arange(w.shape[1]) * q
        net.add_connection(
            source='source',
            target='neurons',
            connection=bn.topology.Connection(
                source=source,
                target=neurons,
                w=w.T))

        # If there is a progress interval, use a progress bar. If it is
        # zero, the chunk size for simulations should be 100ms.
        if progress_interval is None:
            progress_interval = 0.0
            pbar = None
        else:
            pbar = tqdm(total=T + warmup_time, unit='sim sec',
                        unit_scale=1e-3)
        if progress_interval <= 0:
            progress_interval = 1e2

        # Do the warmup simulation without recording.
        if warmup_time > 0:
            warmup_T = warmup_time / warmup_steps
            for i in range(warmup_steps):
                warmup_ramp = np.interp(i, [0,warmup_steps], [10,1])
                run_net_with_rates(net, warmup_ramp*R, warmup_T, Npre)
                if pbar: pbar.update(warmup_T)

        # Add recording before finishing the simulation.
        monitor = bn.monitors.Monitor(obj=neurons, state_vars=['s'],
                                      time=int(T/net.dt))
        net.add_monitor(monitor=monitor, name='neurons')

        # Run the simulation broken up into chunks.
        residue = T % progress_interval
        if residue > net.dt:
            run_net_with_rates(net, R, residue, Npre)
        for i in range(int(T/progress_interval)):
            run_net_with_rates(net, R, progress_interval, Npre)
            if pbar: pbar.update(progress_interval)

        if pbar: pbar.close()

        # Grab the monitor's spike matrix and turn it into SpikeData.
        times, _, idces = torch.nonzero(monitor.get('s'), as_tuple=True)
        sd = ba.SpikeData(idces, times*dt, length=float(T), N=M)

    # BindsNET simulations need to be cleared from GPU memory to allow other
    # simulations to follow them. This is weird because references to the
    # tensors on GPU are still in Python memory, so GC them first.
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    return sd


@memory.cache()
def sim_neurons_brian2(model, q, R, dt, T, M=None, seed=42):
    '''
    Simulate M neurons using Brian2. They receive balanced Poisson inputs
    with connection strength q and rate R.

    Since Brian2 has no built-in models, a model must be specified as an
    entire dictionary where 'model' contains a multi-line string with its
    dynamic equations, 'threshold' contains the spike threshold condition as
    a boolean expression, etc.
    '''
    R = np.atleast_1d(R)
    if M is None:
        M = len(R)
    elif  len(R) != M:
        raise ValueError('R must be scalar or length M.')

    br.defaultclock.dt = dt*br.ms
    neurons = br.NeuronGroup(M, **model)
    monitor = br.SpikeMonitor(neurons)
    # All constructed objects must be explicitly named and in scope so Brian
    # can extract them for the run() call.
    if len(R) == 1:
        input_pos = br.PoissonInput(neurons, 'v', 1, R[0]/2*br.Hz, q)
        input_neg = br.PoissonInput(neurons, 'v', 1, R[0]/2*br.Hz, -q)
    else:
        # For each postsynaptic neuron, create Npre separate presynaptic
        # Poisson inputs. The odd ones have negative weights, and the split
        # must be significantly greater than 2 because it's impossible for
        # a single neuron to spike multiple times per step.
        Npre = 42
        source = br.PoissonGroup(Npre*M, R.repeat(Npre)/Npre*br.Hz)
        syn = br.Synapses(source, neurons, on_pre='v_post += (-1)**i * q')
        syn.connect(j=f'int(i/{Npre})')
    br.run(T*br.ms, namespace=dict(q=q*br.mV))
    return ba.SpikeData(monitor.i, monitor.t, length=T, N=M)


def sim_neurons_unimplemented(*args, **kwargs):
    raise NotImplementedError('Brian2 is not yet supported.')


SIM_BACKENDS = dict(
    default=sim_neurons_nest,
    nest=sim_neurons_nest,
    bindsnet=sim_neurons_bindsnet,
    norse=sim_neurons_unimplemented,
    brian2=sim_neurons_brian2)


def voltage_slew_to_current(neuron, slew):
    '''
    Take a neuron and multiply a voltage slew by the membrane capacitance in
    order to turn it into the current that would have produced that slew.
    '''
    return slew * nest.GetStatus(neuron[0])[0].get('C_m', 1.0)


def psp_corrected_weight(neuron, q):
    '''
    Take a neuron and a desired synaptic weight for a delta PSP and return
    the synaptic weight which should be used instead so that this neuron
    will receive an equivalent voltage injection from its PSCs.
    '''
    # Modify the base model name to match the name scheme of the others.
    if neuron[0].model == 'izhikevich':
        model = ['izhikevich', 'psc', 'delta']
    else:
        model = neuron[0].model.split('_')

    # For PSC neurons, it doesn't matter whether they're HH or I&F, but the
    # shape of the PSC does matter. Can assume that all such neurons have
    # membrane capacitance C_m as well as synaptic time constants, except
    # the ones with delta synapses, which inject voltage like Izhikevich.
    if model[1] == 'psc':
        postfix = '_ex' if q > 0 else '_in'
        if len(model) < 3:
            pass  # skip to the NotImplemented at the end
        if model[2] == 'delta':
            return q
        elif model[2] in ('exp', 'alpha'):
            # Alpha and exponential PSCs differ only in normalization.
            corr = 1.0 if model[2] == 'exp' else 1/np.e
            tau = nest.GetStatus(neuron, 'tau_syn'+postfix)[0]
            return q*corr/tau * neuron.C_m
    raise NotImplementedError(f'Model {neuron[0].model} not supported.')


class Connectivity:
    def connect(self, neurons):
        if hasattr(neurons, 'cuda'):
            self.connect_bindsnet(neurons)
        else:
            self.connect_nest(neurons)


class RandomConnectivity(Connectivity):
    def __init__(self, N, q, synapse_params={}):
        self.N = N
        self.q = q
        self._sp = synapse_params

    def connect_bindsnet(self, net):
        M = net.layers['neurons'].n
        topology = torch.zeros(M, M)
        for i in range(M):
            idces = torch.randperm(M)[:self.N]
            topology[i, idces[:self.N//2]] = self.q
            topology[i, idces[self.N//2:]] = -self.q
        conn = bn.topology.Connection(
            net.layers['neurons'], net.layers['neurons'],
            w=topology, **self._sp)
        net.add_connection(conn, 'neurons', 'neurons')

    def connect_nest(self, neurons):
        M = len(neurons)
        for q in (self.q, -self.q):
            weight = psp_corrected_weight(neurons[0], q)
            nest.Connect(neurons, neurons,
                         dict(rule='fixed_indegree',
                              indegree=self.N//2),
                         dict(synapse_model='static_synapse',
                              weight=weight, **self._sp))


class BernoulliAllToAllConnectivity:
    def __init__(self, p, q):
        self.p = p
        self.q = q

    def connect(self, neurons):
        M = len(neurons)
        for q in (self.q, -self.q):
            nest.Connect(neurons, neurons, 'all_to_all',
                         dict(synapse_model='bernoulli_synapse',
                              weight=psp_corrected_weight(neurons[0], q),
                              p_transmit=self.p/2))


class BiasedPoisson:
    def __init__(self, N, r, q):
        self.N = N
        self.r = np.atleast_1d(r)
        self.q = np.atleast_1d(q)
        if len(self.r) == 1 and len(self.q) == 1:
            self.M = None
        elif len(self.r) == 1:
            self.M = len(self.q)
        elif len(self.q) == 1:
            self.M = len(self.r)
        elif len(self.r) == len(self.q):
            self.M = len(self.r)
        else:
            raise ValueError('r and q must be scalar or have length M.')

    def connect(self, neurons):
        if self.M is not None and len(neurons) != self.M:
            raise ValueError('The number of neurons must be equal to M.')

        total_rate = self.N * self.r
        stim = nest.Create('poisson_generator', n=len(total_rate),
                           params=dict(rate=total_rate))

        for n,w,s in zip(neurons, itertools.cycle(self.q),
                         itertools.cycle(stim)):
            # Set the bias current to counteract the mean Poisson input.
            n.I_e -= voltage_slew_to_current(n, s.rate * w)
            nest.Connect(s, n, 'one_to_one',
                         dict(weight=psp_corrected_weight(n, w)))


class CombinedConnectivity:
    def __init__(self, *connectivities):
        self.connectivities = []
        for c in connectivities:
            if isinstance(c, CombinedConnectivity):
                self.connectivities.extend(c.connectivities)
            else:
                self.connectivities.append(c)

    def connect(self, neurons):
        for conn in self.connectivities:
            conn.connect(neurons)


class BalancedPoisson(CombinedConnectivity):
    def __init__(self, N, r, q):
        super().__init__(
            BiasedPoisson(N//2, r, q),
            BiasedPoisson(N//2, r, -q))


@contextmanager
def figure(name, save_args={}, save_exts=['png'], **kwargs):
    'Create a named figure and save it when done.'
    f = plt.figure(name, **kwargs)
    try:
        f.clf()
    except Exception:
        plt.close()
        f = plt.figure(name, **kwargs)

    yield f

    fname = name.lower().strip().replace(' ', '-')
    for ext in save_exts:
        if ext[0] != '.':
            ext = '.' + ext
        path = os.path.join(figdir(), fname + ext)
        f.savefig(path, **save_args)


def figdir(path=None):
    if path is not None:
        path = os.path.expanduser(path.strip())
        if path == '':
            figdir.dir = 'figures'
        elif path[0] == '/':
            figdir.dir = path
        else:
            figdir.dir = os.path.join('figures', path)
        if not os.path.exists(figdir.dir):
            os.makedirs(figdir.dir)
    return os.path.abspath(figdir.dir)
figdir('')
