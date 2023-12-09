# Queue Simulations
#
# This script just has a giant list of the simulations requested by the final
# figure generation script, checks if each one is already cached, and then
# pushes all the ones that aren't cached onto the queue.
import os

import numpy as np
from braingeneers.iot.messaging import MessageBroker

from mfsupport import (BernoulliAllToAllConnectivity, RandomConnectivity,
                       sim_neurons_nest_eta)

added = 0
model_names = {
    "iaf_psc_delta": "Leaky Integrate-and-Fire",
    "izhikevich": "Izhikevich",
    "hh_psc_alpha": "Hodgkin-Huxley",
}


def _firing_rates_needs_run(
    model,
    q,
    M=500,
    sigma_max=None,
    R_max=None,
    return_times=False,
    uniform_input=False,
    seed=42,
    model_params={},
    **kwargs,
):
    """
    A replacement for mfsupport.firing_rates that instead of running a
    simulation just checks to see if its results are cached, and if not,
    pushes the parameters onto the queue.
    """
    if R_max is None and sigma_max is not None:
        R_max = 1e3 * (sigma_max / q) ** 2
    elif (R_max is None) == (sigma_max is None):
        raise ValueError("Either R_max or sigma_max must be given!")
    R = R_max if uniform_input else np.linspace(0, R_max, num=M)

    return not sim_neurons_nest_eta.check_call_in_cache(
        model=model, q=q, R=R, M=M, seed=seed, model_params=model_params, **kwargs
    )


def firing_rates(**params):
    global added
    if _firing_rates_needs_run(**params):
        added += 1
        print("Queueing", params)
        put_params(params)


def fig2():
    print("Figure 2")
    q = 1.0
    dt = 0.1
    sigma_max = 10.0
    for model in model_names:
        firing_rates(T=1e2, q=q, dt=dt, model=model, sigma_max=sigma_max)

    T = 1e5
    q = 1.0
    dt = 0.1
    sigma_max = 10.0
    N_samples = 100

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

    for model in sample_params:
        for _ in range(N_samples):
            firing_rates(
                T=T,
                q=q,
                dt=dt,
                model=model,
                sigma_max=sigma_max,
                model_params=sample_params[model](),
            )


def fig3():
    print("Figure 3")
    dt = 0.1
    q = 1.0
    sigma_max = 10.0
    M = 100
    Mmax = 10000
    T = 1e5
    Tmax = 1e7
    for m in model_names:
        firing_rates(model=m, q=q, dt=dt, T=T, M=Mmax, sigma_max=sigma_max)
        firing_rates(model=m, q=q, dt=dt, T=Tmax, M=M, sigma_max=sigma_max)


def fig4():
    print("Figure 4")
    dt = 0.1
    q = 5.0
    model = "iaf_psc_delta"
    firing_rates(model=model, q=q, dt=dt, T=1e5, M=100, sigma_max=10.0)


def fig5():
    print("Figure 5")
    dt = 0.1
    M = 10000
    eta = 0.2
    T = 2e3
    model = "iaf_psc_delta"
    conditions = [
        # R_bg, q, annealed_average
        (0.1e3, 5.0, False),
        (10e3, 3.0, False),
    ]
    N_theo = np.arange(30, 91)
    N_sim = np.linspace(N_theo[0], N_theo[-1], 8).astype(int)
    for R_background, q, annealed_average in conditions:
        firing_rates(
            model=model,
            q=q,
            dt=dt,
            T=1e5,
            M=100,
            sigma_max=10.0,
        )
        for N in N_sim:
            if annealed_average:
                connectivity = BernoulliAllToAllConnectivity(N / M, q)
            else:
                connectivity = RandomConnectivity(N, eta, q, delay=5.0)
            same_args = dict(
                model=model,
                q=q,
                dt=dt,
                T=T,
                M=M,
                R_max=R_background,
                progress_interval=10.0,
                uniform_input=True,
                connectivity=connectivity,
                return_times=True,
            )
            firing_rates(warmup_time=1e3, warmup_rate=100e3, **same_args)
            firing_rates(**same_args)


def fig6():
    print("Figure 6")
    T = 1e5
    model = "iaf_psc_delta"
    sigma_max = 10.0
    t_refs = [0.0, 2.0]
    qs = [0.1, 1.0]
    for t_ref in t_refs:
        for q in qs:
            firing_rates(
                q=q,
                T=T,
                sigma_max=sigma_max,
                model_params=dict(t_ref=t_ref),
                model=model,
                dt=0.001,
            )


def fig7():
    print("Figure 7")
    model = "iaf_psc_delta"
    Tmax = 1e7
    dt = 0.1
    qs = np.geomspace(0.1, 10, num=20)
    for q in qs:
        firing_rates(
            q=q,
            sigma_max=10.0,
            M=100,
            model=model,
            dt=dt,
            T=Tmax,
        )


if __name__ == "__main__":
    mb = MessageBroker()
    qname = f"{os.environ['S3_USER']}/sim-job-queue"
    queue = mb.get_queue(qname)

    def put_params(params):
        queue.put(dict(retries_allowed=3, params=params))

    import sys

    if "--dryrun" in sys.argv or "-n" in sys.argv:
        print("Dry run, not actually queueing anything")
        put_params = lambda params: None  # noqa: F811

    fig2()
    fig3()
    fig4()
    fig5()
    fig6()
    fig7()

    print("Added", added, "jobs to queue")
