# Sinusoid.py
# Testing what happens when a sinusoidal input is provided to the LIF neurons.
import matplotlib.pyplot as plt
import nest
import numpy as np
from matplotlib.ticker import PercentFormatter
from scipy import optimize
from tqdm import tqdm

from mfsupport import (LIF, CombinedConnectivity, PoissonInput,
                       RandomConnectivity, figure, firing_rates, softplus_ref)


plt.ion()
if "elsevier" in plt.style.available:
    plt.style.use("elsevier")


# %%
# Get the SoftPlus fit for the model we'll be using.

model = LIF
eta = 0.8
q = 3.0
dt = 0.1
backend = "nest"

R, rates = firing_rates(
    model=model, eta=eta, q=q, dt=dt, T=1e5, M=100, sigma_max=10.0, backend=backend
)
p = optimize.curve_fit(softplus_ref, R, rates, method="trf")[0]


# %%
# Simulate the model for each value of N.


def sim_sinusoid(N, M=10000, T=2e3, bin_size_ms=10.0, warmup_time=250.0):
    t = np.arange(0, T, bin_size_ms)
    r = firing_rates(
        model=model,
        q=q,
        M=M,
        T=T,
        dt=dt,
        connectivity=CombinedConnectivity(
            input := PoissonInput(eta, q, 10e3, 3e3, 1.0),
            RandomConnectivity(N, eta, q, delay=nest.random.uniform(1.0, 10.0)),
        ),
        return_times=True,
        R_max=0.0,
        uniform_input=True,
        cache=False,
        backend=backend,
        progress_interval=None,
        warmup_time=warmup_time,
    )[1].binned(bin_size_ms) / (M * bin_size_ms / 1e3)

    last_r, r_pred = 0.0, []
    r_inputs = input.firing_rate(t + warmup_time)
    for r_input in r_inputs:
        # This discrete-time model is what the CT model reduces to when
        # integrated using forward Euler with timestep equal to its time
        # constant, so just use this for now.
        last_r = softplus_ref(r_input + N * last_r, *p)
        r_pred.append(last_r)

    return t, r_inputs, np.array(r_pred), r


# Ns = np.logspace(1, 3, 101).astype(int)
Ns = [0]
results = [sim_sinusoid(N) for N in tqdm(Ns)]


# %%
# Plot the results.

with figure("Sinusoidal Example"):
    t, r_inputs, r_pred, r = results[len(results) // 2]
    plt.plot(t, r, label="Actual")
    plt.plot(t, r_pred, label="Predicted")
    plt.xlabel("Time (ms)")
    plt.ylabel("Firing rate (Hz)")
    plt.legend()

percentages = []
errors = []

for N, (t, r_inputs, r_pred, r) in zip(Ns, results):
    input_percentage = np.mean(1 / (1 + r_inputs / r_pred / N))
    frmse = np.sqrt(np.mean((r - r_pred) ** 2)) / np.mean(r)
    percentages.append(input_percentage)
    errors.append(frmse)
    print(f"With {input_percentage:.1%} recurrent input, error was {frmse:.1%}.")

with figure("Sinusoidal Input Error") as f:
    ax = f.gca()
    ax.plot(percentages, errors)
    ax.set_xlabel("Amount of Recurrent Input")
    ax.set_ylabel("Error")
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
