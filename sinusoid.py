# Sinusoid.py
# Testing what happens when a sinusoidal input is provided to the LIF neurons.
import matplotlib.pyplot as plt
import nest
import numpy as np
from matplotlib.ticker import PercentFormatter
from scipy import optimize
from tqdm import tqdm

from mfsupport import (LIF, CombinedConnectivity, Connectivity,
                       RandomConnectivity, figure, firing_rates,
                       parametrized_F_Finv, psp_corrected_weight, softplus_ref)

plt.ion()
if "elsevier" in plt.style.available:
    plt.style.use("elsevier")


class SinusoidalInput(Connectivity):
    def __init__(self, R_mean, R_amplitude, frequency_Hz, q):
        self.rate = R_mean
        self.amplitude = R_amplitude
        self.frequency = frequency_Hz
        self.q = q

    def connect_nest(self, neurons, model_name=None):
        # The assumption is that firing rates measure the total of an input
        # with balanced connectivity, so divide it into halves.
        input = nest.Create(
            "sinusoidal_poisson_generator",
            params=dict(
                amplitude=self.amplitude / 2,
                frequency=self.frequency,
                rate=self.rate / 2,
            ),
        )
        for weight in (self.q, -self.q):
            weight = psp_corrected_weight(neurons[0], weight, model_name)
            nest.Connect(input, neurons, "all_to_all", dict(weight=weight))

    def firing_rate(self, t):
        """
        Calculate the firing rate of the internal Poisson neurons at a given
        time in milliseconds.
        """
        f = 2e-3 * np.pi * self.frequency
        return self.rate + self.amplitude * np.sin(f * t)


# %%
# Get the SoftPlus fit for the model we'll be using.

model = LIF
q = 3.0
dt = 0.1
backend = "nest"

R, rates = firing_rates(
    model=model, q=q, dt=dt, T=1e5, M=100, sigma_max=10.0, backend=backend
)
p = optimize.curve_fit(softplus_ref, R, rates, method="trf")[0]


# %%
# Simulate the model for each value of N.

def sim_sinusoid(N, M=10000, T=2e3):
    bin_size_ms = 5.0
    warmup_time = 10.0

    t = np.arange(0, T, bin_size_ms)
    r = (
        firing_rates(
            model=model,
            q=q,
            M=M,
            T=T,
            dt=dt,
            connectivity=CombinedConnectivity(
                input := SinusoidalInput(10e3, 10e3, 1.0, q),
                RandomConnectivity(N, q),
            ),
            return_times=True,
            R_max=0.0,
            uniform_input=True,
            cache=False,
            backend=backend,
            progress_interval=None,
            warmup_time=warmup_time,
        )[1].binned(bin_size_ms)
        / M
        / bin_size_ms
        * 1e3
    )

    last_r, r_pred = 0.0, []
    r_inputs = input.firing_rate(t + warmup_time)
    for r_input in r_inputs:
        F, _ = parametrized_F_Finv(p, r_input, N, q)
        last_r = optimize.fixed_point(F, last_r, method="iteration", maxiter=5000)
        r_pred.append(last_r)
    r_pred = np.array(r_pred)

    return t, r_inputs, r_pred, r


Ns = np.logspace(1, 3, 101).astype(int)
results = [sim_sinusoid(N) for N in tqdm(Ns)]


# %%
# Plot the results.

with figure("Sinusoidal Example"):
    t, r_inputs, r_pred, r = results[50]
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
