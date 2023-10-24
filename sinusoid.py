# Sinusoid.py
# Testing what happens when a sinusoidal input is provided to the LIF neurons.
import matplotlib.pyplot as plt
import nest
import numpy as np
from scipy import optimize

from mfsupport import (LIF, CombinedConnectivity, Connectivity,
                       RandomConnectivity, figure, firing_rates,
                       parametrized_F_Finv, softplus_ref)

plt.ion()
plt.rcParams["figure.dpi"] = 300
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
        nest.Connect(input, neurons, "all_to_all", dict(weight=self.q))
        nest.Connect(input, neurons, "all_to_all", dict(weight=-self.q))

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
# Try the sinusoidal input, grab its input rate, and plot the predicted vs. actual
# firing rate at each time for the model.

N = 100
M = 10000
T = 2e3
connectivity = CombinedConnectivity(
    input := SinusoidalInput(10e3, 10e3, 1.0, q),
    RandomConnectivity(N, q),
)

_, sd = firing_rates(
    model=model,
    q=q,
    M=M,
    T=T,
    dt=dt,
    connectivity=connectivity,
    return_times=True,
    R_max=0.0,
    uniform_input=True,
    cache=False,
    backend=backend,
    progress_interval=10.0,
)

t, r = sd.population_firing_rate(bin_size=2.0, average=True)
t = t[:-1]

last_r, r_pred = 0.0, []
r_inputs = input.firing_rate(t)
for r_input in r_inputs:
    F, _ = parametrized_F_Finv(p, r_input, N, q)
    last_r = optimize.fixed_point(F, last_r, method="iteration", maxiter=5000)
    r_pred.append(last_r)
r_pred = np.array(r_pred)

input_percentage = np.mean(1 / (1 + r_inputs / r_pred / N))
print(f"{input_percentage:.2%} of the input was recurrent.")

with figure("Sinusoidal example"):
    plt.plot(t, r * 1e3, label="Actual")
    plt.plot(t, r_pred, label="Predicted")
    plt.xlabel("Time (ms)")
    plt.ylabel("Firing rate (Hz)")
    plt.legend()


# %%
# ISI distribution.
# The units must not be very Poisson, or the approximation would work!

isis = sd.interspike_intervals()
mean = np.array([np.mean(isi) for isi in isis])
std = np.array([np.std(isi) for isi in isis])

with figure("ISI distribution", save_exts=[]) as f:
    ax = f.gca()
    ax.set_aspect("equal")
    ax.scatter(mean, std)
    xa, xb = ax.get_xlim()
    ya, yb = ax.get_ylim()
    ab = min(xa, ya), max(xb, yb)
    ax.plot(ab, ab, "k:")
    ax.set_xlim(ab)
    ax.set_ylim(ab)
    ax.set_xlabel("Mean ISI (ms)")
    ax.set_ylabel("Std. dev. ISI (ms)")
