# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# from IPython import get_ipython

# # This notebook contains more advanced examples

# from __future__ import division, absolute_import, print_function

# get_ipython().run_line_magic('matplotlib', 'inline')

import lmfit
import matplotlib.pyplot as plt
import numpy as np

from resonator import background, reflection, see


# Make some fake data to use for the example fits
f_r = 1e9
frequency = np.linspace(f_r - 1e6, f_r + 1e6, 501)
bg = background.MagnitudePhase().func(
    frequency=frequency, magnitude=0.01, phase=np.pi / 3)
fg = reflection.LinearReflection().func(frequency=frequency,
                                        resonance_frequency=f_r, internal_loss=1e-5, coupling_loss=5e-5)
data = bg * fg + 0.0002 * \
    (np.random.randn(frequency.size) + 1j * np.random.randn(frequency.size))

# ## Use initial params to improve on the guessing function
# The default algorithm used by `lmfit` is Levenberg-Marquardt, which is fast but finds only the local minimum of the residual function given by the initial values. The most common reason for a fit to fail is that the `guess` function provides initial values that are in a local minimum that is not the global minimum. If a data set that looks "reasonable" fails to converge to the correct values, a quick fix is to try different initial values. If best-fit parameters from a previous successful fit are available and the data to be fit is similar, this same technique can be used to accelerate the fit or achieve convergence.

params = lmfit.Parameters()
params.add(name='resonance_frequency', value=1e9)
r = reflection.LinearReflectionFitter(
    frequency=frequency, data=data, params=params)
fig, ax = see.magnitude_vs_frequency(resonator=r)
print(r.result.fit_report())

# ## Use initial parameters to control whether or not to vary a parameter in the fit
# For example, the value of the coupling can be fixed to a value from a simulation.

params = lmfit.Parameters()
params.add(name='coupling_loss', value=5e-5, vary=False)
r = reflection.LinearReflectionFitter(
    frequency=frequency, data=data, params=params)
fig, ax = see.magnitude_vs_frequency(resonator=r)
print(r.result.fit_report())  # Note that coupling_loss is now fixed
plt.show()
