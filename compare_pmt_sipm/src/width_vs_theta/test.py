import numpy as np
from scipy.stats import nbinom
from scipy.optimize import minimize

import matplotlib.pyplot as plt

w_vals = np.array([3,4,5,6,7,8,9,10])
p_obs  = np.array([0.287,0.236,0.198,0.140,0.080,0.037,0.016,0.006])

Ntot = 100000  # arbitrary scaling
counts = (p_obs * Ntot).astype(int)


def truncated_nb_pmf(w, mu, k):

    p = k/(k+mu)

    P_full = nbinom.pmf(w, k, p)

    trunc_norm = 1 - nbinom.cdf(2, k, p)

    return P_full / trunc_norm

def neg_log_likelihood(params):

    mu, k = params

    if mu <= 0 or k <= 0:
        return np.inf

    P = truncated_nb_pmf(w_vals, mu, k)

    if np.any(P <= 0):
        return np.inf

    return -np.sum(counts * np.log(P))



result = minimize(
    neg_log_likelihood,
    x0=[5,2],   # initial guess
    bounds=[(0.1,None),(0.1,None)]
)

mu_fit, k_fit = result.x

P_model = truncated_nb_pmf(w_vals, mu_fit, k_fit)

fig, ax = plt.subplots(1, 1, figsize=(16, 7))

ax.scatter(w_vals, p_obs, label="data")
ax.plot(w_vals, P_model, label="NB fit")
ax.legend()
ax.set_xlabel("w")
ax.set_ylabel("P(w | w>=3)")

figname = "test.png"


# chi2 goodness of fit
expected = Ntot * P_model

chi2 = np.sum((counts - expected)**2 / expected)

ndf = len(w_vals) - 2 - 1  # bins - parameters - normalization

print("chi2/ndf =", chi2, ndf, chi2/ndf)

# effficiency
efficiency = 1 - nbinom.cdf(2, k_fit, k_fit/(k_fit+mu_fit))

print(f"efficiency = {efficiency:.3f}")



# TEst 2
import numpy as np
from scipy.stats import poisson
from scipy.optimize import minimize

w_vals = np.array([3,4,5,6,7,8,9,10])
counts = np.array([287,236,198,140,80,37,16,6])

def truncated_poisson_pmf(w, lam):

    P = poisson.pmf(w, lam)

    norm = 1 - poisson.cdf(2, lam)

    return P / norm


def neglogL(lam):

    if lam <= 0:
        return np.inf

    P = truncated_poisson_pmf(w_vals, lam)

    return -np.sum(counts*np.log(P))


res = minimize(lambda x: neglogL(x[0]), x0=[4])

lam_fit = res.x[0]

print(f"lambda_fit truncated Poiss = {lam_fit:.3f}")

P_model_poiss = truncated_poisson_pmf(w_vals, lam_fit)

ax.plot(w_vals, P_model_poiss, "--^", label="Truncated Poiss")

fig.savefig(figname, bbox_inches="tight")
