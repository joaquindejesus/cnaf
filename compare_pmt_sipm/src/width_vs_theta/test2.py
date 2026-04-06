import numpy as np
from scipy.stats import poisson
from scipy.optimize import fsolve

def truncated_poisson_mean(lambda_full):
    """
    Compute the mean of a Poisson truncated at w>=3
    """
    trunc_norm = 1 - poisson.cdf(2, lambda_full)
    # sum w*P(w) from w=3 to large enough number
    w = np.arange(3, int(lambda_full + 10*np.sqrt(lambda_full)) + 1)
    mean_trunc = np.sum(w * poisson.pmf(w, lambda_full)) / trunc_norm
    return mean_trunc

def lambda_from_truncated_mean(w_trunc_mean):
    """
    Given an observed truncated mean <w> (w>=3),
    find the full Poisson mean lambda
    """
    def func(lambda_guess):
        return truncated_poisson_mean(lambda_guess) - w_trunc_mean
    lambda_guess = w_trunc_mean  # initial guess
    lambda_full = fsolve(func, lambda_guess)[0]
    return lambda_full

def efficiency_from_lambda(lambda_full):
    """
    Compute efficiency (P(w>=3)) from full Poisson mean
    """
    return 1 - poisson.cdf(2, lambda_full)

def efficiency_from_truncated_mean(w_trunc_mean):
    """
    Convenience function: given truncated <w> (w>=3),
    return efficiency accounting for missing w<3
    """
    lambda_full = lambda_from_truncated_mean(w_trunc_mean)
    return efficiency_from_lambda(lambda_full)

###
# observed truncated means for different zenith angles


x = [1.033552976935475, 1.102486242864234, 1.1710407384877488, 1.2419230485558508, 1.3079747260299546, 1.3790015078541027]
y = [4.7329745778437555, 4.754653937947494, 4.849797444686819, 4.940789473684211, 4.90990099009901, 5.084920634920635]

trunc_mean_vs_zenith = y # example <w>_trunc(theta)


# compute efficiencies
eff_model_vs_zenith = np.array([efficiency_from_truncated_mean(w) for w in trunc_mean_vs_zenith])

print("Efficiencies with model without anchored:", eff_model_vs_zenith)

# anchor to lab
eps0 = 0.87  # measured vertical efficiency

eff_model_vert = eff_model_vs_zenith[0]

# anchor to measured vertical efficiency
eff_vs_zenith = eps0 * eff_model_vs_zenith / eff_model_vert

print("Efficiencies with model anchored:", eff_vs_zenith)
