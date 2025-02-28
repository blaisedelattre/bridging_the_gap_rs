import math
import numpy as np
from scipy.stats import norm
from scipy.integrate import quad
from scipy.optimize import brentq


def compute_lipschitz_constant_smoothed_classifier(L, r, sigma):
    """
    Computes the Lipschitz constant of a Lipschitz neural network after Gaussian smoothing.

    This function implements the formula from:

        "The Lipschitz-Variance-Margin Tradeoff for Enhanced Randomized Smoothing"
        Blaise Delattre, Alexandre Araujo, Quentin Barthélemy, Alexandre Allauzen.
        https://openreview.net/forum?id=C36v8541Ns

    Given a function f: R^d → [0, r] with Lipschitz constant L and Gaussian noise with standard deviation sigma,
    the Lipschitz constant of the smoothed function, denoted L(𝑓̃), is given by:

        L(\tilde{f}) = L * erf( r / (sqrt(2) * L * sigma) )

    Parameters:
        L (float): Lipschitz constant of the original function f.
        r (float): Upper bound on the output range of f.
        sigma (float): Standard deviation of the Gaussian noise.

    Returns:
        float: The Lipschitz constant of the smoothed function 𝑓̃.
    """
    return L * math.erf(r / (math.sqrt(2) * L * sigma))


def compute_radius_tsuzuku_for_smoothed_classifier(margin, r, L, sigma):
    """
    Computes the certified radius for a smoothed classifier based on Tsuzuku's method.

    This implementation is based on the paper:

        "Lipschitz-Margin Training: Scalable Certification of Perturbation Invariance for Deep Neural Networks" Tsuzuku et al. (2018)
        https://arxiv.org/abs/1802.04034

    The certified radius is given by:

        Radius = margin / (sqrt(2) * L(\tilde{f}))

    where L(\tilde{f}) is the Lipschitz constant of the smoothed classifier computed by
    compute_lipschitz_constant_smoothed_classifier.

    Parameters:
        margin (float): The classification margin between top and second class logits.
        r (float): Upper bound on the output range of the classifier logits.
        L (float): The Lipschitz constant of the original network f.
        sigma (float): Standard deviation of the Gaussian noise used for smoothing.

    Returns:
        float: The certified radius.
    """
    lipschitz_smoothed = compute_lipschitz_constant_smoothed_classifier(L, r, sigma)
    return margin / (np.sqrt(2) * lipschitz_smoothed)


def phi_sigma(s, sigma):
    """
    Computes the Gaussian cumulative distribution function (CDF)
    for a N(0, sigma^2) at point s.

    Parameters:
        s (float): The value at which to evaluate the CDF.
        sigma (float): The standard deviation of the Gaussian.

    Returns:
        float: The CDF value at s.
    """
    return norm.cdf(s / sigma)


def compute_s0(p1, L, sigma):
    """
    Computes the switching point s0 for a given p1, L, and sigma.

    This function solves the equation:
        p1 = 1 - L * ∫[s0, s0 + 1/L] phi_sigma(s, sigma) ds

    Parameters:
        p1 (float): The probability value associated with the top class.
        L (float): The Lipschitz constant of the original function.
        sigma (float): The standard deviation of the Gaussian noise.

    Returns:
        float: The computed switching point s0.

    Raises:
        RuntimeError: If the root-finding procedure fails to bracket the solution.
    """

    def equation(s0):
        integral, _ = quad(phi_sigma, s0, s0 + 1 / L, args=(sigma,))
        return 1 - L * integral - p1

    # Initial guess based on the inverse CDF of 1-p1
    s0_initial = norm.ppf(1 - p1) * sigma

    lower_bound = s0_initial - 5 * sigma
    upper_bound = s0_initial + 5 * sigma

    max_iterations = 10
    iteration = 0
    while iteration < max_iterations:
        f_lower = equation(lower_bound)
        f_upper = equation(upper_bound)
        if f_lower * f_upper < 0:
            break
        lower_bound -= 5 * sigma
        upper_bound += 5 * sigma
        iteration += 1
    else:
        raise RuntimeError("Root finding failed. Please check the inputs.")

    s0 = brentq(equation, lower_bound, upper_bound)
    return s0


def compute_lipschitz_constant_quantile_smoothed_classifier(p1, L, sigma):
    """
    Computes the Lipschitz constant of the smoothed classifier composed with Gaussian quantile function.

    "Bridging the Theoretical Gap in Randomized Smoothing"
    Blaise Delattre, Paul Caillon, Quentin Barthélemy, Erwan Fagnou, Alexandre Allauzen.
    https://openreview.net/forum?id=AZ6T7HdCRt

    Given a function f: R^d → [0, r] with Lipschitz constant L and Gaussian noise with
    standard deviation sigma, the smoothed function 𝑓̃ has a Lipschitz constant computed as:

        L(Phi^{-1} \circ \tilde{f}, B(x, rho)) = L * (Phi_sigma(s0 + 1/L, sigma) - Phi_sigma(s0, sigma)) / [ (1/sqrt(2π)) * exp( - (Φ⁻¹(p1))² / 2 ) ],

    where s0 is obtained by solving:

        p1 = 1 - L * ∫[s0, s0 + 1/L] phi_sigma(s, sigma) ds.

    Parameters:
        p1 (float): A probability value associated with the top class.
        L (float): The Lipschitz constant of the original function f.
        sigma (float): The standard deviation of the Gaussian noise.

    Returns:
        float: The Lipschitz constant of the smoothed function 𝑓̃.
    """
    s0 = compute_s0(p1, L, sigma)
    Phi_s0 = phi_sigma(s0, sigma)
    Phi_s1 = phi_sigma(s0 + 1 / L, sigma)
    numerator = L * (Phi_s1 - Phi_s0)

    Phi_inv_p1 = norm.ppf(p1)
    denominator = (1 / np.sqrt(2 * np.pi)) * np.exp(-((Phi_inv_p1) ** 2) / 2)

    if denominator == 0:
        return np.nan
    lipschitz_smoothed = numerator / denominator
    return lipschitz_smoothed


def compute_term_R_mono(p1, L, sigma):
    """
    Computes a term for the tighter certified radius Rmono for a given p1, L, and sigma.

    This term is defined by the ratio:

        Rmono(p1) = Φ⁻¹(p1) / L(Phi^{-1} \circ \tilde{f}, B(x, rho)),

    where L(𝑓̃) is the Lipschitz constant of the smoothed classifier computed by
    compute_lipschitz_constant_smoothed.
    Here rho is the radius of the ball around the input x and is taken as 0 as per the paper approximation.

    Parameters:
        p1 (float): A probability value associated with the top class.
        L (float): The Lipschitz constant of the original function f.
        sigma (float): Standard deviation of the Gaussian noise.

    Returns:
        float: The computed term Rmono(p1).
    """
    lipschitz_smoothed = compute_lipschitz_constant_quantile_smoothed_classifier(
        p1, L, sigma
    )
    Phi_inv_p1 = norm.ppf(p1)

    if lipschitz_smoothed == 0:
        return np.nan
    Rmono = Phi_inv_p1 / lipschitz_smoothed
    return Rmono


def compute_R_multi(p1, p2, L, sigma):
    """
    Computes the certified radius Rmulti for a smoothed classifier given probabilities p1 and p2,
    Lipschitz constant L, and noise sigma.

    This function calculates two terms using compute_term_R_mono for p1 and p2, and then computes:

        Rmulti = (term_R_mono(p1) - term_R_mono(p2)) / 2

    Here, p1 and p2 correspond to the top two class probabilities.

    Parameters:
        p1 (float): The probability associated with the top (most confident) class.
        p2 (float): The probability associated with the second most confident class.
        L (float): The Lipschitz constant of the original function.
        sigma (float): Standard deviation of the Gaussian noise.

    Returns:
        float: The computed certified radius Rmulti.
    """
    term1 = compute_term_R_mono(p1, L, sigma)
    term2 = compute_term_R_mono(p2, L, sigma)
    return (term1 - term2) / 2


def compute_R_mono(p1, L, sigma):
    """
    Computes the certified radius Rmono for a smoothed classifier, using the term computed from p1.

    This is a wrapper around compute_term_R_mono, intended for cases where one wishes to compare
    the radii computed for two different probability levels p1 and p2.

    Parameters:
        p1 (float): The probability associated with the top class.
        p2 (float): The probability associated with the second class (unused in this function).
        L (float): The Lipschitz constant of the original function.
        sigma (float): Standard deviation of the Gaussian noise.

    Returns:
        float: The computed certified radius Rmono.
    """
    return compute_term_R_mono(p1, L, sigma)
