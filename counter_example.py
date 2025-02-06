from math import comb
from typing import Tuple, Callable

from scipy.stats import norm
from statsmodels.stats.proportion import proportion_confint
import numpy as np


def multinomial_event_probability(
    p: Tuple[float, float, float], n: int, event: Callable[[Tuple[int, int, int]], bool]
) -> float:
    """
    Calculates the probability of a given event under a multinomial distribution.

    Parameters:
    - p: Tuple of probabilities for each category (must sum to 1).
    - n: Total number of trials.
    - event: A callable that takes a tuple of counts and returns True if the event occurs.

    Returns:
    - Total probability of the event occurring.
    """
    total_probability = 0.0

    # Iterate over all possible counts for x0
    for x0 in range(n + 1):
        # Iterate over all possible counts for x1, given x0
        for x1 in range(n - x0 + 1):
            # x2 is determined since x0 + x1 + x2 = n
            x2 = n - x0 - x1
            counts = (x0, x1, x2)
            if event(counts):
                # Calculate the multinomial probability mass function (pmf)
                pmf = (
                    comb(n, x0)
                    * comb(n - x0, x1)
                    * (p[0] ** x0)
                    * (p[1] ** x1)
                    * (p[2] ** x2)
                )
                total_probability += pmf

    return total_probability


def counterexample():
    """
    Demonstrates the impact of improperly applying the Bonferroni correction
    on confidence intervals and the associated probabilities.
    """
    # Define the true probabilities for each category
    p = (0.5, 0.05, 0.45)
    n = 10  # Total number of trials
    alpha = 0.05  # Significance level

    print("Degenerate event")

    def event_degenerate(x: Tuple[int, int, int]):
        # Event where counts are strictly decreasing: x0 > x1 > x2
        return x[0] > x[1] > x[2]

    # Calculate the probability of the degenerate event
    probability = multinomial_event_probability(p, n, event_degenerate)
    print(f"Probability of the event: {probability:.4f}")

    print("\nFirst Radius")

    # Proper application of Bonferroni correction
    def event_bonferroni(x: Tuple[int, int, int]):
        # Compute confidence intervals with adjusted alpha
        # Clopper–Pearson (exact) interval
        adjusted_alpha = 2 * alpha / 3  # Adjusted for three comparisons
        p0_lower = proportion_confint(
            count=x[0], nobs=n, alpha=adjusted_alpha, method="beta"
        )[0]
        p1_upper = proportion_confint(
            count=x[1], nobs=n, alpha=adjusted_alpha, method="beta"
        )[1]
        p2_upper = proportion_confint(
            count=x[2], nobs=n, alpha=adjusted_alpha, method="beta"
        )[1]
        # Check if the minimum observed difference exceeds the confidence interval difference
        return min((x[0] - x[1]) / n, (x[0] - x[2]) / n) > min(
            p0_lower - p1_upper, p0_lower - p2_upper
        )

    print("Bonferroni Correction Applied")
    probability = multinomial_event_probability(p, n, event_bonferroni)
    print(f"Probability of the event: {probability:.4f}")
    print(f"Theoretical minimum probability (1 - alpha): {1 - alpha:.4f}")
    assert (
        probability >= 1 - alpha
    )  # Should pass if Bonferroni correction is properly applied

    # Improper application without Bonferroni correction
    def event_unadjusted(x: Tuple[int, int, int]):
        # Compute confidence intervals without adjusting alpha
        p0_lower = proportion_confint(count=x[0], nobs=n, alpha=alpha, method="beta")[0]
        p1_upper = proportion_confint(count=x[1], nobs=n, alpha=alpha, method="beta")[1]
        # Check if the observed difference exceeds the confidence interval difference
        return min((x[0] - x[1]) / n, (x[0] - x[2]) / n) > p0_lower - p1_upper

    print("\nNo Bonferroni Correction Applied")
    probability = multinomial_event_probability(p, n, event_unadjusted)
    print(f"Probability of the event: {probability:.4f}")
    print(f"Theoretical minimum probability (1 - alpha): {1 - alpha:.4f}")
    # The assertion is commented out because the probability may not meet the theoretical minimum
    # assert probability >= 1 - alpha

    print("\nSecond Radius")

    # Proper application with Bonferroni correction in transformed space
    def event_bonferroni_transformed(x: Tuple[int, int, int]):
        adjusted_alpha = 2 * alpha / 3  # Adjusted for three comparisons
        # Compute confidence intervals
        p0_lower = proportion_confint(
            count=x[0], nobs=n, alpha=adjusted_alpha, method="beta"
        )[0]
        p1_upper = proportion_confint(
            count=x[1], nobs=n, alpha=adjusted_alpha, method="beta"
        )[1]
        p2_upper = proportion_confint(
            count=x[2], nobs=n, alpha=adjusted_alpha, method="beta"
        )[1]
        # Transform observed proportions using the inverse CDF of the normal distribution
        # z0 = norm.ppf(x[0] / n)
        # z1 = norm.ppf(x[1] / n)
        # z2 = norm.ppf(x[2] / n)
        # Clip the values to avoid numerical instability
        eps = 1e-12
        z0 = norm.ppf(np.clip(x[0] / n, eps, 1 - eps))
        z1 = norm.ppf(np.clip(x[1] / n, eps, 1 - eps))
        z2 = norm.ppf(np.clip(x[2] / n, eps, 1 - eps))
        # Transform confidence bounds
        z0_lower = norm.ppf(p0_lower)
        z1_upper = norm.ppf(p1_upper)
        z2_upper = norm.ppf(p2_upper)
        # Check if the minimum observed difference exceeds the confidence interval difference
        return min(z0 - z1, z0 - z2) > min(z0_lower - z1_upper, z0_lower - z2_upper)

    print("Bonferroni Correction Applied in Transformed Space")
    probability = multinomial_event_probability(p, n, event_bonferroni_transformed)
    print(f"Probability of the event: {probability:.4f}")
    print(f"Theoretical minimum probability (1 - alpha): {1 - alpha:.4f}")
    assert (
        probability >= 1 - alpha
    )  # Should pass if Bonferroni correction is properly applied

    # Improper application without Bonferroni correction in transformed space
    def event_unadjusted_transformed(x: Tuple[int, int, int]):
        # Compute confidence intervals without adjusting alpha
        p0_lower = proportion_confint(count=x[0], nobs=n, alpha=alpha, method="beta")[0]
        p1_upper = proportion_confint(count=x[1], nobs=n, alpha=alpha, method="beta")[1]
        # Transform observed proportions
        # z0 = norm.ppf(x[0] / n)
        # z1 = norm.ppf(x[1] / n)
        # z2 = norm.ppf(x[2] / n)
        # Clip the values to avoid numerical instability
        eps = 1e-12
        z0 = norm.ppf(np.clip(x[0] / n, eps, 1 - eps))
        z1 = norm.ppf(np.clip(x[1] / n, eps, 1 - eps))
        z2 = norm.ppf(np.clip(x[2] / n, eps, 1 - eps))
        # Transform confidence bounds
        z0_lower = norm.ppf(p0_lower)
        z1_upper = norm.ppf(p1_upper)
        # Check if the observed difference exceeds the confidence interval difference
        return min(z0 - z1, z0 - z2) > z0_lower - z1_upper

    print("\nNo Bonferroni Correction Applied in Transformed Space")
    probability = multinomial_event_probability(p, n, event_unadjusted_transformed)
    print(f"Probability of the event: {probability:.4f}")
    print(f"Theoretical minimum probability (1 - alpha): {1 - alpha:.4f}")
    # The assertion is commented out because the probability may not meet the theoretical minimum
    # assert probability >= 1 - alpha


if __name__ == "__main__":
    counterexample()
