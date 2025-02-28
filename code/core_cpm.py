import torch
from scipy.stats import norm, binom_test
import numpy as np
from math import ceil
from statsmodels.stats.proportion import proportion_confint

class Smooth(object):
    """A smoothed classifier g """

    # to abstain, Smooth returns this int
    ABSTAIN = -1

    def __init__(self, base_classifier: torch.nn.Module, num_classes: int, sigma: float):
        """
        :param base_classifier: maps from [batch x channel x height x width] to [batch x num_classes]
        :param num_classes:
        :param sigma: the noise level hyperparameter
        """
        self.base_classifier = base_classifier
        self.num_classes = num_classes
        self.sigma = sigma

    def get_radius(self, pABar, pBBar, mode="Rmono"):
        if mode == "Rmulti":
            radius = self.sigma/2 * (norm.ppf(pABar) - norm.ppf(pBBar))
        elif mode == "Rmono":
            radius = self.sigma * norm.ppf(pABar)
        else:
            raise ValueError("radius not recognized")
        return radius
    
    def get_alpha_prime(self, alpha, factor=1, mode="Rmono"):
        if mode == "Rmono":
            return alpha
        else:
            return alpha/factor
    

    def certify_from_log_cpm(self, counts_selection, counts_estimation, N, N0, alpha, mode="Rmono"):
        counts_selection = torch.tensor(counts_selection)
        sorted_counts_selection, idxs = torch.sort(counts_selection, descending=True)
        sorted_indices = idxs.tolist()
        sorted_counts_selection = sorted_counts_selection.tolist()

        I1 = sorted_indices[0]
        I2 = sorted_indices[1]
        counts_I1 = sorted_counts_selection[0]
        counts_I2 = sorted_counts_selection[1]

        M_indices = sorted_indices[2:]
        counts_M = sorted_counts_selection[2:]

        buckets = [
            {'labels': [I1], 'n_selection': counts_I1},
            {'labels': [I2], 'n_selection': counts_I2}
        ]

        sum_counts_M = sum(counts_M)
        while sum_counts_M > counts_I2 and len(M_indices) > 0:
            # Find class with the highest count in M
            max_count_idx = counts_M.index(max(counts_M))
            max_class = M_indices.pop(max_count_idx)
            max_count = counts_M.pop(max_count_idx)
            # Add this class as a new bucket
            buckets.append({'labels': [max_class], 'n_selection': max_count})
            # Update the sum of counts in M
            sum_counts_M = sum(counts_M)

        # Add the remaining classes in M as a bucket if any
        if len(M_indices) > 0:
            buckets.append({'labels': M_indices, 'n_selection': sum_counts_M})

        # Proceed with estimation counts and confidence bounds as before
        # ...
         # Now, compute n_estimation for each bucket from counts_estimation
        counts_estimation_np = np.array(counts_estimation)
        for bucket in buckets:
            labels = bucket['labels']
            n_estimation = counts_estimation_np[labels].sum()
            bucket['n_estimation'] = n_estimation

        num_buckets = len(buckets)

        if num_buckets < 2:
            mode = "Rmono"

        # Adjust alpha for multiple comparisons
        alpha_prime = self.get_alpha_prime(alpha, factor=num_buckets, mode=mode)

        # Compute confidence bounds
        # Lower confidence bound for the first bucket (nA)
        nA = buckets[0]['n_estimation']
        pABar = self._lower_confidence_bound(nA, N, alpha_prime)

        # Upper confidence bounds for other buckets
        pBBar_list = []
        for bucket in buckets[1:]:
            n_bucket = bucket['n_estimation']
            pBBar = self._upper_confidence_bound(n_bucket, N, alpha_prime)
            pBBar_list.append(pBBar)


        # Take the maximum of the upper bounds
        pOtherBar = max(pBBar_list) if pBBar_list else 0

        # Compute the certified radius
        radius = self.get_radius(pABar, pOtherBar, mode=mode)

        if radius < 0.0:
            return Smooth.ABSTAIN, 0.0
        else:
            cAHat = buckets[0]['labels'][0]
            return cAHat, radius


    def certify_from_log_bonferroni(self, counts_selection, counts_estimation, N, N0, alpha, mode="R3", lip=None):

        self.base_classifier.eval()
        # estimate heap
        counts_selection = torch.tensor(counts_selection)
        _, idxs  = torch.sort(counts_selection, descending=True)

        cAHat = idxs[0].item()
           
        alpha_prime = self.get_alpha_prime(alpha, factor=self.num_classes, mode=mode)
        # we do bonferoni correction
        nA = counts_estimation[cAHat].item()

        pABar = self._lower_confidence_bound(nA, N, alpha_prime)

        # remove cAHat from counts_estimation
        counts_estimation_pBBars = np.delete(counts_estimation, cAHat)

        potential_pBBar = self._upper_confidence_bound_list(counts_estimation_pBBars, N, alpha_prime)

        pBBar = np.max(potential_pBBar)


        radius = self.get_radius(pABar, pBBar, mode=mode, lip=lip)

        if radius < 0.0:
            return Smooth.ABSTAIN, 0.0
        else:
            return cAHat, radius


    def certify_from_log(self, counts_selection, counts_estimation, n, alpha, mode="R3", lip=None):
        self.base_classifier.eval()
        cAHat = counts_selection.argmax().item()
        nA = counts_estimation[cAHat].item()
        pABar = self._lower_confidence_bound(nA, n, alpha)
        if pABar < 0.5:
            return Smooth.ABSTAIN, 0.0
        else:
            radius = self.sigma * norm.ppf(pABar)
            return cAHat, radius

    def certify(self, x: torch.tensor, n0: int, n: int, alpha: float, batch_size: int) -> (int, float):
        """ Monte Carlo algorithm for certifying that g's prediction around x is constant within some L2 radius.
        With probability at least 1 - alpha, the class returned by this method will equal g(x), and g's prediction will
        robust within a L2 ball of radius R around x.

        :param x: the input [channel x height x width]
        :param n0: the number of Monte Carlo samples to use for selection
        :param n: the number of Monte Carlo samples to use for estimation
        :param alpha: the failure probability
        :param batch_size: batch size to use when evaluating the base classifier
        :return: (predicted class, certified radius)
                 in the case of abstention, the class will be ABSTAIN and the radius 0.
        """
        self.base_classifier.eval()
        # draw samples of f(x+ epsilon)
        counts_selection = self._sample_noise(x, n0, batch_size)
        # use these samples to take a guess at the top class
        cAHat = counts_selection.argmax().item()
        # draw more samples of f(x + epsilon)
        counts_estimation = self._sample_noise(x, n, batch_size)
        # use these samples to estimate a lower bound on pA
        nA = counts_estimation[cAHat].item()
        pABar = self._lower_confidence_bound(nA, n, alpha)
        if pABar < 0.5:
            return Smooth.ABSTAIN, 0.0
        else:
            radius = self.sigma * norm.ppf(pABar)
            return cAHat, radius

    def log_certify(self, x: torch.tensor, n: int, batch_size: int):
        """ Monte Carlo algorithm for certifying that g's prediction around x is constant within some L2 radius.
        With probability at least 1 - alpha, the class returned by this method will equal g(x), and g's prediction will
        robust within a L2 ball of radius R around x.

        :param x: the input [channel x height x width]
        :param n0: the number of Monte Carlo samples to use for selection
        :param n: the number of Monte Carlo samples to use for estimation
        :param alpha: the failure probability
        :param batch_size: batch size to use when evaluating the base classifier
        :return: (predicted class, certified radius)
                 in the case of abstention, the class will be ABSTAIN and the radius 0.
        """
        self.base_classifier.eval()
        # draw samples of f(x+ epsilon)
        counts_selection = self._sample_noise(x, n, batch_size)
        # use these samples to take a guess at the top class
        return counts_selection

    def predict(self, x: torch.tensor, n: int, alpha: float, batch_size: int) -> int:
        """ Monte Carlo algorithm for evaluating the prediction of g at x.  With probability at least 1 - alpha, the
        class returned by this method will equal g(x).

        This function uses the hypothesis test described in https://arxiv.org/abs/1610.03944
        for identifying the top category of a multinomial distribution.

        :param x: the input [channel x height x width]
        :param n: the number of Monte Carlo samples to use
        :param alpha: the failure probability
        :param batch_size: batch size to use when evaluating the base classifier
        :return: the predicted class, or ABSTAIN
        """
        self.base_classifier.eval()
        counts = self._sample_noise(x, n, batch_size)
        top2 = counts.argsort()[::-1][:2]
        count1 = counts[top2[0]]
        count2 = counts[top2[1]]
        if binom_test(count1, count1 + count2, p=0.5) > alpha:
            return Smooth.ABSTAIN
        else:
            return top2[0]

    def _sample_noise(self, x: torch.tensor, num: int, batch_size) -> np.ndarray:
        """ Sample the base classifier's prediction under noisy corruptions of the input x.

        :param x: the input [channel x width x height]
        :param num: number of samples to collect
        :param batch_size:
        :return: an ndarray[int] of length num_classes containing the per-class counts
        """
        with torch.no_grad():
            counts = np.zeros(self.num_classes, dtype=int)
            for _ in range(ceil(num / batch_size)):
                this_batch_size = min(batch_size, num)
                num -= this_batch_size

                batch = x.repeat((this_batch_size, 1, 1, 1))
                noise = torch.randn_like(batch, device='cuda') * self.sigma
                predictions = self.base_classifier(batch + noise).argmax(1)
                counts += self._count_arr(predictions.cpu().numpy(), self.num_classes)
            return counts

    def _count_arr(self, arr: np.ndarray, length: int) -> np.ndarray:
        counts = np.zeros(length, dtype=int)
        for idx in arr:
            counts[idx] += 1
        return counts

    def _upper_confidence_bound(self, NA: int, N: int, alpha: float) -> float:
        """ Returns a (1 - alpha) lower confidence bound on a bernoulli proportion.

        This function uses the Clopper-Pearson method.

        :param NA: the number of "successes"
        :param N: the number of total draws
        :param alpha: the confidence level
        :return: a lower bound on the binomial proportion which holds true w.p at least (1 - alpha) over the samples
        """
        return proportion_confint(NA, N, alpha=2 * alpha, method="beta")[1]
    
    from statsmodels.stats.proportion import proportion_confint

    def _upper_confidence_bound_list(self, NAs: list, N: int, alpha: float) -> list:
        """ Returns a (1 - alpha) upper confidence bound on a bernoulli proportion for a list of "successes".

        This function uses the Clopper-Pearson method for each entry in the NA list.

        :param NA: a list of the number of "successes"
        :param N: the number of total draws (same for all NA)
        :param alpha: the confidence level
        :return: a list of upper bounds on the binomial proportions
        """
        # Apply proportion_confint to each value in the list NA
        return [proportion_confint(na, N, alpha=2 * alpha, method="beta")[1] for na in NAs]

    
    def _lower_confidence_bound(self, NA: int, N: int, alpha: float) -> float:
        """ Returns a (1 - alpha) lower confidence bound on a bernoulli proportion.

        This function uses the Clopper-Pearson method.

        :param NA: the number of "successes"
        :param N: the number of total draws
        :param alpha: the confidence level
        :return: a lower bound on the binomial proportion which holds true w.p at least (1 - alpha) over the samples
        """
        return proportion_confint(NA, N, alpha=2 * alpha, method="beta")[0]
