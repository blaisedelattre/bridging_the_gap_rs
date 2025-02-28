from math import ceil
from statsmodels.stats.proportion import proportion_confint
import torch
from scipy.stats import norm, binom_test
import torch.nn as nn
from sparsemax import Sparsemax
import numpy as np


class Smooth(object):
    """A smoothed classifier g"""

    # to abstain, Smooth returns this int
    ABSTAIN = -1

    def __init__(
        self, base_classifier: torch.nn.Module, num_classes: int, sigma: float, t: int
    ):
        """
        :param base_classifier: maps from [batch x channel x height x width] to [batch x num_classes]
        :param num_classes:
        :param sigma: the noise level hyperparameter
        """
        self.base_classifier = base_classifier
        self.num_classes = num_classes
        self.t = t
        self.sigma = sigma
        self.simplex_layer_tmp = [1.0] + list(np.linspace(0.1, 50, 100)) + [10000000]
        self.simplex_layers = [Sparsemax(), nn.Softmax(dim=1)]

    def _sample_noise_sparsemax_reg(
        self, x: torch.tensor, num: int, batch_size
    ) -> np.ndarray:
        """Sample the base classifier's prediction under noisy corruptions of the input x.
        :param x: the input [channel x width x height]
        :param num: number of samples to collect
        :param batch_size:
        :return: an ndarray[int] of length num_classes containing the per-class counts
        """
        total_num = num
        len_proj_simplex = len(self.simplex_layers) * len(self.simplex_layer_tmp)
        with torch.no_grad():
            average = np.zeros((len_proj_simplex, self.num_classes))
            variances = np.zeros((len_proj_simplex, self.num_classes))
            counts = np.zeros(self.num_classes, dtype=int)
            batch_sizes = []
            for _ in range(ceil(num / batch_size)):
                this_batch_size = min(batch_size, num)
                batch_sizes.append(this_batch_size)
                num -= this_batch_size
                batch = x.repeat((this_batch_size, 1, 1, 1))
                predictions = self.base_classifier(batch, self.t)
                argmax_predictions = predictions.argmax(1)
                counts += self._count_arr(
                    argmax_predictions.cpu().numpy(), self.num_classes
                )
                # compute max layer for different temperature
                for idx_tmp, tmp in enumerate(self.simplex_layer_tmp):
                    for idx_simplex_layer, simplex_layer in enumerate(
                        self.simplex_layers
                    ):
                        idx = idx_simplex_layer * len(self.simplex_layer_tmp) + idx_tmp
                        simplex_predictions = simplex_layer(
                            predictions * torch.tensor(tmp)
                        )
                        variances[idx] += (
                            simplex_predictions.var(dim=0).cpu().numpy()
                            * this_batch_size
                        )
                        average[idx] += simplex_predictions.sum(dim=0).cpu().numpy()
            average /= total_num
            variances /= np.sum(batch_sizes)
            stds = np.sqrt(variances)
            return average, stds, counts

    def certify_ours(
        self, x: torch.tensor, n0: int, n: int, alpha: float, batch_size: int
    ) -> (int, float):
        """Monte Carlo algorithm for certifying that g's prediction around x is constant within some L2 radius.
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
        radii = []
        cAHats = []
        self.base_classifier.eval()
        outputs, stds, counts_estimation = self._sample_noise_sparsemax_reg(
            x, n, batch_size
        )
        indices = np.argsort(counts_estimation)
        cBHat_argmax = indices[-2]
        # argmax with pearson clopper
        lowers_p, uppers_p = proportion_confint(
            counts_estimation, n, alpha=2 * alpha / self.num_classes, method="beta"
        )
        cBHat_argmax = np.argsort(uppers_p)[-2]
        cAHat_argmax = np.argmax(lowers_p)
        pABar_argmax = lowers_p[cAHat_argmax]
        pBBar_argmax = uppers_p[cBHat_argmax]
        if pABar_argmax > 0:
            # R2
            radius_argmax = self.get_R2(pABar_argmax, pBBar_argmax)
        else:
            radius_argmax = 0.0
        radii.append(radius_argmax)
        cAHats.append(cAHat_argmax)
        # others simplex projections with bernstein
        for idx, (p, std) in enumerate(zip(outputs, stds)):
            shifts = self.get_shift(alpha / self.num_classes, n, std)
            uppers_p = p + shifts
            lowers_p = p - shifts
            cAHat = np.argmax(lowers_p)
            cBHat = np.argsort(uppers_p)[-2]
            pABar = lowers_p[cAHat]
            pBBar = uppers_p[cBHat]
            if pABar > 0:
                radius = self.get_R2(pABar, pBBar)
                radii.append(radius)
                cAHats.append(cAHat)
                lip = 1 / np.sqrt(np.pi * self.sigma**2)
                radius2 = (pABar - pBBar) / (lip * np.sqrt(2))
                radii.append(radius2)
                cAHats.append(cAHat)
        id_max = np.argmax(radii)
        radius = radii[id_max]
        cAHat = cAHats[id_max]
        if radius < 0:
            return Smooth.ABSTAIN, 0.0
        else:
            return cAHat, radius

    def predict(self, x: torch.tensor, n: int, alpha: float, batch_size: int) -> int:
        """Monte Carlo algorithm for evaluating the prediction of g at x.  With probability at least 1 - alpha, the
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

    def predict_ours(
        self, x: torch.tensor, n: int, alpha: float, batch_size: int
    ) -> int:
        """Monte Carlo algorithm for evaluating the prediction of g at x.  With probability at least 1 - alpha, the
        class returned by this method will equal g(x).

        This function uses the hypothesis test described in https://arxiv.org/abs/1610.03944
        for identifying the top category of a multinomial distribution.

        :param x: the input [channel x height x width]
        :param n: the number of Monte Carlo samples to use
        :param alpha: the failure probability
        :param batch_size: batch size to use when evaluating the base classifier
        :return: the predicted class, or ABSTAIN
        """
        radii = []
        cAHats = []
        self.base_classifier.eval()
        outputs, stds, counts_estimation = self._sample_noise_sparsemax_reg(
            x, n, batch_size
        )
        indices = np.argsort(counts_estimation)
        cBHat_argmax = indices[-2]
        # argmax with pearson clopper
        lowers_p, uppers_p = proportion_confint(
            counts_estimation, n, alpha=2 * alpha / self.num_classes, method="beta"
        )
        cBHat_argmax = np.argsort(uppers_p)[-2]
        cAHat_argmax = np.argmax(lowers_p)
        pABar_argmax = lowers_p[cAHat_argmax]
        pBBar_argmax = uppers_p[cBHat_argmax]
        if pABar_argmax > 0:
            # R2
            radius_argmax = self.get_R2(pABar_argmax, pBBar_argmax)
        else:
            radius_argmax = 0.0
        radii.append(radius_argmax)
        cAHats.append(cAHat_argmax)
        # others simplex projections with bernstein
        for idx, (p, std) in enumerate(zip(outputs, stds)):
            shifts = self.get_shift(alpha / self.num_classes, n, std)
            uppers_p = p + shifts
            lowers_p = p - shifts
            cAHat = np.argmax(lowers_p)
            cBHat = np.argsort(uppers_p)[-2]
            pABar = lowers_p[cAHat]
            pBBar = uppers_p[cBHat]
            if pABar > 0:
                # R2
                radius = self.get_R2(pABar, pBBar)
                radii.append(radius)
                cAHats.append(cAHat)
                # R2
                lip = 1 / np.sqrt(np.pi * self.sigma**2)
                radius2 = (pABar - pBBar) / (lip * np.sqrt(2))
                radii.append(radius2)
                cAHats.append(cAHat)
        id_max = np.argmax(radii)
        radius = radii[id_max]
        cAHat = cAHats[id_max]
        if radius < 0:
            return Smooth.ABSTAIN
        else:
            return cAHat

    def predict(self, x: torch.tensor, n: int, alpha: float, batch_size: int) -> int:
        """Monte Carlo algorithm for evaluating the prediction of g at x.  With probability at least 1 - alpha, the
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

    def get_R2(self, pA, pB):
        radius = (norm.ppf(pA) - norm.ppf(pB)) * self.sigma / 2
        return radius

    def get_R3(self, pA):
        radius = norm.ppf(pA) * self.sigma
        return radius

    def _count_arr(self, arr: np.ndarray, length: int) -> np.ndarray:
        counts = np.zeros(length, dtype=int)
        for idx in arr:
            counts[idx] += 1
        return counts

    def get_shift(self, alpha, n, std):
        val_bernstein = std * np.sqrt(2 * np.log(2 / alpha) / n) + 7 * np.log(
            2 / alpha
        ) / (3 * (n - 1))
        return val_bernstein

    def _upper_confidence_bound(self, NA: int, N: int, alpha: float) -> float:
        """Returns a (1 - alpha) lower confidence bound on a bernoulli proportion.

        This function uses the Clopper-Pearson method.

        :param NA: the number of "successes"
        :param N: the number of total draws
        :param alpha: the confidence level
        :return: a lower bound on the binomial proportion which holds true w.p at least (1 - alpha) over the samples
        """
        return proportion_confint(NA, N, alpha=2 * alpha, method="beta")[1]

    def _lower_confidence_bound(self, NA: int, N: int, alpha: float) -> float:
        """Returns a (1 - alpha) lower confidence bound on a bernoulli proportion.

        This function uses the Clopper-Pearson method.

        :param NA: the number of "successes"
        :param N: the number of total draws
        :param alpha: the confidence level
        :return: a lower bound on the binomial proportion which holds true w.p at least (1 - alpha) over the samples
        """
        return proportion_confint(NA, N, alpha=2 * alpha, method="beta")[0]
