import math
import functools
from functools import partial
import torch
from torch import Tensor, Size
from torch.distributions.gamma import Gamma
from torch.distributions.gumbel import Gumbel
from torch import Tensor

from abc import ABC, abstractmethod
from typing import Callable, Optional

class BaseNoiseDistribution(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def sample(self,
               shape: Size) -> Tensor:
        raise NotImplementedError
    
class BaseTargetDistribution(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def params(self,
               theta: Tensor,
               dy: Tensor) -> Tensor:
        raise NotImplementedError


class TargetDistribution(BaseTargetDistribution):
    r"""
    Creates a generator of target distributions parameterized by :attr:`alpha` and :attr:`beta`.

    Example::

        >>> import torch
        >>> target_distribution = TargetDistribution(alpha=1.0, beta=1.0)
        >>> target_distribution.params(theta=torch.tensor([1.0]), dy=torch.tensor([1.0]))
        tensor([2.])

    Args:
        alpha (float): weight of the initial distribution parameters theta
        beta (float): weight of the downstream gradient dy
    """
    def __init__(self,
                 alpha: float = 1.0,
                 beta: float = 1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta

    def params(self,
               theta: Tensor,
               dy: Tensor) -> Tensor:
        theta_prime = self.alpha * theta - self.beta * dy
        return theta_prime


class SumOfGammaNoiseDistribution(BaseNoiseDistribution):
    r"""
    Creates a generator of samples for the Sum-of-Gamma distribution [1], parameterized
    by :attr:`k`, :attr:`nb_iterations`, and :attr:`device`.

    [1] Mathias Niepert, Pasquale Minervini, Luca Franceschi - Implicit MLE: Backpropagating Through Discrete
    Exponential Family Distributions. NeurIPS 2021 (https://arxiv.org/abs/2106.01798)

    Example::

        >>> import torch
        >>> noise_distribution = SumOfGammaNoiseDistribution(k=5, nb_iterations=100)
        >>> noise_distribution.sample(torch.Size([5]))
        tensor([ 0.2504,  0.0112,  0.5466,  0.0051, -0.1497])

    Args:
        k (float): k parameter -- see [1] for more details.
        nb_iterations (int): number of iterations for estimating the sample.
        device (torch.devicde): device where to store samples.
    """
    def __init__(self,
                 k: float,
                 nb_iterations: int = 10,
                 device: Optional[torch.device] = None):
        super().__init__()
        self.k = k
        self.nb_iterations = nb_iterations
        self.device = device

    def sample(self,
               shape: Size) -> Tensor:
        samples = torch.zeros(size=shape, device=self.device)
        for i in range(1, self.nb_iterations + 1):
            concentration = torch.tensor(1. / self.k, device=self.device)
            rate = torch.tensor(i / self.k, device=self.device)

            gamma = Gamma(concentration=concentration, rate=rate)
            samples = samples + gamma.sample(sample_shape=shape).to(self.device)
        samples = (samples - math.log(self.nb_iterations)) / self.k
        return samples.to(self.device)


class GumbelDistribution(BaseNoiseDistribution):
    def __init__(self, loc: float = 0., scale: float = 1.0, device: torch.device = 'cpu'):
        super().__init__()
        self.loc = loc
        self._scale = scale
        self.device = device

    @property
    def scale(self):
        return self._scale

    @scale.setter
    def scale(self, value):
        self._scale = value

    def sample(self, shape: Size) -> Tensor:
        gumbel = Gumbel(loc=self.loc, scale=self.scale)
        samples = gumbel.sample(shape).to(self.device)
        return samples    


class IMLESampler:
    def __init__(self, algorithm='round', beta=0.1, noise_scale=1, device: torch.device = 'cpu'):
        self.algorithm = instantiate_algorithm(algorithm)
        self.beta = beta
        self.noise_scale = noise_scale
        self.device = device

    def __call__(self, scores: torch.Tensor, times_sampled: int, *args):
        imle_algo = imle(
            self.algorithm,
            target_distribution=TargetDistribution(alpha=1.0, beta=self.beta),
            noise_distribution=GumbelDistribution(0., self.noise_scale, scores.device),
            nb_samples=times_sampled,
            input_noise_temperature=0.16,
            target_noise_temperature=0.16,
        )
        return imle_algo(scores, *args)


def instantiate_algorithm(algorithm):
        return round


def round(logits, *args):
    with torch.no_grad():
        return torch.where(logits >= 0.5, 1., 0.), None    


def imle(function: Callable[[Tensor], Tensor] = None,
         target_distribution: Optional[BaseTargetDistribution] = None,
         noise_distribution: Optional[BaseNoiseDistribution] = None,
         nb_samples: int = 1,
         input_noise_temperature: float = 1.0,
         target_noise_temperature: float = 1.0):
    r"""Turns a black-box combinatorial solver in an Exponential Family distribution via Perturb-and-MAP and I-MLE [1].

    The input function (solver) needs to return the solution to the problem of finding a MAP state for a constrained
    exponential family distribution -- this is the case for most black-box combinatorial solvers [2]. If this condition
    is violated though, the result would not hold and there is no guarantee on the validity of the obtained gradients.

    This function can be used directly or as a decorator.

    [1] Mathias Niepert, Pasquale Minervini, Luca Franceschi - Implicit MLE: Backpropagating Through Discrete
    Exponential Family Distributions. NeurIPS 2021 (https://arxiv.org/abs/2106.01798)
    [2] Marin Vlastelica, Anselm Paulus, Vít Musil, Georg Martius, Michal Rolínek - Differentiation of Blackbox
    Combinatorial Solvers. ICLR 2020 (https://arxiv.org/abs/1912.02175)

    Example::

        >>> from graip.imle import imle
        >>> from graip.imle import TargetDistribution
        >>> from graip.imle import SumOfGammaNoiseDistribution
        >>> target_distribution = TargetDistribution(alpha=0.0, beta=10.0)
        >>> noise_distribution = SumOfGammaNoiseDistribution(k=21, nb_iterations=100)
        >>> @imle(target_distribution=target_distribution, noise_distribution=noise_distribution, nb_samples=100,
        >>>       input_noise_temperature=input_noise_temperature, target_noise_temperature=5.0)
        >>> def imle_solver(weights_batch: Tensor) -> Tensor:
        >>>     return torch_solver(weights_batch)

    Args:
        function (Callable[[Tensor], Tensor]): black-box combinatorial solver
        target_distribution (Optional[BaseTargetDistribution]): factory for target distributions
        noise_distribution (Optional[BaseNoiseDistribution]): noise distribution
        nb_samples (int): number of noise sammples
        input_noise_temperature (float): noise temperature for the input distribution
        target_noise_temperature (float): noise temperature for the target distribution
    """
    if target_distribution is None:
        target_distribution = TargetDistribution(alpha=1.0, beta=1.0)

    if function is None:
        return functools.partial(imle,
                                 target_distribution=target_distribution,
                                 noise_distribution=noise_distribution,
                                 nb_samples=nb_samples,
                                 input_noise_temperature=input_noise_temperature,
                                 target_noise_temperature=target_noise_temperature)

    @functools.wraps(function)
    def wrapper(input: Tensor, *args):
        class WrappedFunc(torch.autograd.Function):

            @staticmethod
            def forward(ctx, input: Tensor, *args):
                # [BATCH_SIZE, ...]
                input_shape = input.shape
                dims = input.dim()

                batch_size = input_shape[0]
                instance_shape = input_shape[1:]

                # (B x n_sample x N x N x E) or (B x n_sample x N x E)
                perturbed_input_shape = [batch_size, nb_samples] + list(instance_shape)

                if noise_distribution is None:
                    noise = torch.zeros(size=perturbed_input_shape)
                else:
                    noise = noise_distribution.sample(shape=torch.Size(perturbed_input_shape))

                input_noise = noise * input_noise_temperature

                repeats = [1] * len(perturbed_input_shape)
                repeats[1] = nb_samples
                perturbed_input_3d = input[:, None, ...].repeat(repeats).view(perturbed_input_shape)
                perturbed_input_3d = perturbed_input_3d + input_noise

                # [BATCH_SIZE * N_SAMPLES, ...]
                perturbed_input_2d = perturbed_input_3d.view([-1] + perturbed_input_shape[2:])

                # [BATCH_SIZE * N_SAMPLES, ...]
                perturbed_output, aux_outputs = function(perturbed_input_2d, *args)
                # [BATCH_SIZE, N_SAMPLES, ...]
                perturbed_output = perturbed_output.view(perturbed_input_shape)

                ctx.save_for_backward(input, noise, perturbed_output)
                ctx.args = args

                # [BATCH_SIZE * N_SAMPLES, ...]
                res = perturbed_output.transpose(0, 1)
                return res, aux_outputs

            @staticmethod
            def backward(ctx, dy, _):
                # the grad of the second input is None, as it is not differentiable
                # input: B x N x N x E
                # noise: B x VE x N x N x E
                # perturbed_output_3d: B x VE x N x N x E
                input, noise, perturbed_output_3d = ctx.saved_variables
                args = ctx.args

                input_shape = input.shape

                dy = dy.transpose(0, 1)
                # B x VE x N x N x E
                dy_shape = dy.shape
                # B x VE x N x N x E
                noise_shape = noise.shape

                repeats = [1] * len(noise_shape)
                repeats[1] = nb_samples
                input_2d = input[:, None, ...].repeat(repeats).view(dy_shape)
                target_input_2d = target_distribution.params(input_2d, dy)

                # [BATCH_SIZE, NB_SAMPLES, ...]
                target_input_3d = target_input_2d.view(noise_shape)

                # [BATCH_SIZE, NB_SAMPLES, ...]
                target_noise = noise * target_noise_temperature

                # [BATCH_SIZE, N_SAMPLES, ...]
                perturbed_target_input_3d = target_input_3d + target_noise

                # [BATCH_SIZE * N_SAMPLES, ...]
                perturbed_target_input_2d = perturbed_target_input_3d.view((-1,) + input_shape[1:])

                # [BATCH_SIZE * N_SAMPLES, ...]
                target_output_2d, _ = function(perturbed_target_input_2d, *args)

                # [BATCH_SIZE, N_SAMPLES, ...]
                target_output_3d = target_output_2d.view(noise_shape)

                # [BATCH_SIZE, ...]
                gradient = (perturbed_output_3d - target_output_3d)
                gradient = gradient.mean(axis=1)
                # the gradient of the second input is None
                return gradient, None

        return WrappedFunc.apply(input, *args)
    return wrapper
